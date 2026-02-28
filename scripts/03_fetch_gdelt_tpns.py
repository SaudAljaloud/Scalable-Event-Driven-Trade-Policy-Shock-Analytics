import time
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "raw" / "gdelt"
CACHE_DIR = OUT_DIR / "cache_tpns"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Try both hostnames (some networks treat them differently)
GDELT_HOSTS = [
    "https://api.gdeltproject.org/api/v2/doc/doc",
    "https://api.gdeltproject.org/api/v2/doc/doc"  # keep same but allows easy swap later
]

QUERY = (
    '(tariff OR "trade war" OR "import duty" OR "customs duty" OR '
    '"trade retaliation" OR "export restriction" OR protectionism)'
)

@dataclass
class GdeltRequest:
    mode: str
    startdatetime: str
    enddatetime: str
    format: str = "json"

def month_starts(start="2016-01-01", end="2026-12-01"):
    return pd.date_range(start=start, end=end, freq="MS")

def dtfmt(ts: pd.Timestamp, end=False) -> str:
    if end:
        return ts.strftime("%Y%m%d") + "235959"
    return ts.strftime("%Y%m%d") + "000000"

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "tpns-bdcc-research/1.0"})

    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",)
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def call_gdelt(session: requests.Session, endpoint: str, req: GdeltRequest) -> dict:
    params = {
        "query": QUERY,
        "mode": req.mode,
        "format": req.format,
        "startdatetime": req.startdatetime,
        "enddatetime": req.enddatetime,
    }
    url = f"{endpoint}?{urlencode(params)}"

    # Much more generous timeouts
    r = session.get(url, timeout=(60, 240))
    r.raise_for_status()
    return r.json()

def parse_timeline(json_obj: dict) -> pd.DataFrame:
    timeline = json_obj.get("timeline", [])
    if not timeline:
        return pd.DataFrame(columns=["Date", "Value"])
    df = pd.DataFrame(timeline)
    df["Date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 8), format="%Y%m%d")
    df = df.rename(columns={"value": "Value"})[["Date", "Value"]]
    return df.groupby("Date", as_index=False)["Value"].sum()

def parse_sourcecountry(json_obj: dict) -> pd.DataFrame:
    timeline = json_obj.get("timeline", [])
    if not timeline:
        return pd.DataFrame()
    df = pd.DataFrame(timeline)
    df["Date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 8), format="%Y%m%d")
    df = df.drop(columns=["date"])
    cols = ["Date"] + [c for c in df.columns if c != "Date"]
    df = df[cols].groupby("Date", as_index=False).sum()
    return df

def cache_path(mode: str, startdt: str, enddt: str) -> Path:
    return CACHE_DIR / f"{mode}_{startdt}_{enddt}.json"

def fetch_month(session: requests.Session, endpoint: str, mode: str, startdt: str, enddt: str) -> dict:
    cp = cache_path(mode, startdt, enddt)
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            return json.load(f)

    data = call_gdelt(session, endpoint, GdeltRequest(mode, startdt, enddt))
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f)

    time.sleep(1.0)
    return data

def main():
    session = make_session()

    # Quick connectivity test (single lightweight request)
    print("Testing GDELT connectivity...")
    test_start = "20160101000000"
    test_end = "20160101235959"
    ok = False
    for endpoint in GDELT_HOSTS:
        try:
            _ = call_gdelt(session, endpoint, GdeltRequest("TimelineVolRaw", test_start, test_end))
            print("Connected OK to:", endpoint)
            ok = True
            chosen = endpoint
            break
        except Exception as e:
            print("Failed endpoint:", endpoint, "->", type(e).__name__, e)

    if not ok:
        raise RuntimeError(
            "Cannot reach GDELT DOC endpoint from this network. "
            "See Plan B below (event DB route)."
        )

    vol_frames, tone_frames, sc_frames = [], [], []

    months = month_starts("2016-01-01", "2026-12-01")
    for m in months:
        start = m
        end = m + pd.offsets.MonthEnd(1)
        startdt = dtfmt(start, end=False)
        enddt = dtfmt(end, end=True)

        print(f"Month {start.strftime('%Y-%m')}")

        print("  TimelineVolRaw")
        vol_json = fetch_month(session, chosen, "TimelineVolRaw", startdt, enddt)
        vol_frames.append(parse_timeline(vol_json).rename(columns={"Value": "tp_count"}))

        print("  TimelineTone")
        tone_json = fetch_month(session, chosen, "TimelineTone", startdt, enddt)
        tone_frames.append(parse_timeline(tone_json).rename(columns={"Value": "tp_tone"}))

        print("  TimelineSourceCountry")
        sc_json = fetch_month(session, chosen, "TimelineSourceCountry", startdt, enddt)
        sc_df = parse_sourcecountry(sc_json)
        if not sc_df.empty:
            sc_frames.append(sc_df)

    vol = pd.concat(vol_frames, ignore_index=True).drop_duplicates("Date").sort_values("Date")
    tone = pd.concat(tone_frames, ignore_index=True).drop_duplicates("Date").sort_values("Date")
    sc = pd.concat(sc_frames, ignore_index=True) if sc_frames else pd.DataFrame(columns=["Date"])
    if not sc.empty:
        sc = sc.sort_values("Date")

    vol.to_csv(OUT_DIR / "tp_timeline_volraw.csv", index=False)
    tone.to_csv(OUT_DIR / "tp_timeline_tone.csv", index=False)
    sc.to_csv(OUT_DIR / "tp_timeline_sourcecountry.csv", index=False)

    print("Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()