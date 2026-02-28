# scripts/03c_build_tpns_daily_from_events.py
#
# Build daily Trade Policy News Shock (TPNS) components from GDELT daily Events export ZIPs.
# - Downloads (or uses cached) daily files: YYYYMMDD.export.CSV.zip from GDELT raw file server
# - Filters events by a trade-policy proxy CAMEO EventCode set (default: {"163"})
# - Aggregates to daily:
#     tp_count                = number of filtered events
#     tp_tone_mean            = mean AvgTone of filtered events (time-varying sentiment proxy)
#     tp_dispersion_entropy   = entropy of ActionGeo_CountryCode distribution (geographic spread)
# - Saves: data/raw/gdelt/tp_events_daily.csv
#
# Note: Uses HTTP to avoid SSL interception issues you encountered.
#       Cached zips stored in: data/raw/gdelt/cache_events_zips/

import io
import zipfile
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "raw" / "gdelt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUT_DIR / "cache_events_zips"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "tp_events_daily.csv"

# -----------------------------
# Data source (HTTP avoids your SSL hostname mismatch)
# -----------------------------
BASE_URL = "http://data.gdeltproject.org/events"

# -----------------------------
# Date range
# -----------------------------
START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 2, 22)  # matches what you already downloaded up to

# -----------------------------
# Trade policy proxy (start narrow; can expand later)
# 163 ~ embargo/boycott/sanctions (trade restriction proxy)
# -----------------------------
CODES = {"163"}

# -----------------------------
# GDELT daily Events export is 58 columns (0-based indices)
# We only need a few stable columns:
#   SQLDATE               -> 1   (not used for Date assignment; we use loop day)
#   EventCode             -> 26
#   AvgTone               -> 34
#   ActionGeo_CountryCode -> 53
# -----------------------------
IDX_EVENTCODE = 26
IDX_AVGTONE = 34
IDX_ACTIONGEO_CC = 53


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def entropy(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    return float(-(p * np.log(p)).sum())


def download_zip(session: requests.Session, day: date) -> Path:
    """
    Download one day's zip if not already cached.
    """
    fname = day.strftime("%Y%m%d") + ".export.CSV.zip"
    local = CACHE_DIR / fname
    if local.exists() and local.stat().st_size > 0:
        return local

    url = f"{BASE_URL}/{fname}"
    r = session.get(url, timeout=(60, 300))
    r.raise_for_status()
    local.write_bytes(r.content)

    # gentle throttle
    time.sleep(0.25)
    return local


def read_zip_to_df(zip_path: Path) -> pd.DataFrame:
    """
    Read the tab-delimited inner CSV from the zip into a DataFrame with 58 unnamed columns.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        inner = z.namelist()[0]
        raw = z.read(inner)

    df = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        dtype=str,
        low_memory=False,
    )
    return df


def process_day(zip_path: Path, day: date, debug: bool = False) -> dict | None:
    """
    Compute TPNS components for one day from its zip.
    Returns dict with Date + components, or None if format unexpected.
    """
    df = read_zip_to_df(zip_path)

    if df.shape[1] != 58:
        return None

    eventcode = df.iloc[:, IDX_EVENTCODE]
    avgtone = df.iloc[:, IDX_AVGTONE]
    actiongeo_cc = df.iloc[:, IDX_ACTIONGEO_CC]

    mask = eventcode.isin(CODES)
    tp_count = int(mask.sum())

    tone = pd.to_numeric(avgtone[mask], errors="coerce")
    tp_tone_mean = float(tone.mean(skipna=True)) if tone.notna().any() else np.nan

    tp_dispersion_entropy = entropy(actiongeo_cc[mask]) if tp_count > 0 else np.nan

    if debug:
        # diagnostics to confirm AvgTone looks sensible and varies
        t_all = pd.to_numeric(avgtone, errors="coerce")
        print("DEBUG one-day check:", day)
        print("  AvgTone overall (non-NA) count:", int(t_all.notna().sum()))
        if t_all.notna().any():
            print("  AvgTone overall min/mean/max:", float(t_all.min()), float(t_all.mean()), float(t_all.max()))
        print("  Filtered tp_count:", tp_count)
        if tone.notna().any():
            print("  Filtered AvgTone min/mean/max:", float(tone.min()), float(tone.mean()), float(tone.max()))

    return {
        "Date": pd.to_datetime(day),
        "tp_count": tp_count,
        "tp_tone_mean": tp_tone_mean,
        "tp_dispersion_entropy": tp_dispersion_entropy,
    }


def main():
    print(f"Building TPNS daily from GDELT Events: {START_DATE} -> {END_DATE}")
    print("Cache dir:", CACHE_DIR)
    print("Output:", OUT_PATH)
    print("Event codes:", sorted(CODES))
    print("Base URL:", BASE_URL)

    session = requests.Session()
    session.headers.update({"User-Agent": "tpns-bdcc-research/1.0"})

    rows: list[dict] = []
    total = (END_DATE - START_DATE).days + 1
    debug_done = False

    for i, day in enumerate(daterange(START_DATE, END_DATE), start=1):
        if i % 30 == 0 or i == 1:
            print(f"Progress: {i}/{total} (current {day})")

        try:
            zp = download_zip(session, day)
            rec = process_day(zp, day, debug=(not debug_done))
            debug_done = True

            if rec is None:
                print("Skipping (unexpected format):", day)
                continue

            rows.append(rec)

        except requests.HTTPError as e:
            # Some days can be missing; skip.
            print("HTTPError:", day, e)
            continue
        except Exception as e:
            # Keep going; you can rerun if needed
            print("Error:", day, type(e).__name__, e)
            continue

        # Periodic save so progress isn't lost
        if i % 14 == 0:
            pd.DataFrame(rows).sort_values("Date").to_csv(OUT_PATH, index=False)

    pd.DataFrame(rows).sort_values("Date").to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)

    # Quick sanity print
    df_out = pd.read_csv(OUT_PATH)
    print("Rows written:", len(df_out))
    print("Top 3 rows:")
    print(df_out.head(3).to_string(index=False))
    print("Bottom 3 rows:")
    print(df_out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()