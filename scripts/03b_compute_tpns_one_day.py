import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Could not locate project root (expected data/ and scripts/).")

def entropy(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    return float(-(p * np.log(p)).sum())

def main():
    base = find_project_root(Path(__file__).resolve().parent)

    matches = list(base.rglob("20260222.export.CSV.zip"))
    if not matches:
        raise FileNotFoundError("Could not find 20260222.export.CSV.zip anywhere under the project.")
    zip_path = matches[0]
    print("Using ZIP:", zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        inner = z.namelist()[0]
        raw = z.read(inner)

    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, low_memory=False)
    print("Columns:", df.shape[1])
    if df.shape[1] != 58:
        raise ValueError(f"Expected 58 columns, got {df.shape[1]}")

    # Column indices (0-based) for the daily export:
    # 1=SQLDATE, 26=EventCode, 30=GoldsteinScale, 53=ActionGeo_CountryCode
    sqldate = df.iloc[:, 1]
    eventcode = df.iloc[:, 26]
    goldstein = df.iloc[:, 30]
    actiongeo_cc = df.iloc[:, 53]

    # Trade policy proxy: start with EventCode 163
    CODES = {"163"}
    mask = eventcode.isin(CODES)

    day = pd.to_datetime(str(sqldate.iloc[0])[:8], format="%Y%m%d", errors="coerce")

    tp_count = int(mask.sum())

    gold = pd.to_numeric(goldstein[mask], errors="coerce")
    tp_goldstein_mean = float(gold.mean(skipna=True)) if gold.notna().any() else np.nan

    tp_dispersion_entropy = entropy(actiongeo_cc[mask]) if tp_count > 0 else np.nan

    out = pd.DataFrame([{
        "Date": day,
        "tp_count": tp_count,
        "tp_goldstein_mean": tp_goldstein_mean,
        "tp_dispersion_entropy": tp_dispersion_entropy
    }])

    print(out.to_string(index=False))

if __name__ == "__main__":
    main()