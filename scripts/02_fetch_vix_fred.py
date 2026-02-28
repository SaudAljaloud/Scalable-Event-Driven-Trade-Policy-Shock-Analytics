from pathlib import Path
import pandas as pd
import requests
from io import StringIO

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "raw" / "vix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_VIX_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

def main():
    print("Downloading VIX from FRED...")

    response = requests.get(FRED_VIX_CSV, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    print("Columns received:", df.columns.tolist())

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # handle FRED variants
    date_candidates = ["date", "observation_date", "observation date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No recognizable date column found. Columns: {df.columns.tolist()}")

    # detect value column (prefer vixcls if present)
    if "vixcls" in df.columns:
        value_col = "vixcls"
    else:
        # fallback: first non-date column
        non_date_cols = [c for c in df.columns if c != date_col]
        if not non_date_cols:
            raise ValueError("No value column found in FRED response.")
        value_col = non_date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.rename(columns={date_col: "Date", value_col: "VIX"})
    df = df[["Date", "VIX"]].sort_values("Date").reset_index(drop=True)

    out_path = OUT_DIR / "vix_fred.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()