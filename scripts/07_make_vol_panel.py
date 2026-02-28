from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "panel_daily.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"

MIN_DATE = pd.Timestamp("2016-01-01")
MAX_DATE = pd.Timestamp("2025-12-31")

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Restrict to paper window
    df = df[(df["Date"] >= MIN_DATE) & (df["Date"] <= MAX_DATE)].copy()

    required = ["TPNSI", "VIX_z", "BTCUSD_RV5", "DAX_RV5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in panel_daily.csv: {missing}")

    before = len(df)
    df2 = df.dropna(subset=required).copy()
    after = len(df2)

    df2.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows before:", before)
    print("Rows after :", after)
    print("Date range :", df2['Date'].min(), "to", df2['Date'].max())
    print("TPNSI non-null:", int(df2["TPNSI"].notna().sum()))
    print("BTC RV5 non-null:", int(df2["BTCUSD_RV5"].notna().sum()))
    print("DAX RV5 non-null:", int(df2["DAX_RV5"].notna().sum()))

if __name__ == "__main__":
    main()