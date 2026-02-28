from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MARKETS_PATH = BASE_DIR / "data" / "raw" / "markets" / "markets_merged.csv"
SP_PATH = BASE_DIR / "data" / "raw" / "markets" / "sp500.csv"

def main():
    if not MARKETS_PATH.exists():
        raise FileNotFoundError(f"Missing: {MARKETS_PATH}")
    if not SP_PATH.exists():
        raise FileNotFoundError(f"Missing: {SP_PATH}")

    markets = pd.read_csv(MARKETS_PATH)
    sp = pd.read_csv(SP_PATH)

    markets["Date"] = pd.to_datetime(markets["Date"], errors="coerce")
    sp["Date"] = pd.to_datetime(sp["Date"], errors="coerce")

    markets = markets.dropna(subset=["Date"]).sort_values("Date")
    sp = sp.dropna(subset=["Date"]).sort_values("Date")

    merged = markets.merge(sp, on="Date", how="left")

    merged.to_csv(MARKETS_PATH, index=False)

    print("Updated:", MARKETS_PATH)
    print("SP500 non-null:", int(merged["SP500_Close"].notna().sum()))
    print("Columns now:", merged.columns.tolist())

if __name__ == "__main__":
    main()