from pathlib import Path
import yfinance as yf
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "raw" / "markets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2016-01-01"
END = "2025-12-31"

def main():
    print("Downloading S&P 500 (^GSPC) from Yahoo Finance...")
    sp = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)

    if sp.empty:
        raise ValueError("No data downloaded for ^GSPC")

    # Flatten multi-index columns if necessary
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)

    print("Columns received:", sp.columns.tolist())

    # Prefer Close (since auto_adjust=True already adjusts)
    if "Close" not in sp.columns:
        raise ValueError("Close column not found in Yahoo download.")

    sp = sp.reset_index()[["Date", "Close"]]
    sp.rename(columns={"Close": "SP500_Close"}, inplace=True)

    out_path = OUT_DIR / "sp500.csv"
    sp.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(sp))
    print("Date range:", sp["Date"].min(), "to", sp["Date"].max())

if __name__ == "__main__":
    main()