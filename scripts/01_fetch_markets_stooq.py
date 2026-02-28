import os
import time
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "markets"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = {
    "SPX": "^spx",
    "UKX": "^ukx",
    "NKX": "^nkx",
    "DAX": "^dax",
    "BTCUSD": "btcusd",
}

def stooq_url(symbol: str) -> str:
    # Daily data: i=d
    # Example: https://stooq.com/q/d/l/?s=^spx&i=d
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"

def fetch_symbol(name: str, symbol: str) -> pd.DataFrame:
    url = stooq_url(symbol)
    df = pd.read_csv(url)
    # Stooq columns: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.rename(columns={"Close": f"{name}_Close"}, inplace=True)
    return df[["Date", f"{name}_Close"]]

def main():
    merged = None
    for name, symbol in SYMBOLS.items():
        print(f"Fetching {name} ({symbol}) ...")
        df = fetch_symbol(name, symbol)
        out = RAW_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        merged = df if merged is None else merged.merge(df, on="Date", how="outer")
        time.sleep(0.7)  # gentle throttle

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged.to_csv(RAW_DIR / "markets_merged.csv", index=False)
    print("Saved:", RAW_DIR / "markets_merged.csv")

if __name__ == "__main__":
    main()