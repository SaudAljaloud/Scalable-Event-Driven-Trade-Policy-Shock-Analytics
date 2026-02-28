from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "panel_daily.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "panel_model.csv"

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Identify all 1-day ahead return targets
    targets = [c for c in df.columns if c.endswith("_Ret_t1")]

    if not targets:
        raise ValueError("No *_Ret_t1 columns found in panel_daily.csv")

    # ---- Explicitly keep BTC, DAX, and S&P (SPX or SP500) ----
    wanted_keys = ["BTC", "DAX", "SP500", "SPX"]

    keep_targets = []
    for key in wanted_keys:
        for t in targets:
            if key in t.upper():
                keep_targets.append(t)

    # remove duplicates while preserving order
    keep_targets = list(dict.fromkeys(keep_targets))

    if not keep_targets:
        raise ValueError(f"No matching targets found. Available targets: {targets}")

    # Required columns
    required_cols = ["TPNSI"]
    if "VIX" in df.columns:
        required_cols.append("VIX")

    req = required_cols + keep_targets
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)

    df2 = df.dropna(subset=required_cols).copy()
    df2 = df2.dropna(subset=keep_targets, how="all").copy()

    after = len(df2)

    df2.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows before:", before)
    print("Rows after :", after)
    print("Date range :", df2['Date'].min(), "to", df2['Date'].max())
    print("Targets kept:", keep_targets)

if __name__ == "__main__":
    main()