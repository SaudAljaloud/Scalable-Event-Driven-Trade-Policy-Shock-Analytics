# scripts/07a_vol_event_study.py
# Volatility event study: High TPNSI - Low TPNSI
# Writes BTCUSD, DAX, SP500 (if present)
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_VOL = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "vol_event_study.csv"

ASSETS = ["BTCUSD", "DAX", "SP500"]
HORIZONS = [1, 3, 5, 10]

def forward_mean(series: pd.Series, h: int) -> pd.Series:
    # Forward mean of a daily measure over horizons:
    # t -> average over [t, t+h-1]
    # (this aligns well with RV-style measures)
    return series.rolling(window=h, min_periods=h).mean().shift(-(h-1))

def main():
    if not PANEL_VOL.exists():
        raise FileNotFoundError(f"Missing: {PANEL_VOL}")

    df = pd.read_csv(PANEL_VOL)
    if "Date" not in df.columns or "TPNSI" not in df.columns:
        raise KeyError("panel_vol_model.csv must include Date and TPNSI columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # TPNSI high threshold
    tp = df["TPNSI"].astype(float)
    q90 = float(tp.quantile(0.90))
    high = tp >= q90
    low = tp < q90

    rows = []

    for asset in ASSETS:
        vol_col = f"{asset}_RV5"
        if vol_col not in df.columns:
            print(f"Skipping {asset}: missing column {vol_col}")
            continue

        v = df[vol_col].astype(float)

        for h in HORIZONS:
            v_f = forward_mean(v, h)

            sub = pd.DataFrame({
                "Date": df["Date"],
                "TPNSI": tp,
                "is_high": high,
                "is_low": low,
                "v_f": v_f
            }).dropna(subset=["v_f", "TPNSI"])

            mean_high = float(sub.loc[sub["is_high"], "v_f"].mean())
            mean_low = float(sub.loc[sub["is_low"], "v_f"].mean())

            rows.append({
                "asset": asset,
                "vol_measure": vol_col,
                "horizon_days": h,
                "tpns_q90": q90,
                "mean_vol_high": mean_high,
                "mean_vol_low": mean_low,
                "diff_high_minus_low": mean_high - mean_low,
                "n_high": int(sub["is_high"].sum()),
                "n_low": int(sub["is_low"].sum()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE)
    print("Assets in output:", sorted(out["asset"].unique().tolist()))
    print(out)

if __name__ == "__main__":
    main()