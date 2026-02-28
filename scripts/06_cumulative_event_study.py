from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "panel_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def forward_cum_logret(ret: pd.Series, h: int) -> pd.Series:
    """
    Forward cumulative log return from t to t+h-1.
    Example: h=3 -> r_t + r_{t+1} + r_{t+2}
    """
    return ret.rolling(window=h).sum().shift(-(h - 1))

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    targets = ["BTCUSD_Ret", "DAX_Ret"]
    for c in targets:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in panel_model.csv (needed for cumulative horizons)")

    # High-shock indicator (top decile) using TPNSI at time t
    q90 = df["TPNSI"].quantile(0.90)
    df["TPNSI_high"] = (df["TPNSI"] >= q90).astype(int)

    horizons = [1, 3, 5, 10]
    rows = []

    for asset in ["BTCUSD", "DAX"]:
        r = df[f"{asset}_Ret"]
        for h in horizons:
            col = f"{asset}_cum{h}"
            df[col] = forward_cum_logret(r, h)

            sub = df[["Date", "TPNSI_high", col]].dropna()
            high = sub.loc[sub["TPNSI_high"] == 1, col]
            low  = sub.loc[sub["TPNSI_high"] == 0, col]

            rows.append({
                "asset": asset,
                "horizon_days": h,
                "tpns_q90": float(q90),
                "mean_cumret_high": float(high.mean()),
                "mean_cumret_low": float(low.mean()),
                "diff_high_minus_low": float(high.mean() - low.mean()),
                "n_high": int(high.shape[0]),
                "n_low": int(low.shape[0]),
            })

    out = pd.DataFrame(rows).sort_values(["asset", "horizon_days"])
    out_path = OUT_DIR / "cumulative_event_study.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()