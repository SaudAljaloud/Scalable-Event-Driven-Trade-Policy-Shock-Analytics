# scripts/04_build_tpns_features.py
#
# Build final trading-day panel for modeling:
# - Markets (BTC, DAX, etc.) + VIX + TPNS components/TPNSI
# - Restrict sample to full years: 2016-01-01 through 2025-12-31 (Option B)
# - Compute log returns, 5-day realized volatility, TPNSI/VIX lags + rolling stats
# - Create 1-day-ahead return targets (*_Ret_t1)
#
# Output:
#   data/processed/panel_daily.csv

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

MARKETS_PATH = BASE_DIR / "data" / "raw" / "markets" / "markets_merged.csv"
VIX_PATH = BASE_DIR / "data" / "raw" / "vix" / "vix_fred.csv"
TP_PATH = BASE_DIR / "data" / "raw" / "gdelt" / "tp_events_daily.csv"

OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Paper design window (Option B)
# -----------------------------
DESIGN_MIN_DATE = pd.Timestamp("2016-01-01")
DESIGN_MAX_DATE = pd.Timestamp("2025-12-31")


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def compute_returns(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
    """
    Log returns: Ret_t = log(P_t) - log(P_{t-1})
    """
    out = df.copy()
    for col in price_cols:
        out[col.replace("_Close", "_Ret")] = np.log(out[col]).diff()
    return out


def realized_vol(df: pd.DataFrame, ret_cols: list[str], window: int = 5) -> pd.DataFrame:
    """
    Rolling std of returns (proxy realized vol).
    """
    out = df.copy()
    for col in ret_cols:
        out[col.replace("_Ret", f"_RV{window}")] = out[col].rolling(window).std()
    return out


def add_lags_and_rolls(
    df: pd.DataFrame,
    cols: list[str],
    lags=(1, 2, 3),
    roll_windows=(5,),
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
        for w in roll_windows:
            out[f"{c}_roll{w}_mean"] = out[c].rolling(w).mean()
            out[f"{c}_roll{w}_std"] = out[c].rolling(w).std()
    return out


def main():
    # ---- Load inputs
    for p in [MARKETS_PATH, VIX_PATH, TP_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    markets = pd.read_csv(MARKETS_PATH)
    if "Date" not in markets.columns:
        raise ValueError("markets_merged.csv missing Date column")
    markets["Date"] = pd.to_datetime(markets["Date"], errors="coerce")
    markets = markets.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    vix = pd.read_csv(VIX_PATH)
    if "Date" not in vix.columns:
        raise ValueError("vix_fred.csv missing Date column")
    vix["Date"] = pd.to_datetime(vix["Date"], errors="coerce")
    vix = vix.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    tp = pd.read_csv(TP_PATH)
    if "Date" not in tp.columns:
        raise ValueError("tp_events_daily.csv missing Date column")
    tp["Date"] = pd.to_datetime(tp["Date"], errors="coerce")
    tp = tp.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    required = {"tp_count", "tp_tone_mean", "tp_dispersion_entropy"}
    missing = required - set(tp.columns)
    if missing:
        raise ValueError(f"tp_events_daily.csv missing columns: {missing}")

    # ---- Determine aligned window (bounded by Option B design window)
    tp_min, tp_max = tp["Date"].min(), tp["Date"].max()
    m_min, m_max = markets["Date"].min(), markets["Date"].max()
    v_min, v_max = vix["Date"].min(), vix["Date"].max()

    MIN_DATE = max(DESIGN_MIN_DATE, tp_min, m_min, v_min)
    MAX_DATE = min(DESIGN_MAX_DATE, tp_max, m_max, v_max)

    if MIN_DATE >= MAX_DATE:
        raise ValueError(f"Non-overlapping date ranges after alignment: MIN_DATE={MIN_DATE} MAX_DATE={MAX_DATE}")

    print("Aligned window:", MIN_DATE, "to", MAX_DATE)

    # ---- Filter inputs
    markets = markets[(markets["Date"] >= MIN_DATE) & (markets["Date"] <= MAX_DATE)].copy()
    vix = vix[(vix["Date"] >= MIN_DATE) & (vix["Date"] <= MAX_DATE)].copy()
    tp = tp[(tp["Date"] >= MIN_DATE) & (tp["Date"] <= MAX_DATE)].copy()

    # ---- TPNSI (z-score within the TP sample window)
    tp["tp_count_z"] = zscore(tp["tp_count"])
    tp["tp_tone_z"] = zscore(tp["tp_tone_mean"])
    tp["tp_disp_z"] = zscore(tp["tp_dispersion_entropy"])
    tp["TPNSI"] = (tp["tp_count_z"] + tp["tp_tone_z"] + tp["tp_disp_z"]) / 3.0

    # ---- Merge panel (markets drive trading days)
    panel = markets.merge(vix, on="Date", how="left").merge(
        tp[["Date", "tp_count", "tp_tone_mean", "tp_dispersion_entropy", "TPNSI"]],
        on="Date",
        how="left"
    ).sort_values("Date").reset_index(drop=True)

    # ---- Compute returns and realized vol
    price_cols = [c for c in panel.columns if c.endswith("_Close")]
    if not price_cols:
        raise ValueError("No *_Close columns found in markets_merged.csv after filtering")

    panel = compute_returns(panel, price_cols)
    ret_cols = [c for c in panel.columns if c.endswith("_Ret")]
    panel = realized_vol(panel, ret_cols, window=5)

    # ---- Add VIX + TPNSI features
    if "VIX" in panel.columns:
        panel["VIX_z"] = zscore(panel["VIX"])
    else:
        panel["VIX_z"] = np.nan

    panel = add_lags_and_rolls(panel, cols=["TPNSI", "VIX_z"], lags=(1, 2, 3), roll_windows=(5,))

    # ---- 1-day-ahead targets for each return series
    for rcol in ret_cols:
        panel[rcol.replace("_Ret", "_Ret_t1")] = panel[rcol].shift(-1)

    # ---- Save
    out_path = OUT_DIR / "panel_daily.csv"
    panel.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(panel), "Cols:", len(panel.columns))
    print("Date range:", panel["Date"].min(), "to", panel["Date"].max())
    print("TP rows:", len(tp))
    print("TPNSI non-null (panel):", int(panel["TPNSI"].notna().sum()))


if __name__ == "__main__":
    main()