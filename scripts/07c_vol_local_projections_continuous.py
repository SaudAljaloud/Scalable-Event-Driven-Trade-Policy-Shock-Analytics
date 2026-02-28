from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def forward_mean(x: pd.Series, h: int) -> pd.Series:
    return x.rolling(window=h).mean().shift(-(h - 1))

def nw_lags(h: int) -> int:
    return max(1, h)

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd

def run_lp(df: pd.DataFrame, ycol: str, asset: str, horizons=(1,3,5,10), use_z_tpns=True):
    rows = []
    tp = zscore(df["TPNSI"]) if use_z_tpns else df["TPNSI"]

    for h in horizons:
        y = forward_mean(df[ycol], h)
        X = pd.DataFrame({
            "TPNSI_cont": tp.astype(float),
            "VIX_z": df["VIX_z"].astype(float),
        })
        d = pd.concat([y.rename("y"), X], axis=1).dropna()

        Y = d["y"].values
        Xmat = sm.add_constant(d[["TPNSI_cont", "VIX_z"]].values)

        fit = sm.OLS(Y, Xmat).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(h)})

        beta = float(fit.params[1])
        se = float(fit.bse[1])
        t = float(beta / se) if se > 0 else np.nan
        p = float(fit.pvalues[1])

        rows.append({
            "asset": asset,
            "vol_measure": ycol,
            "horizon_days": h,
            "tpns_scale": "zscore" if use_z_tpns else "level",
            "beta_TPNSI_cont": beta,
            "se_NW": se,
            "t_stat": t,
            "p_value": p,
            "n_obs": int(len(d)),
            "nw_lags": int(nw_lags(h)),
        })

    return pd.DataFrame(rows)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["TPNSI", "VIX_z", "BTCUSD_RV5", "DAX_RV5", "SP500_RV5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    horizons = (1, 3, 5, 10)

    out = pd.concat([
        run_lp(df, "BTCUSD_RV5", "BTCUSD", horizons=horizons, use_z_tpns=True),
        run_lp(df, "DAX_RV5", "DAX", horizons=horizons, use_z_tpns=True),
        run_lp(df, "SP500_RV5", "SP500", horizons=horizons, use_z_tpns=True),

        # optional: also export level spec for completeness
        run_lp(df, "BTCUSD_RV5", "BTCUSD", horizons=horizons, use_z_tpns=False),
        run_lp(df, "DAX_RV5", "DAX", horizons=horizons, use_z_tpns=False),
        run_lp(df, "SP500_RV5", "SP500", horizons=horizons, use_z_tpns=False),
    ], ignore_index=True)

    out_path = OUT_DIR / "vol_local_projections_continuous.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)

    view = out[out["tpns_scale"] == "zscore"].sort_values(["asset", "horizon_days"])
    print("\nContinuous TPNSI (z-scored) local projections:")
    print(view.to_string(index=False))

if __name__ == "__main__":
    main()