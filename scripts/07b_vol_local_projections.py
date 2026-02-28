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

def run_lp(df: pd.DataFrame, ycol: str, asset: str, horizons=(1,3,5,10)):
    rows = []
    for h in horizons:
        y = forward_mean(df[ycol], h)
        X = pd.DataFrame({
            "TPNSI_high": df["TPNSI_high"].astype(float),
            "VIX_z": df["VIX_z"].astype(float),
        })
        d = pd.concat([y.rename("y"), X], axis=1).dropna()

        Y = d["y"].values
        Xmat = sm.add_constant(d[["TPNSI_high", "VIX_z"]].values)
        fit = sm.OLS(Y, Xmat).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(h)})

        beta = float(fit.params[1])
        se = float(fit.bse[1])
        t = float(beta / se) if se > 0 else np.nan
        p = float(fit.pvalues[1])

        rows.append({
            "asset": asset,
            "vol_measure": ycol,
            "horizon_days": h,
            "beta_TPNSI_high": beta,
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

    q90 = df["TPNSI"].quantile(0.90)
    df["TPNSI_high"] = (df["TPNSI"] >= q90).astype(int)

    out = pd.concat([
        run_lp(df, "BTCUSD_RV5", "BTCUSD"),
        run_lp(df, "DAX_RV5", "DAX"),
    ], ignore_index=True)

    out_path = OUT_DIR / "vol_local_projections.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("TPNSI q90:", float(q90))
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()