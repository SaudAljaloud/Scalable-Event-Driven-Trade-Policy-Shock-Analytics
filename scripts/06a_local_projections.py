# scripts/06a_local_projections.py
from pathlib import Path
import numpy as np
import pandas as pd

import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "panel_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def forward_cum_logret(ret: pd.Series, h: int) -> pd.Series:
    """
    Forward cumulative log return from t to t+h-1:
      cum_h(t) = r_t + r_{t+1} + ... + r_{t+h-1}
    """
    return ret.rolling(window=h).sum().shift(-(h - 1))


def nw_lags(h: int) -> int:
    # Standard simple choice: maxlags proportional to horizon
    return max(1, h)


def run_lp(df: pd.DataFrame, asset: str, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    """
    Local projection:
      CumRet_{t->t+h} = a_h + b_h*TPNSI_high_t + c_h*VIX_z_t + e_{t,h}
    HAC(Newey-West) SE with maxlags = h
    """
    rows = []
    ret_col = f"{asset}_Ret"
    if ret_col not in df.columns:
        raise ValueError(f"Missing return column: {ret_col}")

    if "TPNSI_high" not in df.columns:
        raise ValueError("Missing TPNSI_high indicator")

    # VIX_z optional: if missing, it is set to 0 and we still estimate b_h.
    has_vix = "VIX_z" in df.columns

    for h in horizons:
        y = forward_cum_logret(df[ret_col], h)

        X = pd.DataFrame({
            "TPNSI_high": df["TPNSI_high"].astype(float),
            "VIX_z": df["VIX_z"].astype(float) if has_vix else 0.0,
        })

        d = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(d) < 200:
            raise ValueError(f"Too few observations after NA drop for {asset}, h={h}: {len(d)}")

        Y = d["y"].values
        Xmat = sm.add_constant(d[["TPNSI_high", "VIX_z"]].values)

        fit = sm.OLS(Y, Xmat).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(h)})

        # params order: const, TPNSI_high, VIX_z
        beta = float(fit.params[1])
        se = float(fit.bse[1])
        t = float(beta / se) if se > 0 else np.nan
        p = float(fit.pvalues[1]) if hasattr(fit, "pvalues") else np.nan

        rows.append({
            "asset": asset,
            "horizon_days": h,
            "beta_TPNSI_high": beta,
            "se_NW": se,
            "t_stat": t,
            "p_value": p,
            "n_obs": int(len(d)),
            "nw_lags": int(nw_lags(h)),
            "controls": "VIX_z" if has_vix else "(none)",
        })

    return pd.DataFrame(rows)


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for c in ["TPNSI", "BTCUSD_Ret", "DAX_Ret"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # High-shock indicator at time t (top decile)
    q90 = df["TPNSI"].quantile(0.90)
    df["TPNSI_high"] = (df["TPNSI"] >= q90).astype(int)

    horizons = (1, 3, 5, 10)

    out = pd.concat(
        [
            run_lp(df, "BTCUSD", horizons=horizons),
            run_lp(df, "DAX", horizons=horizons),
        ],
        ignore_index=True
    )

    out_path = OUT_DIR / "local_projections_tpns.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("\nTPNSI top-decile threshold (q90):", float(q90))
    print(out.sort_values(["asset", "horizon_days"]).to_string(index=False))


if __name__ == "__main__":
    main()