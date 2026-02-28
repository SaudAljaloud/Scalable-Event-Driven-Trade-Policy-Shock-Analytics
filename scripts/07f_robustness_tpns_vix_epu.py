from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import diagnostic as smd

BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"
EPU_PATH = BASE_DIR / "data" / "raw" / "epu" / "All_Daily_Policy_Data.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def load_epu():
    epu = pd.read_csv(EPU_PATH)

    epu["Date"] = pd.to_datetime(
        dict(year=epu["year"], month=epu["month"], day=epu["day"]),
        errors="coerce"
    )
    epu["EPU_raw"] = pd.to_numeric(epu["daily_policy_index"], errors="coerce")
    epu = epu.dropna(subset=["Date", "EPU_raw"]).copy()
    epu["EPU_z"] = zscore(epu["EPU_raw"])

    return epu[["Date", "EPU_z"]]

def main():

    panel = pd.read_csv(PANEL_PATH)
    panel["Date"] = pd.to_datetime(panel["Date"])

    epu_df = load_epu()

    df = panel.merge(epu_df, on="Date", how="inner")

    assets = ["BTCUSD", "DAX", "SP500"]
    rows = []

    for asset in assets:

        rv_col = f"{asset}_RV5"
        if rv_col not in df.columns:
            continue

        sub = df[[rv_col, "TPNSI", "VIX_z", "EPU_z"]].dropna().copy()

        y = sub[rv_col]
        X = sub[["TPNSI", "VIX_z", "EPU_z"]]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        rows.append({
            "asset": asset,
            "beta_TPNSI": model.params["TPNSI"],
            "se_TPNSI": model.bse["TPNSI"],
            "t_TPNSI": model.tvalues["TPNSI"],
            "p_TPNSI": model.pvalues["TPNSI"],
            "beta_VIX": model.params["VIX_z"],
            "beta_EPU": model.params["EPU_z"],
            "R2": model.rsquared,
            "n_obs": int(model.nobs)
        })

    results = pd.DataFrame(rows)
    out_path = OUT_DIR / "robustness_tpns_vix_epu.csv"
    results.to_csv(out_path, index=False)

    print("\nRobustness Regression Results:")
    print(results)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()