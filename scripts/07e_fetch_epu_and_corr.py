from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"
EPU_PATH = BASE_DIR / "data" / "raw" / "epu" / "All_Daily_Policy_Data.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {PANEL_PATH}")
    if not EPU_PATH.exists():
        raise FileNotFoundError(f"Missing: {EPU_PATH}")

    panel = pd.read_csv(PANEL_PATH)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values("Date")

    epu = pd.read_csv(EPU_PATH)
    print("EPU columns:", epu.columns.tolist())

    required = ["day", "month", "year", "daily_policy_index"]
    missing = [c for c in required if c not in epu.columns]
    if missing:
        raise ValueError(f"Missing required columns in EPU file: {missing}")

    # Build Date from Y/M/D
    epu_df = epu[["year", "month", "day", "daily_policy_index"]].copy()
    epu_df["year"] = pd.to_numeric(epu_df["year"], errors="coerce")
    epu_df["month"] = pd.to_numeric(epu_df["month"], errors="coerce")
    epu_df["day"] = pd.to_numeric(epu_df["day"], errors="coerce")
    epu_df["daily_policy_index"] = pd.to_numeric(epu_df["daily_policy_index"], errors="coerce")

    epu_df = epu_df.dropna(subset=["year", "month", "day", "daily_policy_index"]).copy()

    epu_df["Date"] = pd.to_datetime(
        dict(year=epu_df["year"].astype(int),
             month=epu_df["month"].astype(int),
             day=epu_df["day"].astype(int)),
        errors="coerce"
    )
    epu_df = epu_df.dropna(subset=["Date"]).copy()

    epu_df["EPU_z"] = zscore(epu_df["daily_policy_index"])

    # Merge with panel
    df = panel.merge(epu_df[["Date", "EPU_z"]], on="Date", how="inner")

    print("Merged rows:", len(df))
    if len(df) < 50:
        print("WARNING: Very small overlap. Check Date ranges:")
        print("Panel:", panel["Date"].min(), "to", panel["Date"].max())
        print("EPU  :", epu_df["Date"].min(), "to", epu_df["Date"].max())

    corr = df[["TPNSI", "VIX_z", "EPU_z"]].corr()

    out_path = OUT_DIR / "corr_tpns_vix_epu.csv"
    corr.to_csv(out_path)

    print("\nCorrelation Matrix (TPNSI, VIX_z, EPU_z):")
    print(corr)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()