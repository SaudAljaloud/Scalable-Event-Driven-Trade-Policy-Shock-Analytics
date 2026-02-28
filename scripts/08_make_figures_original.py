# scripts/08_make_figures.py
# ---------------------------------------------
# Build paper-ready figures (PNG + PDF) for BDCC
# Figures:
#   01 TPNSI time series (daily + 30D MA)
#   02 TPNSI vs VIX (z-scored, daily + 30D MA)
#   03 Local projections: TPNSI -> forward realized volatility (RV5), with 95% CI
#   04 Volatility event study: High TPNSI - Low TPNSI (BTCUSD, DAX, SP500)
#   05 Rolling correlation: TPNSI vs RV5 (window=60)
#
# Inputs expected (relative to project root):
#   data/processed/panel_daily.csv
#   data/processed/panel_vol_model.csv
#   data/processed/results/vol_local_projections_continuous.csv
#   data/processed/results/vol_event_study.csv
#
# Outputs:
#   figures/*.png and figures/*.pdf
# ---------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

PANEL_DAILY = ROOT / "data" / "processed" / "panel_daily.csv"
PANEL_VOL = ROOT / "data" / "processed" / "panel_vol_model.csv"

LP_VOL = ROOT / "data" / "processed" / "results" / "vol_local_projections_continuous.csv"
VOL_EVENT = ROOT / "data" / "processed" / "results" / "vol_event_study.csv"

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

ROLL_WIN = 60  # days
DPI = 300

ASSETS = ["BTCUSD", "DAX", "SP500"]  # keep explicit to avoid silent drops


# -----------------------------
# Helpers
# -----------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path)

def _to_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

def save_fig(fig: plt.Figure, stem: str) -> None:
    png_path = OUTDIR / f"{stem}.png"
    pdf_path = OUTDIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


# -----------------------------
# Figure 01: TPNSI time series
# -----------------------------
def fig01_tpns_time(panel_daily: pd.DataFrame) -> None:
    df = panel_daily[["Date", "TPNSI"]].dropna().copy()
    df = df.sort_values("Date")
    df["TPNSI_30dma"] = df["TPNSI"].rolling(30, min_periods=10).mean()

    fig = plt.figure(figsize=(12.5, 4.2))
    ax = plt.gca()
    ax.plot(df["Date"], df["TPNSI"], alpha=0.45, label="TPNSI (daily)")
    ax.plot(df["Date"], df["TPNSI_30dma"], linewidth=2.3, label="TPNSI (30D MA)")
    ax.set_title("TPNSI (Trade Policy News Shock Index)")
    ax.set_xlabel("Date")
    ax.set_ylabel("TPNSI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    save_fig(fig, "fig01_tpns_time_polished")
    plt.close(fig)


# -----------------------------
# Figure 02: TPNSI vs VIX (z)
# -----------------------------
def fig02_tpns_vs_vix(panel_daily: pd.DataFrame) -> None:
    cols = ["Date", "TPNSI", "VIX"]
    missing = [c for c in cols if c not in panel_daily.columns]
    if missing:
        raise KeyError(f"panel_daily missing columns: {missing}")

    df = panel_daily[cols].copy()
    df = df.dropna(subset=["TPNSI", "VIX"]).sort_values("Date")

    df["TPNSI_z"] = zscore(df["TPNSI"])
    df["VIX_z"] = zscore(df["VIX"])

    df["TPNSI_z_30dma"] = df["TPNSI_z"].rolling(30, min_periods=10).mean()
    df["VIX_z_30dma"] = df["VIX_z"].rolling(30, min_periods=10).mean()

    fig = plt.figure(figsize=(13.5, 4.8))
    ax = plt.gca()

    ax.plot(df["Date"], df["TPNSI_z"], alpha=0.30, label="TPNSI (z, daily)")
    ax.plot(df["Date"], df["VIX_z"], alpha=0.30, label="VIX (z, daily)")
    ax.plot(df["Date"], df["TPNSI_z_30dma"], linewidth=2.2, label="TPNSI (z, 30D MA)")
    ax.plot(df["Date"], df["VIX_z_30dma"], linewidth=2.2, label="VIX (z, 30D MA)")

    ax.set_title("TPNSI vs VIX (Standardized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized level (z)")
    ax.grid(True, alpha=0.25)

    # Put legend outside to keep plot uncluttered
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    save_fig(fig, "fig02_tpns_vs_vix_polished")
    plt.close(fig)


# -----------------------------
# Figure 03: Local projections (continuous TPNSI -> RV5)
# -----------------------------
def fig03_lp_volatility(lp: pd.DataFrame) -> None:
    # Expected columns from your output:
    # asset, horizon_days, beta_TPNSI_cont, se_NW
    needed = {"asset", "horizon_days", "beta_TPNSI_cont", "se_NW"}
    if not needed.issubset(lp.columns):
        raise KeyError(f"vol_local_projections_continuous.csv must include {sorted(needed)}")

    df = lp.copy()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
    df = df.dropna(subset=["asset", "horizon_days", "beta_TPNSI_cont", "se_NW"])
    df = df[df["asset"].isin(ASSETS)].copy()

    # Force sorting to avoid “vertical jump” artifacts
    df = df.sort_values(["asset", "horizon_days"])

    fig = plt.figure(figsize=(10.5, 5.2))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1.2)

    for asset in ASSETS:
        d = df[df["asset"] == asset].copy()
        if d.empty:
            continue
        h = d["horizon_days"].values
        b = d["beta_TPNSI_cont"].values
        se = d["se_NW"].values
        lo = b - 1.96 * se
        hi = b + 1.96 * se

        ax.plot(h, b, marker="o", linewidth=2.2, label=asset)
        ax.fill_between(h, lo, hi, alpha=0.12)

    ax.set_title("Local Projections: TPNSI → Forward Realized Volatility (RV5)")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel(r"$\beta(\mathrm{TPNSI})$ with 95% CI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, "fig03_lp_volatility_polished")
    plt.close(fig)


# -----------------------------
# Figure 04: Volatility event study (High - Low TPNSI), include SP500
# -----------------------------
def fig04_eventstudy_vol(vol_event: pd.DataFrame) -> None:
    needed = {"asset", "horizon_days", "diff_high_minus_low"}
    if not needed.issubset(vol_event.columns):
        raise KeyError(f"vol_event_study.csv must include {sorted(needed)}")

    df = vol_event.copy()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
    df = df.dropna(subset=["asset", "horizon_days", "diff_high_minus_low"])
    df = df[df["asset"].isin(ASSETS)].copy()
    df = df.sort_values(["asset", "horizon_days"])

    fig = plt.figure(figsize=(10.5, 5.0))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1.2)

    for asset in ASSETS:
        d = df[df["asset"] == asset]
        if d.empty:
            continue
        ax.plot(
            d["horizon_days"].values,
            d["diff_high_minus_low"].values,
            marker="o",
            linewidth=2.2,
            label=asset,
        )

    ax.set_title("Volatility Event Study: High TPNSI − Low TPNSI")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Difference in mean RV5")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, "fig04_eventstudy_vol_polished")
    plt.close(fig)


# -----------------------------
# Figure 05: Rolling correlation TPNSI vs RV5 (60D), include SP500
# -----------------------------
def fig05_rolling_corr(panel_vol: pd.DataFrame) -> None:
    cols = ["Date", "TPNSI"]
    for a in ASSETS:
        cols.append(f"{a}_RV5")

    missing = [c for c in cols if c not in panel_vol.columns]
    if missing:
        raise KeyError(f"panel_vol_model.csv missing columns: {missing}")

    df = panel_vol[cols].copy()
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.set_index("Date")

    fig = plt.figure(figsize=(12.5, 4.8))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1.1)

    for a in ASSETS:
        x = df["TPNSI"].astype(float)
        y = df[f"{a}_RV5"].astype(float)

        # Align and drop NaNs per-series so SP500 has a fair chance to appear
        tmp = pd.concat([x, y], axis=1).dropna()
        if len(tmp) < ROLL_WIN + 10:
            print(f"Skipping {a}: not enough overlap after NaN drop (n={len(tmp)})")
            continue

        corr = tmp["TPNSI"].rolling(ROLL_WIN).corr(tmp[f"{a}_RV5"])
        ax.plot(corr.index, corr.values, linewidth=2.0, label=f"{a} RV5")

    ax.set_title(f"Rolling Correlation: TPNSI vs Realized Volatility (window = {ROLL_WIN} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    save_fig(fig, "fig05_rolling_corr_tpns_rv_polished")
    plt.close(fig)


def main() -> None:
    print("Project root:", ROOT)
    print("Reading inputs...")

    panel_daily = _to_datetime(_read_csv(PANEL_DAILY), "Date")
    panel_vol = _to_datetime(_read_csv(PANEL_VOL), "Date")

    lp = _read_csv(LP_VOL)
    vol_event = _read_csv(VOL_EVENT)

    # Basic sanity
    if "TPNSI" not in panel_daily.columns:
        raise KeyError("panel_daily.csv must contain TPNSI")
    if "VIX" not in panel_daily.columns:
        raise KeyError("panel_daily.csv must contain VIX (or update fig02 to your VIX column name)")

    print("Building figures...")
    fig01_tpns_time(panel_daily)
    fig02_tpns_vs_vix(panel_daily)
    fig03_lp_volatility(lp)
    fig04_eventstudy_vol(vol_event)
    fig05_rolling_corr(panel_vol)

    print("\nDone. Outputs in:", OUTDIR)


if __name__ == "__main__":
    main()