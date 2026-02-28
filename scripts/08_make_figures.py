#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_make_figures.py
------------------
Generates publication-ready figures (PNG + PDF) for the BDCC manuscript.

Inputs (expected):
- data/processed/panel_daily.csv
- data/processed/panel_vol_model.csv
- data/processed/results/vol_local_projections_continuous.csv
- data/processed/results/vol_event_study.csv

Outputs (default):
- figures/fig01_tpns_time.(png|pdf)
- figures/fig02_tpns_vs_vix.(png|pdf)
- figures/fig03_lp_volatility.(png|pdf)
- figures/fig04_eventstudy_vol.(png|pdf)   # includes BTCUSD, DAX, SP500
- figures/fig05_rolling_corr_tpns_rv.(png|pdf)

Run:
    python scripts/08_make_figures.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _to_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' in dataframe; got: {list(df.columns)[:20]} ...")
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def _save_both(fig: plt.Figure, out_base: Path, dpi_png: int = 300) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi_png, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def fig01_tpns_time(panel_daily: pd.DataFrame, outdir: Path) -> None:
    df = panel_daily[["Date", "TPNSI"]].dropna().copy()
    df["TPNSI_30D_MA"] = df["TPNSI"].rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(df["Date"], df["TPNSI"], linewidth=1.8, alpha=0.45, label="TPNSI (daily)")
    ax.plot(df["Date"], df["TPNSI_30D_MA"], linewidth=2.6, label="TPNSI (30D MA)")

    ax.set_title("TPNSI (Trade Policy News Shock Index)")
    ax.set_xlabel("Date")
    ax.set_ylabel("TPNSI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    _save_both(fig, outdir / "fig01_tpns_time")
    plt.close(fig)


def fig02_tpns_vs_vix(panel_daily: pd.DataFrame, outdir: Path) -> None:
    df = panel_daily.copy()
    if "VIX_z" in df.columns:
        df["VIX_z_use"] = pd.to_numeric(df["VIX_z"], errors="coerce")
    elif "VIX" in df.columns:
        df["VIX_z_use"] = _zscore(df["VIX"])
    else:
        raise KeyError("panel_daily must have either 'VIX_z' or 'VIX' column.")

    df["TPNSI_z"] = _zscore(df["TPNSI"])
    df = df[["Date", "TPNSI_z", "VIX_z_use"]].dropna().copy()
    df["TPNSI_z_30D_MA"] = df["TPNSI_z"].rolling(30, min_periods=10).mean()
    df["VIX_z_30D_MA"] = df["VIX_z_use"].rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(df["Date"], df["TPNSI_z"], linewidth=1.8, alpha=0.30, label="TPNSI (z, daily)")
    ax.plot(df["Date"], df["VIX_z_use"], linewidth=1.8, alpha=0.30, label="VIX (z, daily)")
    ax.plot(df["Date"], df["TPNSI_z_30D_MA"], linewidth=2.6, label="TPNSI (z, 30D MA)")
    ax.plot(df["Date"], df["VIX_z_30D_MA"], linewidth=2.6, label="VIX (z, 30D MA)")

    ax.set_title("TPNSI vs VIX (Standardized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized level (z)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    _save_both(fig, outdir / "fig02_tpns_vs_vix")
    plt.close(fig)


def fig03_lp_volatility(vol_lp_cont: pd.DataFrame, outdir: Path) -> None:
    needed = ["asset", "horizon_days", "beta_TPNSI_cont", "se_NW"]
    for c in needed:
        if c not in vol_lp_cont.columns:
            raise KeyError(f"vol_local_projections_continuous missing column: {c}")

    df = vol_lp_cont.copy()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce").astype(int)
    df["beta"] = pd.to_numeric(df["beta_TPNSI_cont"], errors="coerce")
    df["se"] = pd.to_numeric(df["se_NW"], errors="coerce")
    df["lo"] = df["beta"] - 1.96 * df["se"]
    df["hi"] = df["beta"] + 1.96 * df["se"]

    assets_order = ["BTCUSD", "DAX", "SP500"]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for a in assets_order:
        sub = df[df["asset"] == a].sort_values("horizon_days")
        if sub.empty:
            continue
        ax.plot(sub["horizon_days"], sub["beta"], marker="o", linewidth=2.4, label=a)
        ax.fill_between(sub["horizon_days"], sub["lo"], sub["hi"], alpha=0.12)

    ax.axhline(0.0, linewidth=1.4)
    ax.set_title("Local Projections: TPNSI → Forward Realized Volatility (RV5)")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel(r"$\beta(\mathrm{TPNSI})$ with 95% CI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)

    _save_both(fig, outdir / "fig03_lp_volatility")
    plt.close(fig)


def fig04_eventstudy_vol(vol_event: pd.DataFrame, outdir: Path) -> None:
    needed = ["asset", "horizon_days", "diff_high_minus_low"]
    for c in needed:
        if c not in vol_event.columns:
            raise KeyError(f"vol_event_study missing column: {c}")

    df = vol_event.copy()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce").astype(int)
    df["diff"] = pd.to_numeric(df["diff_high_minus_low"], errors="coerce")

    assets_order = ["BTCUSD", "DAX", "SP500"]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    any_plotted = False
    for a in assets_order:
        sub = df[df["asset"] == a].sort_values("horizon_days")
        if sub.empty:
            continue
        any_plotted = True
        ax.plot(sub["horizon_days"], sub["diff"], marker="o", linewidth=2.4, label=a)

    if not any_plotted:
        raise ValueError("No assets plotted. Check 'asset' values in vol_event_study.csv.")

    ax.axhline(0.0, linewidth=1.4)
    ax.set_title("Volatility Event Study: High TPNSI − Low TPNSI")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Difference in mean RV5")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    _save_both(fig, outdir / "fig04_eventstudy_vol")
    plt.close(fig)


def fig05_rolling_corr(panel_vol: pd.DataFrame, outdir: Path, window: int = 60) -> None:
    needed = ["Date", "TPNSI", "BTCUSD_RV5", "DAX_RV5", "SP500_RV5"]
    missing = [c for c in needed if c not in panel_vol.columns]
    if missing:
        raise KeyError(f"panel_vol_model missing columns: {missing}")

    df = panel_vol[["Date", "TPNSI", "BTCUSD_RV5", "DAX_RV5", "SP500_RV5"]].copy()
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["TPNSI_z"] = _zscore(df["TPNSI"])

    fig, ax = plt.subplots(figsize=(12.5, 6.0))

    for label, col in [("BTC RV5", "BTCUSD_RV5"), ("DAX RV5", "DAX_RV5"), ("SP500 RV5", "SP500_RV5")]:
        tmp = df[["Date", "TPNSI_z", col]].dropna().copy()
        if tmp.empty:
            continue
        tmp["rv_z"] = _zscore(tmp[col])
        tmp["corr"] = tmp["TPNSI_z"].rolling(window).corr(tmp["rv_z"])
        ax.plot(tmp["Date"], tmp["corr"], linewidth=2.0, label=label)

    ax.axhline(0.0, linewidth=1.4)
    ax.set_title(f"Rolling Correlation: TPNSI vs Realized Volatility (window = {window} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    _save_both(fig, outdir / "fig05_rolling_corr_tpns_rv")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../tpns-transmission-bdcc
    proc_dir = project_root / "data" / "processed"
    res_dir = proc_dir / "results"
    fig_dir = project_root / "figures"
    _ensure_dir(fig_dir)

    panel_daily = _to_datetime(_read_csv(proc_dir / "panel_daily.csv"))
    panel_vol = _to_datetime(_read_csv(proc_dir / "panel_vol_model.csv"))
    vol_lp = _read_csv(res_dir / "vol_local_projections_continuous.csv")
    vol_event = _read_csv(res_dir / "vol_event_study.csv")

    fig01_tpns_time(panel_daily, fig_dir)
    fig02_tpns_vs_vix(panel_daily, fig_dir)
    fig03_lp_volatility(vol_lp, fig_dir)
    fig04_eventstudy_vol(vol_event, fig_dir)  # includes SP500
    fig05_rolling_corr(panel_vol, fig_dir, window=60)

    out = sorted([p.name for p in fig_dir.glob("fig*.png")]) + sorted([p.name for p in fig_dir.glob("fig*.pdf")])
    print("Saved figures to:", fig_dir)
    for name in out:
        print(" -", name)


if __name__ == "__main__":
    main()
