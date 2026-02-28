# scripts/05_train_models.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def expanding_splits(n, first_train=500, test_size=126, step=126):
    """
    Expanding window splits:
      train: [0:train_end)
      test : [train_end:train_end+test_size)
    Defaults ~ 2 years train then 6 months test for daily data.
    """
    splits = []
    train_end = first_train
    while train_end + test_size <= n:
        splits.append((np.arange(0, train_end), np.arange(train_end, train_end + test_size)))
        train_end += step
    return splits


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def pick_targets(df, wanted_keys=("BTC", "DAX", "SP500")):
    """
    Explicitly select the three targets for consistency across the paper.
    """
    targets = [c for c in df.columns if c.endswith("_Ret_t1")]
    if not targets:
        raise ValueError("No *_Ret_t1 columns found in panel_model.csv")

    chosen = []
    for key in wanted_keys:
        found = [t for t in targets if key in t.upper()]
        if not found:
            raise ValueError(f"Could not find target for {key}. Available targets: {targets}")
        chosen.append(found[0])

    # de-duplicate while preserving order
    out = []
    for t in chosen:
        if t not in out:
            out.append(t)
    return out


def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {PANEL_PATH}")

    df = pd.read_csv(PANEL_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # --- FIX: explicitly include BTC, DAX, SP500 ---
    chosen = pick_targets(df, wanted_keys=("BTC", "DAX", "SP500"))
    print("Targets:", chosen)

    # Base features (already computed in panel)
    base_feats = safe_cols(df, [
        "VIX_z", "VIX_z_lag1", "VIX_z_lag2", "VIX_z_lag3", "VIX_z_roll5_mean", "VIX_z_roll5_std"
    ])

    tp_feats = safe_cols(df, [
        "TPNSI", "TPNSI_lag1", "TPNSI_lag2", "TPNSI_lag3", "TPNSI_roll5_mean", "TPNSI_roll5_std"
    ])

    if "TPNSI" not in df.columns:
        raise ValueError("TPNSI not found in panel_model.csv. Rebuild features first (04_build_tpns_features.py).")

    # Event-study threshold
    tp_q90 = df["TPNSI"].quantile(0.90)
    df["TPNSI_high"] = (df["TPNSI"] >= tp_q90).astype(int)

    results_rows = []
    event_rows = []

    # Expanding splits (for full df; later rebuilt on per-target usable subsample)
    n = len(df)
    splits = expanding_splits(n)
    if not splits:
        train_end = int(n * 0.7)
        splits = [(np.arange(0, train_end), np.arange(train_end, n))]

    for ycol in chosen:
        # Construct AR(1) feature from same-day return
        xret = ycol.replace("_Ret_t1", "_Ret")
        if xret in df.columns:
            df[f"{xret}_lag1"] = df[xret].shift(1)
            ar_feats = [f"{xret}_lag1"]
        else:
            ar_feats = []  # model falls back to mean benchmark

        # Feature sets
        ctrl_feats = ar_feats + base_feats
        full_feats = ctrl_feats + tp_feats

        def prep(cols):
            sub = df[["Date", ycol] + cols].dropna().copy()
            X = sub[cols].values if cols else np.zeros((len(sub), 0))
            y = sub[ycol].values
            return sub.reset_index(drop=True), X, y

        sub0, X0, y0 = prep(ar_feats)
        sub1, X1, y1 = prep(ctrl_feats)
        sub2, X2, y2 = prep(full_feats)

        # Use sub2 as primary (most complete feature set). If too small, fallback.
        use = sub2
        cols_use = full_feats
        if len(use) < 800:
            use = sub1
            cols_use = ctrl_feats
        if len(use) < 400:
            use = sub0
            cols_use = ar_feats

        # rebuild splits on 'use'
        n_use = len(use)
        splits_use = expanding_splits(n_use)
        if not splits_use:
            train_end = int(n_use * 0.7)
            splits_use = [(np.arange(0, train_end), np.arange(train_end, n_use))]

        y = use[ycol].values
        dates = use["Date"].values

        def eval_model(name, model, X):
            y_true_all, y_pred_all, d_all = [], [], []
            for tr, te in splits_use:
                if X.shape[1] == 0:
                    mu = float(np.mean(y[tr]))
                    pred = np.full(len(te), mu)
                else:
                    model.fit(X[tr], y[tr])
                    pred = model.predict(X[te])
                y_true_all.append(y[te])
                y_pred_all.append(pred)
                d_all.append(dates[te])

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)
            d_all = np.concatenate(d_all)

            return {
                "target": ycol,
                "model": name,
                "rmse": rmse(y_true_all, y_pred_all),
                "mae": mae(y_true_all, y_pred_all),
                "n_obs": int(len(y_true_all)),
                "n_train_end": int(splits_use[-1][0][-1]) if splits_use else None,
                "features": ",".join(cols_use) if cols_use else "(mean)"
            }, pd.DataFrame({"Date": pd.to_datetime(d_all), "y_true": y_true_all, "y_pred": y_pred_all})

        # Build X matrices from `use` directly (consistent row alignment)
        def X_from(cols):
            if not cols:
                return np.zeros((len(use), 0))
            return use[cols].values

        # Models
        res_mean, pred_mean = eval_model("Mean", LinearRegression(), np.zeros((len(use), 0)))
        res_ar, pred_ar = eval_model("AR1", LinearRegression(), X_from(ar_feats))
        res_ols, pred_ols = eval_model("OLS_controls", LinearRegression(), X_from(ctrl_feats))
        res_tp, pred_tp = eval_model("OLS_controls+TPNS", LinearRegression(), X_from(full_feats))

        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1
        )
        res_rf, pred_rf = eval_model("RF_controls+TPNS", rf, X_from(full_feats))

        results_rows += [res_mean, res_ar, res_ols, res_tp, res_rf]

        # Save predictions
        pred_mean.to_csv(OUT_DIR / f"pred_{ycol}_Mean.csv", index=False)
        pred_ar.to_csv(OUT_DIR / f"pred_{ycol}_AR1.csv", index=False)
        pred_ols.to_csv(OUT_DIR / f"pred_{ycol}_OLS_controls.csv", index=False)
        pred_tp.to_csv(OUT_DIR / f"pred_{ycol}_OLS_TPNS.csv", index=False)
        pred_rf.to_csv(OUT_DIR / f"pred_{ycol}_RF_TPNS.csv", index=False)

        # Event study using the FULL DF (not 'use') because ycol exists there
        es = df[["TPNSI", "TPNSI_high", ycol]].dropna().copy()
        high = es.loc[es["TPNSI_high"] == 1, ycol]
        low = es.loc[es["TPNSI_high"] == 0, ycol]

        event_rows.append({
            "target": ycol,
            "tpns_q90": float(tp_q90),
            "mean_ret_high": float(high.mean()),
            "mean_ret_low": float(low.mean()),
            "diff_high_minus_low": float(high.mean() - low.mean()),
            "n_high": int(high.shape[0]),
            "n_low": int(low.shape[0])
        })

    results = pd.DataFrame(results_rows)
    results_path = OUT_DIR / "model_results.csv"
    results.to_csv(results_path, index=False)

    es_df = pd.DataFrame(event_rows)
    es_path = OUT_DIR / "event_study_tpns.csv"
    es_df.to_csv(es_path, index=False)

    print("\nSaved results:")
    print(" -", results_path)
    print(" -", es_path)

    # Console summary
    for t in results["target"].unique():
        print("\nTarget:", t)
        print(results[results["target"] == t].sort_values("rmse").to_string(index=False))

    print("\nEvent study (top decile TPNSI days):")
    print(es_df.to_string(index=False))


if __name__ == "__main__":
    main()