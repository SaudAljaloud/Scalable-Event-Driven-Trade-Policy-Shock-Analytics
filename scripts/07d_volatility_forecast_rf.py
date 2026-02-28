from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel_vol_model.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def expanding_splits(n, first_train=500, test_size=126, step=126):
    splits = []
    train_end = first_train
    while train_end + test_size <= n:
        splits.append((np.arange(0, train_end),
                       np.arange(train_end, train_end + test_size)))
        train_end += step
    return splits


def main():

    df = pd.read_csv(PANEL_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    assets = ["BTCUSD", "DAX", "SP500"]

    results = []

    for asset in assets:

        vol_col = f"{asset}_RV5"
        if vol_col not in df.columns:
            continue

        # Build lag features
        df[f"{vol_col}_lag1"] = df[vol_col].shift(1)
        df[f"{vol_col}_lag2"] = df[vol_col].shift(2)

        base_feats = [
            f"{vol_col}_lag1",
            f"{vol_col}_lag2",
            "VIX_z"
        ]

        tp_feats = [
            "TPNSI",
            "TPNSI_lag1",
            "TPNSI_lag2"
        ]

        sub = df[["Date", vol_col] + base_feats + tp_feats].dropna().copy()

        X_base = sub[base_feats].values
        X_full = sub[base_feats + tp_feats].values
        y = sub[vol_col].values

        splits = expanding_splits(len(sub))
        if not splits:
            train_end = int(len(sub) * 0.7)
            splits = [(np.arange(0, train_end),
                       np.arange(train_end, len(sub)))]

        def evaluate(X):

            y_true_all, y_pred_all = [], []

            for tr, te in splits:

                model = RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    min_samples_leaf=3,
                    n_jobs=-1
                )

                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])

                y_true_all.append(y[te])
                y_pred_all.append(pred)

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)

            return rmse(y_true_all, y_pred_all)

        rmse_base = evaluate(X_base)
        rmse_full = evaluate(X_full)

        results.append({
            "asset": asset,
            "RMSE_RF_controls": rmse_base,
            "RMSE_RF_controls_plus_TPNSI": rmse_full,
            "Improvement": rmse_base - rmse_full
        })

    out_df = pd.DataFrame(results)
    out_path = OUT_DIR / "volatility_rf_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()