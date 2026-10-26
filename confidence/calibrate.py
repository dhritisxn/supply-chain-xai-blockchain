"""
calibrate.py

Validates the core assumption behind the Confidence Estimation Module:
that a higher confidence score should correspond to lower actual forecast
error. Runs each model over its held-out test set, computes confidence
alongside actual absolute percentage error (APE), and reports:

  1. Pearson correlation between confidence and -APE (expect negative
     correlation between confidence and error, i.e. r(confidence, APE) < 0)
  2. A reliability table: mean APE within each confidence bucket -- should
     decrease monotonically as confidence increases, if the score is doing
     its job.

Also exposes `suggest_alpha()`, a small grid search that picks the alpha
producing the strongest (most negative) confidence-vs-error correlation on
the test set -- use this after swapping in a real dataset to re-calibrate
threshold_config.yaml's alpha values instead of guessing them.

This is deliberately a standalone evaluation script, not part of the live
inference path -- it's meant to be re-run whenever the data or models
change, not on every forecast.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

from confidence_estimator import (
    score_from_interval, _width_to_confidence,
)


def _ape(actual: float, predicted: float) -> float:
    return abs(actual - predicted) / max(abs(actual), 1e-6)


def calibrate_xgboost(alpha_grid=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> dict:
    from xgboost_model import train_xgboost
    from feature_engineering import FEATURE_COLS

    result = train_xgboost(save=False)
    test_df = result["test_df"]
    X_test = test_df[FEATURE_COLS]

    point = result["model"].predict(X_test)
    lower = result["lower_model"].predict(X_test)
    upper = result["upper_model"].predict(X_test)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    actual = test_df["sales"].values

    return _evaluate_alpha_grid("xgboost", point, lower, upper, actual, alpha_grid)


def calibrate_prophet(alpha_grid=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) -> dict:
    from prophet_model import train_prophet_for_series
    from data_loader import load_raw_sales

    raw = load_raw_sales()
    products = sorted(raw["product_id"].unique())
    regions = sorted(raw["region"].unique())

    all_point, all_lower, all_upper, all_actual = [], [], [], []
    for product in products:
        for region in regions:
            _, forecast, _ = train_prophet_for_series(raw, product, region)
            sub = raw[(raw["product_id"] == product) & (raw["region"] == region)]
            merged = sub.rename(columns={"date": "ds", "sales": "y"}).merge(forecast, on="ds")
            all_point.append(merged["yhat"].values)
            all_lower.append(merged["yhat_lower"].values)
            all_upper.append(merged["yhat_upper"].values)
            all_actual.append(merged["y"].values)

    point = np.concatenate(all_point)
    lower = np.concatenate(all_lower)
    upper = np.concatenate(all_upper)
    actual = np.concatenate(all_actual)

    return _evaluate_alpha_grid("prophet", point, lower, upper, actual, alpha_grid)


def calibrate_lstm(alpha_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), n_passes: int = 15) -> dict:
    from lstm_model import train_lstm_for_series, make_windows, WINDOW, predict_with_uncertainty
    from data_loader import load_raw_sales
    from sklearn.preprocessing import StandardScaler

    raw = load_raw_sales()
    products = sorted(raw["product_id"].unique())
    regions = sorted(raw["region"].unique())

    all_point, all_lower, all_upper, all_actual = [], [], [], []
    z = 1.28
    for product in products:
        for region in regions:
            model, scaler, _ = train_lstm_for_series(raw, product, region, epochs=15)
            sub = raw[(raw["product_id"] == product) & (raw["region"] == region)].sort_values("date")
            values = sub["sales"].values.astype("float32")
            scaled = scaler.transform(values.reshape(-1, 1)).flatten()
            X, y = make_windows(scaled, WINDOW)
            split = int(len(X) * 0.85)
            X_test, y_test = X[split:], y[split:]

            mean_s, std_s, _ = predict_with_uncertainty(model, X_test, n_passes=n_passes)
            point = scaler.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            lower = scaler.inverse_transform((mean_s - z * std_s).reshape(-1, 1)).flatten()
            upper = scaler.inverse_transform((mean_s + z * std_s).reshape(-1, 1)).flatten()
            actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            all_point.append(point); all_lower.append(lower)
            all_upper.append(upper); all_actual.append(actual)

    point = np.concatenate(all_point)
    lower = np.concatenate(all_lower)
    upper = np.concatenate(all_upper)
    actual = np.concatenate(all_actual)

    return _evaluate_alpha_grid("lstm", point, lower, upper, actual, alpha_grid)


def _evaluate_alpha_grid(model_type: str, point, lower, upper, actual, alpha_grid) -> dict:
    lower = np.minimum(lower, upper)
    upper = np.maximum(lower, upper)
    ape = np.array([_ape(a, p) for a, p in zip(actual, point)])
    rel_width = (upper - lower) / np.maximum(np.abs(point), 1e-6)

    best = {"alpha": None, "correlation": 1.0}  # correlation should be negative; start high
    for alpha in alpha_grid:
        confidence = np.array([_width_to_confidence(rw, alpha) for rw in rel_width])
        corr = float(np.corrcoef(confidence, ape)[0, 1])
        if corr < best["correlation"]:
            best = {"alpha": alpha, "correlation": corr}

    # reliability table at the best alpha
    confidence = np.array([_width_to_confidence(rw, best["alpha"]) for rw in rel_width])
    bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    labels = ["<0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", "0.95-1.00"]
    bucket = pd.cut(confidence, bins=bins, labels=labels, right=False)
    reliability = pd.DataFrame({"confidence": confidence, "ape": ape, "bucket": bucket})
    table = reliability.groupby("bucket", observed=True)["ape"].agg(["mean", "count"]).reset_index()

    print(f"\n--- {model_type} calibration ---")
    print(f"Best alpha: {best['alpha']}  (confidence-vs-APE correlation: {best['correlation']:.3f})")
    print("Reliability table (mean APE should decrease as confidence bucket increases):")
    print(table.to_string(index=False))

    return {
        "model_type": model_type,
        "best_alpha": best["alpha"],
        "correlation": best["correlation"],
        "reliability_table": table.to_dict(orient="records"),
    }


if __name__ == "__main__":
    results = {}
    results["xgboost"] = calibrate_xgboost()
    results["prophet"] = calibrate_prophet()
    results["lstm"] = calibrate_lstm()

    print("\n=== Suggested threshold_config.yaml alpha values ===")
    for model_type, r in results.items():
        print(f"{model_type}: alpha = {r['best_alpha']}  (corr = {r['correlation']:.3f})")
