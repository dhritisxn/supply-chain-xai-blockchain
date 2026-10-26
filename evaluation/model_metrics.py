"""
model_metrics.py

Computes and reports per-model forecasting performance metrics:
  - RMSE  (Root Mean Squared Error)
  - MAPE  (Mean Absolute Percentage Error)
  - R-squared (coefficient of determination)

Two modes:
  1. Report saved metrics (fast, no retraining) -- reads from saved_models/
  2. Live evaluation (slower, re-runs models on test split) -- recomputes from data

Run:
    python evaluation/model_metrics.py                # saved metrics only
    python evaluation/model_metrics.py --live          # re-evaluate from scratch
    python evaluation/model_metrics.py --output report # save to docs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

PRODUCTS = ["SKU_A", "SKU_B", "SKU_C"]
REGIONS = ["North", "South", "West"]


# ---------------------------------------------------------------------------
#  1. Load saved metrics (fast path)
# ---------------------------------------------------------------------------

def load_saved_metrics() -> dict:
    """
    Load pre-computed metrics from saved_models/ for all three models.
    Returns a dict keyed by model_type with per-series and aggregate stats.
    """
    results = {}

    # --- XGBoost ---
    xgb_path = MODELS_DIR / "benchmark_summary.json"
    if xgb_path.exists():
        with open(xgb_path) as f:
            bench = json.load(f)
        results["xgboost"] = {
            "aggregate": {
                "rmse": bench.get("xgboost", {}).get("rmse"),
                "mape": bench.get("xgboost", {}).get("mape"),
                "r2": bench.get("xgboost", {}).get("r2"),
            },
            "per_series": {},  # XGBoost is trained on all series jointly
        }

    # --- Prophet ---
    prophet_metrics_path = MODELS_DIR / "prophet" / "metrics.json"
    if prophet_metrics_path.exists():
        with open(prophet_metrics_path) as f:
            prophet_all = json.load(f)
        per_series = {}
        for key, m in prophet_all.items():
            product, region = key.split("__")
            per_series[key] = {"rmse": m["rmse"], "mape": m["mape"]}
        avg_rmse = float(np.mean([v["rmse"] for v in per_series.values()]))
        avg_mape = float(np.mean([v["mape"] for v in per_series.values()]))
        results["prophet"] = {
            "aggregate": {"rmse": avg_rmse, "mape": avg_mape, "r2": None},
            "per_series": per_series,
        }

    # --- LSTM ---
    lstm_metrics_path = MODELS_DIR / "lstm" / "metrics.json"
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path) as f:
            lstm_all = json.load(f)
        per_series = {}
        for key, m in lstm_all.items():
            per_series[key] = {"rmse": m["rmse"], "mape": m["mape"]}
        avg_rmse = float(np.mean([v["rmse"] for v in per_series.values()]))
        avg_mape = float(np.mean([v["mape"] for v in per_series.values()]))
        results["lstm"] = {
            "aggregate": {"rmse": avg_rmse, "mape": avg_mape, "r2": None},
            "per_series": per_series,
        }

    return results


# ---------------------------------------------------------------------------
#  2. Live evaluation (slow path -- re-runs models on test split)
# ---------------------------------------------------------------------------

def evaluate_xgboost_live() -> dict:
    """Re-evaluate XGBoost on a chronological test split."""
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

    from data_loader import load_raw_sales
    from feature_engineering import build_feature_table, FEATURE_COLS, TARGET_COL
    from xgboost_model import time_based_split, load_xgboost_models
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

    raw = load_raw_sales()
    feat, _ = build_feature_table(raw)
    train_df, test_df = time_based_split(feat)

    point_model, _, _, _ = load_xgboost_models()
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    preds = point_model.predict(X_test)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def evaluate_prophet_live() -> dict:
    """Re-evaluate Prophet on each series."""
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

    from data_loader import load_raw_sales
    from prophet_model import train_prophet_for_series

    raw = load_raw_sales()
    per_series = {}
    for product in PRODUCTS:
        for region in REGIONS:
            key = f"{product}__{region}"
            try:
                _, _, metrics = train_prophet_for_series(raw, product, region)
                per_series[key] = metrics
            except Exception as e:
                per_series[key] = {"error": str(e)}

    valid = {k: v for k, v in per_series.items() if "rmse" in v}
    avg_rmse = float(np.mean([v["rmse"] for v in valid.values()])) if valid else None
    avg_mape = float(np.mean([v["mape"] for v in valid.values()])) if valid else None
    return {"aggregate": {"rmse": avg_rmse, "mape": avg_mape}, "per_series": per_series}


def evaluate_lstm_live() -> dict:
    """Re-evaluate LSTM on each series."""
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

    from data_loader import load_raw_sales
    from lstm_model import train_lstm_for_series

    raw = load_raw_sales()
    per_series = {}
    for product in PRODUCTS:
        for region in REGIONS:
            key = f"{product}__{region}"
            try:
                _, _, metrics = train_lstm_for_series(raw, product, region, epochs=15)
                per_series[key] = metrics
            except Exception as e:
                per_series[key] = {"error": str(e)}

    valid = {k: v for k, v in per_series.items() if "rmse" in v}
    avg_rmse = float(np.mean([v["rmse"] for v in valid.values()])) if valid else None
    avg_mape = float(np.mean([v["mape"] for v in valid.values()])) if valid else None
    return {"aggregate": {"rmse": avg_rmse, "mape": avg_mape}, "per_series": per_series}


# ---------------------------------------------------------------------------
#  3. Report formatting
# ---------------------------------------------------------------------------

def print_report(results: dict, title: str = "Model Performance Metrics"):
    """Print a formatted report to stdout."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Aggregate comparison table
    print(f"\n{'Model':12s} {'RMSE':>10s} {'MAPE':>10s} {'R-squared':>12s}")
    print("-" * 46)
    for model_name in ["xgboost", "prophet", "lstm"]:
        if model_name in results:
            agg = results[model_name].get("aggregate", {})
            rmse_str = f"{agg['rmse']:10.2f}" if agg.get("rmse") is not None else f"{'N/A':>10s}"
            mape_str = f"{agg['mape']:10.4f}" if agg.get("mape") is not None else f"{'N/A':>10s}"
            r2_str = f"{agg['r2']:12.4f}" if agg.get("r2") is not None else f"{'N/A':>12s}"
            print(f"{model_name:12s} {rmse_str} {mape_str} {r2_str}")

    # Per-series breakdown (Prophet and LSTM)
    for model_name in ["prophet", "lstm"]:
        if model_name in results and results[model_name].get("per_series"):
            print(f"\n--- {model_name.upper()} Per-Series Breakdown ---")
            print(f"{'Series':20s} {'RMSE':>10s} {'MAPE':>10s}")
            print("-" * 42)
            for key, m in sorted(results[model_name]["per_series"].items()):
                if "error" in m:
                    print(f"{key:20s}  ERROR: {m['error']}")
                else:
                    print(f"{key:20s} {m['rmse']:10.2f} {m['mape']:10.4f}")

    # Key findings
    print(f"\n--- Key Findings ---")
    xgb_r2 = results.get("xgboost", {}).get("aggregate", {}).get("r2")
    if xgb_r2:
        print(f"  XGBoost R-squared: {xgb_r2:.4f} (explains {100*xgb_r2:.1f}% of variance)")

    xgb_mape = results.get("xgboost", {}).get("aggregate", {}).get("mape")
    prophet_mape = results.get("prophet", {}).get("aggregate", {}).get("mape")
    lstm_mape = results.get("lstm", {}).get("aggregate", {}).get("mape")

    if all(v is not None for v in [xgb_mape, prophet_mape, lstm_mape]):
        best = min(
            [("XGBoost", xgb_mape), ("Prophet", prophet_mape), ("LSTM", lstm_mape)],
            key=lambda x: x[1],
        )
        print(f"  Best MAPE: {best[0]} ({best[1]:.4f} = {best[1]*100:.2f}%)")

    print("="*70)


# ---------------------------------------------------------------------------
#  4. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model performance metrics")
    parser.add_argument("--live", action="store_true", help="Re-evaluate from scratch (slow)")
    parser.add_argument("--output", default=None, help="Save results to docs/ directory")
    args = parser.parse_args()

    if args.live:
        print("Running live evaluation (this may take a few minutes)...")
        results = {}
        try:
            results["xgboost"] = {"aggregate": evaluate_xgboost_live(), "per_series": {}}
        except Exception as e:
            print(f"XGBoost evaluation failed: {e}")
        try:
            results["prophet"] = evaluate_prophet_live()
        except Exception as e:
            print(f"Prophet evaluation failed: {e}")
        try:
            results["lstm"] = evaluate_lstm_live()
        except Exception as e:
            print(f"LSTM evaluation failed: {e}")
        print_report(results, "Model Performance Metrics (Live Evaluation)")
    else:
        results = load_saved_metrics()
        print_report(results, "Model Performance Metrics (Saved)")

    if args.output:
        out_dir = PROJECT_ROOT / args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
