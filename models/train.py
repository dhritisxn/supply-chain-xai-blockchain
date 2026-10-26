"""
train.py

Orchestrates training of all three forecasting models (XGBoost, Prophet,
LSTM) under the same evaluation protocol and produces the multi-model
benchmark table referenced in the Review-1 report's "Model Evaluation &
XAI Strategy" slide.

Run:
    python train.py                 # trains all three, saves models + metrics
    python train.py --skip-lstm     # faster iteration while tuning XGB/Prophet
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from xgboost_model import train_xgboost
from prophet_model import train_all_prophet_models
from lstm_model import train_all_lstm_models

RESULTS_PATH = Path(__file__).parent / "saved_models" / "benchmark_summary.json"


def main(skip_lstm: bool = False, lstm_epochs: int = 25):
    summary = {}

    print("=== Training XGBoost ===")
    xgb_result = train_xgboost()
    summary["xgboost"] = xgb_result["metrics"]
    print(xgb_result["metrics"])

    print("\n=== Training Prophet (per product x region series) ===")
    _, prophet_metrics, prophet_summary = train_all_prophet_models()
    summary["prophet"] = prophet_summary
    print(prophet_summary)

    if not skip_lstm:
        print("\n=== Training LSTM (per product x region series) ===")
        _, lstm_summary = train_all_lstm_models(epochs=lstm_epochs)
        summary["lstm"] = lstm_summary
        print(lstm_summary)
    else:
        print("\n=== Skipping LSTM ===")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Benchmark Summary ===")
    print(f"{'Model':<10} {'RMSE':>10} {'MAPE':>10}")
    print(f"{'XGBoost':<10} {summary['xgboost']['rmse']:>10.2f} {summary['xgboost']['mape']:>10.4f}")
    print(f"{'Prophet':<10} {summary['prophet']['avg_rmse']:>10.2f} {summary['prophet']['avg_mape']:>10.4f}")
    if not skip_lstm:
        print(f"{'LSTM':<10} {summary['lstm']['avg_rmse']:>10.2f} {summary['lstm']['avg_mape']:>10.4f}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--lstm-epochs", type=int, default=25)
    args = parser.parse_args()
    main(skip_lstm=args.skip_lstm, lstm_epochs=args.lstm_epochs)
