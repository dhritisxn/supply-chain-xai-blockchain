"""
prophet_model.py

Trains one Prophet model per (product_id, region) series. Prophet is
self-explaining -- its native `predict()` output already decomposes the
forecast into trend / weekly / yearly / holiday components, and it produces
`yhat_lower` / `yhat_upper` uncertainty intervals out of the box, which the
Confidence Estimation Module (Phase 2) will use directly.
"""

from __future__ import annotations
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

sys.path.append(str(Path(__file__).parent.parent / "data"))
from data_loader import load_raw_sales, HOLIDAYS

SAVED_DIR = Path(__file__).parent / "saved_models" / "prophet"


def _holidays_df() -> pd.DataFrame:
    return pd.DataFrame({
        "holiday": "public_holiday",
        "ds": HOLIDAYS,
        "lower_window": 0,
        "upper_window": 0,
    })


def _series_to_prophet_frame(raw: pd.DataFrame, product_id: str, region: str) -> pd.DataFrame:
    sub = raw[(raw["product_id"] == product_id) & (raw["region"] == region)].sort_values("date")
    df = sub.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", "is_promo"]]
    return df


def train_prophet_for_series(raw: pd.DataFrame, product_id: str, region: str, test_frac: float = 0.15):
    df = _series_to_prophet_frame(raw, product_id, region)
    cutoff = df["ds"].quantile(1 - test_frac)
    train_df, test_df = df[df["ds"] <= cutoff], df[df["ds"] > cutoff]

    model = Prophet(
        holidays=_holidays_df(),
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,  # 80% uncertainty interval -> yhat_lower/yhat_upper
    )
    model.add_regressor("is_promo")
    model.fit(train_df)

    future = test_df[["ds", "is_promo"]].copy()
    forecast = model.predict(future)

    merged = test_df.merge(forecast, on="ds")
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))),
        "mape": float(mean_absolute_percentage_error(merged["y"], merged["yhat"])),
    }
    return model, forecast, metrics


def train_all_prophet_models(save: bool = True):
    raw = load_raw_sales()
    products = sorted(raw["product_id"].unique())
    regions = sorted(raw["region"].unique())

    models, all_metrics = {}, {}
    for product in products:
        for region in regions:
            key = f"{product}__{region}"
            model, forecast, metrics = train_prophet_for_series(raw, product, region)
            models[key] = model
            all_metrics[key] = metrics
            print(f"  {key}: RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.4f}")

    if save:
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        for key, model in models.items():
            with open(SAVED_DIR / f"{key}.json", "w") as f:
                f.write(model_to_json(model))
        with open(SAVED_DIR / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

    avg_rmse = float(np.mean([m["rmse"] for m in all_metrics.values()]))
    avg_mape = float(np.mean([m["mape"] for m in all_metrics.values()]))
    return models, all_metrics, {"avg_rmse": avg_rmse, "avg_mape": avg_mape}


def model_to_json(model: Prophet) -> str:
    from prophet.serialize import model_to_json as _to_json
    return _to_json(model)


def model_from_json(path: Path) -> Prophet:
    from prophet.serialize import model_from_json as _from_json
    with open(path) as f:
        return _from_json(f.read())


def get_decomposition(model: Prophet, forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the native additive decomposition (trend + weekly + yearly +
    holiday components) already computed by Prophet's predict() output --
    this IS the explainability layer for Prophet, no extra library needed.
    """
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
    for optional in ["weekly", "yearly", "holidays", "public_holiday"]:
        if optional in forecast.columns:
            cols.append(optional)
    return forecast[cols]


if __name__ == "__main__":
    models, metrics, summary = train_all_prophet_models()
    print("Prophet summary:", summary)
