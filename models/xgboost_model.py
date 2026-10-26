"""
xgboost_model.py

Trains an XGBoost regressor on the engineered tabular features to forecast
daily demand per (product_id, region). Also trains lower/upper quantile
regressors on the same features -- these aren't used yet (Phase 1 is just
forecasting), but the Confidence Estimation Module (Phase 2) will consume
them directly to derive a prediction-interval-based confidence score, so
they're produced here where the rest of the XGBoost training already lives.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

sys.path.append(str(Path(__file__).parent.parent / "data"))
from data_loader import load_raw_sales
from feature_engineering import build_feature_table, FEATURE_COLS, TARGET_COL

SAVED_DIR = Path(__file__).parent / "saved_models"

POINT_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

# Quantile params reuse the same tree structure, just swap the objective/alpha.
QUANTILE_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:quantileerror",
    random_state=42,
)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.15):
    """Chronological split (not random) -- correct for time-series evaluation."""
    df = df.sort_values("date")
    cutoff = df["date"].quantile(1 - test_frac)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def train_xgboost(save: bool = True):
    raw = load_raw_sales()
    feat, encoders = build_feature_table(raw)
    train_df, test_df = time_based_split(feat)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = xgb.XGBRegressor(**POINT_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    # Quantile models (used later by the confidence module for interval width)
    lower_model = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=0.1)
    upper_model = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=0.9)
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    if save:
        SAVED_DIR.mkdir(exist_ok=True)
        model.save_model(SAVED_DIR / "xgboost_point.json")
        lower_model.save_model(SAVED_DIR / "xgboost_q10.json")
        upper_model.save_model(SAVED_DIR / "xgboost_q90.json")
        import json
        with open(SAVED_DIR / "xgboost_encoders.json", "w") as f:
            json.dump(encoders, f)

    return {
        "model": model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "encoders": encoders,
        "metrics": metrics,
        "test_df": test_df,
        "preds": preds,
    }


def load_xgboost_models():
    point = xgb.XGBRegressor()
    point.load_model(SAVED_DIR / "xgboost_point.json")
    lower = xgb.XGBRegressor()
    lower.load_model(SAVED_DIR / "xgboost_q10.json")
    upper = xgb.XGBRegressor()
    upper.load_model(SAVED_DIR / "xgboost_q90.json")
    import json
    with open(SAVED_DIR / "xgboost_encoders.json") as f:
        encoders = json.load(f)
    return point, lower, upper, encoders


if __name__ == "__main__":
    result = train_xgboost()
    print("XGBoost metrics:", result["metrics"])
