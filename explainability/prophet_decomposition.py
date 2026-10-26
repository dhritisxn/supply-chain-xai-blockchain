"""
prophet_decomposition.py

Prophet is self-explaining: model.predict() already returns the additive
decomposition (trend, weekly, yearly, holiday components) that sum to
yhat. This module just standardizes that output into the same explanation
shape used by the SHAP explainer and LSTM attention module, so the
Confidence Estimation / Decision Gate / blockchain audit layers can treat
all three models' explanations uniformly.
"""

from __future__ import annotations
import pandas as pd


def explain_forecast_row(forecast_row: pd.Series) -> dict:
    """
    forecast_row: a single row from Prophet's model.predict(future) output.
    Returns a structured explanation matching the SHAP explainer's shape:
    {predicted_value, components (contribution per named component)}.
    """
    components = {}
    for col in ["trend", "weekly", "yearly", "holidays", "public_holiday", "extra_regressors_additive"]:
        if col in forecast_row.index:
            components[col] = float(forecast_row[col])

    sorted_components = dict(
        sorted(components.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

    return {
        "predicted_value": float(forecast_row["yhat"]),
        "lower_bound": float(forecast_row["yhat_lower"]),
        "upper_bound": float(forecast_row["yhat_upper"]),
        "contributions": sorted_components,
        "top_drivers": list(sorted_components.items())[:5],
    }
