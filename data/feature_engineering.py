"""
feature_engineering.py

Builds the tabular feature set used by XGBoost (and available for reuse
elsewhere). Prophet and the LSTM use their own lighter-weight input prep
(see models/prophet_model.py and models/lstm_model.py) since they consume
raw time-ordered sequences rather than engineered tabular features.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


CATEGORICAL_COLS = ["product_id", "region"]

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 30]


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_lag_features(df: pd.DataFrame, group_cols=("product_id", "region")) -> pd.DataFrame:
    group_cols = list(group_cols)
    df = df.sort_values(group_cols + ["date"]).copy()
    grouped = df.groupby(group_cols)["sales"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # shift-by-1 first (so rolling stats never see the current day's own sales),
    # then roll *within each group* -- must re-group the shifted series itself.
    df["_shifted_1"] = grouped.shift(1)
    shifted_grouped = df.groupby(group_cols)["_shifted_1"]

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            shifted_grouped.rolling(window).mean().reset_index(level=list(range(len(group_cols))), drop=True)
        )
        df[f"rolling_std_{window}"] = (
            shifted_grouped.rolling(window).std().reset_index(level=list(range(len(group_cols))), drop=True)
        )

    df = df.drop(columns=["_shifted_1"])
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        df[col + "_enc"] = df[col].map(mapping)
        encoders[col] = mapping
    return df, encoders


def build_feature_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full tabular feature-engineering pipeline for XGBoost.
    Returns (feature_df_with_target, categorical_encoders).
    Drops the initial rows per series that don't have enough history for lags.
    """
    df = build_calendar_features(raw_df)
    df = build_lag_features(df)
    df, encoders = encode_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    return df, encoders


FEATURE_COLS = (
    [f"lag_{l}" for l in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
       "is_holiday", "is_promo", "price", "product_id_enc", "region_enc"]
)
TARGET_COL = "sales"


if __name__ == "__main__":
    from data_loader import load_raw_sales
    raw = load_raw_sales()
    feat, enc = build_feature_table(raw)
    print(feat.shape)
    print(feat[FEATURE_COLS + [TARGET_COL]].head())
