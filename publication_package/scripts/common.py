"""
common.py

Shared helpers for the publication_package experiment scripts.

Everything here is READ-ONLY with respect to the existing pipeline: it loads
the already-saved model artefacts in models/saved_models/ and the cached
dataset in data/raw/, and calls the existing confidence formula
(confidence/confidence_estimator.score_from_interval) rather than
re-implementing it. Nothing in this module writes outside
publication_package/.

Held-out test sets are exactly the ones each model was evaluated on in the
original code:
  - XGBoost : models/xgboost_model.time_based_split (last 15% of dates in the
              feature table, pooled across all 9 series)
  - Prophet : models/prophet_model.train_prophet_for_series (dates above the
              85% date quantile of each series)
  - LSTM    : models/lstm_model.train_lstm_for_series (last 15% of 21-day
              windows of each series)
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")
warnings.filterwarnings("ignore")

import logging

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "publication_package"
INTER = PKG / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

for sub in ["models", "data", "confidence", "inventory", "decision_gate",
            "explainability", "blockchain/scripts"]:
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

SEED = 42
PRODUCTS = ["SKU_A", "SKU_B", "SKU_C"]
REGIONS = ["North", "South", "West"]
SERIES = [(p, r) for p in PRODUCTS for r in REGIONS]
SAVED = ROOT / "models" / "saved_models"

# "Bad forecast" tolerance used by the gate analysis (task 2) and the
# AUROC-style discrimination metric reported in tasks 3/4.
APE_TOLERANCE = 0.10


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_raw() -> pd.DataFrame:
    from data_loader import load_raw_sales
    return load_raw_sales()


def config():
    from confidence_estimator import _load_config
    return _load_config()


def confidence_vec(point, lower, upper, alpha: float) -> np.ndarray:
    """Apply the project's own score_from_interval() row by row, so the
    confidence values used in every analysis are byte-for-byte the ones the
    live pipeline would produce."""
    from confidence_estimator import score_from_interval
    out = np.empty(len(point))
    for i, (p, lo, hi) in enumerate(zip(point, lower, upper)):
        lo, hi = min(lo, hi), max(lo, hi)
        out[i] = score_from_interval(float(p), float(lo), float(hi), alpha,
                                     model_type="", method="").confidence
    return out


def context_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (series, date) context known at forecast time: promo/holiday flags
    and the trailing 7-day coefficient of variation (computed on sales up to
    and including the previous day -- no leakage of the target day)."""
    df = raw.sort_values(["product_id", "region", "date"]).copy()
    g = df.groupby(["product_id", "region"])["sales"]
    shifted = g.shift(1)
    roll_mean = shifted.groupby([df["product_id"], df["region"]]).rolling(7).mean().reset_index(level=[0, 1], drop=True)
    roll_std = shifted.groupby([df["product_id"], df["region"]]).rolling(7).std().reset_index(level=[0, 1], drop=True)
    df["cv7"] = roll_std / roll_mean
    df["day_of_week"] = df["date"].dt.dayofweek
    return df[["product_id", "region", "date", "is_promo", "is_holiday", "day_of_week", "cv7"]]


# ---------------------------------------------------------------------------
#  Test-set predictions from the SAVED models (no retraining)
# ---------------------------------------------------------------------------

def xgb_test_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    from feature_engineering import build_feature_table, FEATURE_COLS, TARGET_COL
    from xgboost_model import time_based_split, load_xgboost_models

    feat, _ = build_feature_table(raw)
    _, test_df = time_based_split(feat)
    point_m, lo_m, hi_m, _ = load_xgboost_models()
    X = test_df[FEATURE_COLS]
    out = pd.DataFrame({
        "model": "xgboost",
        "product_id": test_df["product_id"].values,
        "region": test_df["region"].values,
        "date": test_df["date"].values,
        "actual": test_df[TARGET_COL].values.astype(float),
        "point": point_m.predict(X).astype(float),
        "lower": lo_m.predict(X).astype(float),
        "upper": hi_m.predict(X).astype(float),
    })
    return out


def prophet_test_predictions(raw: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    from prophet.serialize import model_from_json

    frames = []
    for product, region in SERIES:
        key = f"{product}__{region}"
        with open(SAVED / "prophet" / f"{key}.json") as f:
            m = model_from_json(f.read())
        sub = raw[(raw["product_id"] == product) & (raw["region"] == region)].sort_values("date")
        df = sub.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", "is_promo"]]
        cutoff = df["ds"].quantile(1 - 0.15)
        test = df[df["ds"] > cutoff]
        np.random.seed(seed)  # Prophet's interval is Monte-Carlo sampled
        fc = m.predict(test[["ds", "is_promo"]].copy())
        merged = test.merge(fc, on="ds")
        frames.append(pd.DataFrame({
            "model": "prophet", "product_id": product, "region": region,
            "date": merged["ds"].values, "actual": merged["y"].values.astype(float),
            "point": merged["yhat"].values, "lower": merged["yhat_lower"].values,
            "upper": merged["yhat_upper"].values,
        }))
    return pd.concat(frames, ignore_index=True)


def load_lstm(product: str, region: str):
    import keras
    from lstm_model import AdditiveAttention
    key = f"{product}__{region}"
    model = keras.models.load_model(SAVED / "lstm" / f"{key}.keras",
                                    custom_objects={"AdditiveAttention": AdditiveAttention},
                                    compile=False)
    mean = float(np.load(SAVED / "lstm" / f"{key}_scaler_mean.npy")[0])
    scale = float(np.load(SAVED / "lstm" / f"{key}_scaler_scale.npy")[0])
    return model, mean, scale


def lstm_windows(raw: pd.DataFrame, product: str, region: str, mean: float, scale: float,
                 test_frac: float = 0.15):
    """Reproduces train_lstm_for_series' windowing/split exactly."""
    from lstm_model import make_windows, WINDOW
    sub = raw[(raw["product_id"] == product) & (raw["region"] == region)].sort_values("date")
    values = sub["sales"].values.astype("float32")
    scaled = ((values - mean) / scale).astype("float32")
    X, y = make_windows(scaled, WINDOW)
    split = int(len(X) * (1 - test_frac))
    dates = sub["date"].values[WINDOW:]
    return X, y, split, dates


def lstm_test_predictions_mc(raw: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Current production LSTM uncertainty: MC-Dropout on the saved models,
    with n_passes / z_multiplier taken from threshold_config.yaml."""
    from lstm_model import predict_with_uncertainty
    cfg = config()["lstm"]
    n_passes, z = cfg["mc_dropout_passes"], cfg["z_multiplier"]

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    frames = []
    for product, region in SERIES:
        # Keras 3 Dropout layers draw their RNG seed when the model is built/loaded,
        # so the global seed must be set BEFORE loading for MC-Dropout to be repeatable.
        set_seeds(seed)
        model, mean, scale = load_lstm(product, region)
        X, y, split, dates = lstm_windows(raw, product, region, mean, scale)
        X_test, y_test = X[split:], y[split:]
        det, _ = model.predict(X_test, verbose=0)  # deterministic point forecast (as in metrics.json)
        set_seeds(seed)
        mc_mean, mc_std, _ = predict_with_uncertainty(model, X_test, n_passes=n_passes)
        frames.append(pd.DataFrame({
            "model": "lstm", "product_id": product, "region": region,
            "date": dates[split:], "actual": y_test * scale + mean,
            "point_deterministic": det.flatten() * scale + mean,
            "point": mc_mean * scale + mean,
            "lower": (mc_mean - z * mc_std) * scale + mean,
            "upper": (mc_mean + z * mc_std) * scale + mean,
            "std": mc_std * scale,
        }))
    return pd.concat(frames, ignore_index=True)


def cached(name: str, builder, refresh: bool = False) -> pd.DataFrame:
    path = INTER / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    df = builder()
    df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])


def ape(actual, pred) -> np.ndarray:
    actual = np.asarray(actual, float)
    return np.abs(actual - np.asarray(pred, float)) / np.maximum(np.abs(actual), 1e-6)


def auroc(scores, labels) -> float:
    """AUROC of `scores` for detecting labels==1 (ties handled by average rank)."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_metrics(point, lower, upper, actual, alpha: float, label: str) -> dict:
    """Confidence-vs-error diagnostics used for tasks 3 and 4."""
    from scipy.stats import pearsonr, spearmanr
    # round away float noise so mathematically tied scores rank as ties
    conf = np.round(confidence_vec(point, lower, upper, alpha), 10)
    e = ape(actual, point)
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    actual = np.asarray(actual, float)
    pr = pearsonr(conf, e)
    sr = spearmanr(conf, e)
    return {
        "method": label,
        "n": int(len(e)),
        "alpha": alpha,
        "pearson_conf_vs_ape": float(pr.statistic),
        "pearson_p_value": float(pr.pvalue),
        "spearman_conf_vs_ape": float(sr.statistic),
        "spearman_p_value": float(sr.pvalue),
        f"auroc_low_conf_detects_ape_gt_{int(APE_TOLERANCE*100)}pct": auroc(-conf, e > APE_TOLERANCE),
        "interval_coverage": float(np.mean((actual >= lo) & (actual <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
        "std_interval_width": float(np.std(hi - lo)),
        "mean_confidence": float(np.mean(conf)),
        "frac_conf_ge_0.85": float(np.mean(conf >= 0.85)),
        "rmse": float(np.sqrt(np.mean((actual - np.asarray(point)) ** 2))),
        "mape": float(np.mean(e)),
    }
