"""
confidence_estimator.py

Phase 2: Confidence Estimation Module.

Turns each model's own native uncertainty signal into a single 0-1
confidence score using the SAME functional form across all three models,
so the Decision Gate (Phase 4) can threshold them uniformly regardless of
which model produced a given forecast:

    relative_width = interval_width / max(|point_forecast|, eps)
    confidence     = exp(-alpha * relative_width), clipped to [0, 1]

The interval itself comes from each model's own existing uncertainty
mechanism -- nothing here retrains or wraps the forecasting models:

  - XGBoost : q10/q90 quantile regressors already trained in
              models/xgboost_model.py -> interval = q90 - q10
  - Prophet : native yhat_lower/yhat_upper from model.predict()
              (interval_width=0.8, already configured in prophet_model.py)
  - LSTM    : MC-Dropout via models/lstm_model.py::predict_with_uncertainty
              -> interval = 2 * z_multiplier * std across stochastic passes

This module answers ONLY "how much should we trust this specific
forecast." It does not decide autonomous-vs-human action -- that's the
Decision Gate (Phase 4), which will import `get_threshold()` from here
rather than hard-coding 0.85 itself.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

CONFIG_PATH = Path(__file__).parent / "threshold_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_CONFIG = _load_config()


def get_threshold() -> float:
    """
    Single source of truth for the autonomous-execution confidence
    threshold (CONFIDENCE_THRESHOLD in threshold_config.yaml). Always call
    this instead of hard-coding 0.85 anywhere else -- Decision Gate,
    dashboard, evaluation scripts, etc. Re-reads the small YAML file each
    call so a config edit takes effect without restarting long-lived
    processes (backend/dashboard).
    """
    return _load_config()["CONFIDENCE_THRESHOLD"]


@dataclass
class ConfidenceResult:
    model_type: str
    point_forecast: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    relative_width: float
    confidence: float
    method: str
    metadata: dict = field(default_factory=dict)

    def meets_threshold(self, threshold: Optional[float] = None) -> bool:
        threshold = threshold if threshold is not None else get_threshold()
        return self.confidence >= threshold

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "point_forecast": round(self.point_forecast, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "interval_width": round(self.interval_width, 4),
            "relative_width": round(self.relative_width, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "metadata": self.metadata,
        }


def _width_to_confidence(relative_width: float, alpha: float) -> float:
    """
    Monotonically decreasing map: a wider interval (relative to the
    forecast's own scale) yields lower confidence. Exponential decay keeps
    the score smoothly bounded in (0, 1] rather than applying a hard
    cutoff, and `alpha` lets each model's decay rate be calibrated
    independently -- see calibrate.py.
    """
    conf = float(np.exp(-alpha * relative_width))
    return float(np.clip(conf, 0.0, 1.0))


def score_from_interval(point_forecast: float, lower: float, upper: float,
                         alpha: float, model_type: str, method: str,
                         metadata: Optional[dict] = None) -> ConfidenceResult:
    eps = 1e-6
    width = max(upper - lower, 0.0)
    rel_width = width / max(abs(point_forecast), eps)
    confidence = _width_to_confidence(rel_width, alpha)
    return ConfidenceResult(
        model_type=model_type,
        point_forecast=point_forecast,
        lower_bound=lower,
        upper_bound=upper,
        interval_width=width,
        relative_width=rel_width,
        confidence=confidence,
        method=method,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Per-model estimators
# ---------------------------------------------------------------------------

class XGBoostConfidenceEstimator:
    """Uses the q10/q90 quantile models already trained alongside the point model."""

    def __init__(self, point_model=None, lower_model=None, upper_model=None,
                 alpha: Optional[float] = None):
        if point_model is None or lower_model is None or upper_model is None:
            from xgboost_model import load_xgboost_models
            point_model, lower_model, upper_model, _ = load_xgboost_models()
        self.point_model = point_model
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.alpha = alpha if alpha is not None else _CONFIG["xgboost"]["alpha"]

    def estimate(self, feature_row) -> ConfidenceResult:
        import pandas as pd
        from feature_engineering import FEATURE_COLS
        X = pd.DataFrame([feature_row[FEATURE_COLS].astype(float).to_dict()], columns=FEATURE_COLS)

        point = float(self.point_model.predict(X)[0])
        lower = float(self.lower_model.predict(X)[0])
        upper = float(self.upper_model.predict(X)[0])
        # quantile models can rarely cross out of order -- guard before scoring
        lower, upper = min(lower, upper), max(lower, upper)

        return score_from_interval(
            point, lower, upper, self.alpha,
            model_type="xgboost", method="quantile_regression_q10_q90",
            metadata={"quantiles": [0.1, 0.9]},
        )


class ProphetConfidenceEstimator:
    """Uses Prophet's native yhat_lower/yhat_upper from model.predict()."""

    def __init__(self, alpha: Optional[float] = None):
        self.alpha = alpha if alpha is not None else _CONFIG["prophet"]["alpha"]

    def estimate(self, forecast_row) -> ConfidenceResult:
        point = float(forecast_row["yhat"])
        lower = float(forecast_row["yhat_lower"])
        upper = float(forecast_row["yhat_upper"])

        return score_from_interval(
            point, lower, upper, self.alpha,
            model_type="prophet", method="native_uncertainty_interval",
            metadata={"interval_width_pct": 0.8},
        )


class LSTMConfidenceEstimator:
    """Uses MC-Dropout (stochastic forward passes) already wired in models/lstm_model.py."""

    def __init__(self, alpha: Optional[float] = None, n_passes: Optional[int] = None,
                 z_multiplier: Optional[float] = None):
        cfg = _CONFIG["lstm"]
        self.alpha = alpha if alpha is not None else cfg["alpha"]
        self.n_passes = n_passes if n_passes is not None else cfg["mc_dropout_passes"]
        self.z_multiplier = z_multiplier if z_multiplier is not None else cfg["z_multiplier"]

    def estimate(self, model, X_window: np.ndarray, scaler) -> ConfidenceResult:
        from lstm_model import predict_with_uncertainty
        mean_scaled, std_scaled, attn = predict_with_uncertainty(model, X_window, n_passes=self.n_passes)

        point = float(scaler.inverse_transform(mean_scaled.reshape(-1, 1))[0, 0])
        # inverse-transform the +-z*std band back to the original sales scale
        # (note: StandardScaler is linear, so inverse-transforming mean+-band
        # separately is equivalent to scaling the band by scaler.scale_)
        upper_scaled = mean_scaled + self.z_multiplier * std_scaled
        lower_scaled = mean_scaled - self.z_multiplier * std_scaled
        upper = float(scaler.inverse_transform(upper_scaled.reshape(-1, 1))[0, 0])
        lower = float(scaler.inverse_transform(lower_scaled.reshape(-1, 1))[0, 0])
        lower, upper = min(lower, upper), max(lower, upper)

        return score_from_interval(
            point, lower, upper, self.alpha,
            model_type="lstm", method="mc_dropout",
            metadata={
                "n_passes": self.n_passes,
                "z_multiplier": self.z_multiplier,
                "attention_weights": attn[0].tolist() if attn is not None else None,
            },
        )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

class ConfidenceEstimationModule:
    """
    Single facade the Decision Gate (and FastAPI backend) can call without
    caring which of the three model types produced a forecast.
    """

    def __init__(self):
        self._xgb = None
        self._prophet = None
        self._lstm = None

    @property
    def xgboost(self) -> XGBoostConfidenceEstimator:
        if self._xgb is None:
            self._xgb = XGBoostConfidenceEstimator()
        return self._xgb

    @property
    def prophet(self) -> ProphetConfidenceEstimator:
        if self._prophet is None:
            self._prophet = ProphetConfidenceEstimator()
        return self._prophet

    @property
    def lstm(self) -> LSTMConfidenceEstimator:
        if self._lstm is None:
            self._lstm = LSTMConfidenceEstimator()
        return self._lstm


if __name__ == "__main__":
    # Quick smoke test across all three estimators using already-trained
    # Phase 1 models, on the most recent available row of each series.
    from data_loader import load_raw_sales
    from feature_engineering import build_feature_table

    print(f"CONFIDENCE_THRESHOLD = {get_threshold()}\n")

    raw = load_raw_sales()

    # --- XGBoost ---
    feat, _ = build_feature_table(raw)
    row = feat.iloc[-1]
    xgb_est = XGBoostConfidenceEstimator()
    result = xgb_est.estimate(row)
    print("XGBoost:", result.to_dict())
    print("  meets threshold:", result.meets_threshold(), "\n")

    # --- Prophet ---
    from prophet_model import train_prophet_for_series
    model, forecast, _ = train_prophet_for_series(raw, "SKU_A", "North")
    prophet_est = ProphetConfidenceEstimator()
    result = prophet_est.estimate(forecast.iloc[-1])
    print("Prophet:", result.to_dict())
    print("  meets threshold:", result.meets_threshold(), "\n")

    # --- LSTM ---
    from lstm_model import train_lstm_for_series, make_windows, WINDOW
    from sklearn.preprocessing import StandardScaler
    sub = raw[(raw["product_id"] == "SKU_A") & (raw["region"] == "North")].sort_values("date")
    values = sub["sales"].values.astype("float32")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    lstm_model, lstm_scaler, _ = train_lstm_for_series(raw, "SKU_A", "North", epochs=10)
    X, y = make_windows(scaled, WINDOW)
    last_window = X[-1:]  # most recent window, shape (1, WINDOW, 1)
    lstm_est = LSTMConfidenceEstimator()
    result = lstm_est.estimate(lstm_model, last_window, scaler)
    print("LSTM:", result.to_dict())
    print("  meets threshold:", result.meets_threshold())
