"""Unit tests for confidence/confidence_estimator.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import confidence_estimator as ce


# ---------------------------------------------------------------- the formula

@pytest.mark.parametrize("point,lower,upper,alpha,expected", [
    # width 20, |point| 100 -> rel 0.2 -> exp(-0.2)
    (100.0, 90.0, 110.0, 1.0, math.exp(-0.2)),
    # same interval, stricter alpha
    (100.0, 90.0, 110.0, 2.5, math.exp(-0.5)),
    # LSTM default alpha 0.5, rel width 0.4
    (50.0, 40.0, 60.0, 0.5, math.exp(-0.2)),
    # negative forecast uses |point|
    (-100.0, -110.0, -90.0, 1.0, math.exp(-0.2)),
])
def test_exp_decay_formula_hand_computed(point, lower, upper, alpha, expected):
    r = ce.score_from_interval(point, lower, upper, alpha, "m", "x")
    assert r.interval_width == pytest.approx(upper - lower)
    assert r.relative_width == pytest.approx((upper - lower) / abs(point))
    assert r.confidence == pytest.approx(expected, rel=1e-12)


def test_zero_width_gives_full_confidence():
    assert ce.score_from_interval(100, 100, 100, 1.0, "m", "x").confidence == 1.0


def test_confidence_monotonically_decreases_with_width():
    widths = np.linspace(0, 200, 50)
    confs = [ce.score_from_interval(100, 100 - w / 2, 100 + w / 2, 1.0, "m", "x").confidence for w in widths]
    assert all(a > b for a, b in zip(confs, confs[1:]))


def test_forecast_at_zero_uses_eps_floor_and_collapses_confidence():
    r = ce.score_from_interval(0.0, -1.0, 1.0, 1.0, "m", "x")
    assert r.relative_width == pytest.approx(2.0 / 1e-6)
    assert r.confidence == 0.0


def test_forecast_near_zero_is_numerically_safe():
    for p in [1e-12, 1e-9, -1e-9, 1e-6]:
        r = ce.score_from_interval(p, p - 0.5, p + 0.5, 1.0, "m", "x")
        assert 0.0 <= r.confidence <= 1.0 and np.isfinite(r.relative_width)


def test_tiny_forecast_with_tiny_interval_is_still_scored_relatively():
    # width/|point| = 0.1 regardless of absolute scale
    r = ce.score_from_interval(0.01, 0.0095, 0.0105, 1.0, "m", "x")
    assert r.confidence == pytest.approx(math.exp(-0.1))


def test_crossed_bounds_are_clamped_to_zero_width():
    """Documented edge case: score_from_interval clamps a negative width to
    0, i.e. crossed bounds score as *maximum* confidence. The per-model
    estimators sort bounds first (see the XGBoost test below), so this only
    matters for direct callers."""
    r = ce.score_from_interval(100, 110, 90, 1.0, "m", "x")
    assert r.interval_width == 0.0 and r.confidence == 1.0


def test_confidence_always_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = rng.normal(0, 100)
        lo, hi = sorted(rng.normal(p, 50, 2))
        c = ce.score_from_interval(p, lo, hi, rng.uniform(0, 5), "m", "x").confidence
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------- threshold / result object

def test_threshold_comes_from_yaml():
    assert ce.get_threshold() == 0.85


def test_meets_threshold_boundary_is_inclusive():
    r = ce.ConfidenceResult("m", 1, 1, 1, 0, 0, 0.85, "x")
    assert r.meets_threshold(0.85) is True
    assert ce.ConfidenceResult("m", 1, 1, 1, 0, 0, 0.8499, "x").meets_threshold(0.85) is False
    assert r.meets_threshold() is True  # default threshold from config


def test_to_dict_rounds_to_four_places():
    r = ce.score_from_interval(100.123456, 90.0, 110.0, 1.0, "m", "x")
    d = r.to_dict()
    assert d["point_forecast"] == 100.1235 and d["confidence"] == round(r.confidence, 4)


# ---------------------------------------------------------------- per-model estimators

class _Const:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.array([self.v] * len(X))


def _feature_row():
    from feature_engineering import FEATURE_COLS
    return pd.Series({c: 1.0 for c in FEATURE_COLS})


def test_xgboost_estimator_uses_q10_q90_and_sorts_crossed_quantiles():
    est = ce.XGBoostConfidenceEstimator(_Const(100.0), _Const(115.0), _Const(85.0), alpha=1.0)
    r = est.estimate(_feature_row())
    assert (r.lower_bound, r.upper_bound) == (85.0, 115.0)
    assert r.confidence == pytest.approx(math.exp(-0.3))
    assert r.model_type == "xgboost" and r.method == "quantile_regression_q10_q90"


def test_prophet_estimator_reads_native_interval():
    r = ce.ProphetConfidenceEstimator(alpha=1.0).estimate(
        {"yhat": 200.0, "yhat_lower": 180.0, "yhat_upper": 220.0})
    assert r.confidence == pytest.approx(math.exp(-0.2)) and r.model_type == "prophet"


def test_default_alphas_match_yaml():
    cfg = ce._load_config()
    assert ce.ProphetConfidenceEstimator().alpha == cfg["prophet"]["alpha"]
    assert ce.LSTMConfidenceEstimator().alpha == cfg["lstm"]["alpha"]


def test_lstm_estimator_inverse_scales_mc_band(monkeypatch):
    from sklearn.preprocessing import StandardScaler
    import lstm_model

    scaler = StandardScaler().fit(np.array([[0.0], [20.0]]))  # mean 10, scale 10
    monkeypatch.setattr(lstm_model, "predict_with_uncertainty",
                        lambda model, X, n_passes: (np.array([1.0]), np.array([0.5]), None))
    est = ce.LSTMConfidenceEstimator(alpha=0.5, n_passes=3, z_multiplier=1.28)
    r = est.estimate(None, np.zeros((1, 21, 1)), scaler)
    # point = 1*10+10 = 20; band = +-1.28*0.5*10 = +-6.4 -> width 12.8
    assert r.point_forecast == pytest.approx(20.0)
    assert r.interval_width == pytest.approx(12.8)
    assert r.confidence == pytest.approx(math.exp(-0.5 * 12.8 / 20.0))
