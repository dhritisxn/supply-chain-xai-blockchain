"""
conformal.py

Post-hoc conformal recalibration of prediction intervals (additive module --
nothing in the live pipeline imports this yet; see
publication_package/scripts/s03_prophet_conformal.py for how it is evaluated).

Motivation: Prophet's native yhat_lower/yhat_upper interval is driven by one
global observation-noise parameter, so its width barely changes between
easy and hard days (see the CAVEAT in threshold_config.yaml). The
confidence score exp(-alpha * width / |forecast|) therefore cannot rank
forecasts by difficulty. Three standard recalibrations are provided:

  split_conformal        Vovk et al. / Lei et al. 2018. Score = |y - yhat|.
                         Gives finite-sample marginal coverage, but the
                         interval width is a constant -- it CANNOT fix a
                         homoscedastic interval by itself.
  normalized_conformal   Locally-weighted split conformal (Papadopoulos 2008;
                         Lei et al. 2018 sec. 5.2). Score = |y - yhat| / s(x)
                         for a difficulty model s(x) fit on separate data.
  cqr                    Conformalized quantile regression (Romano, Patterson
                         & Candes 2019) applied to the residual y - yhat.

All three use the finite-sample corrected quantile level
ceil((n + 1) * coverage) / n on a calibration set disjoint from both the
data used to fit the forecaster's residual models and the test set.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """The k-th smallest calibration score, k = ceil((n+1)*coverage) (capped
    at n) -- i.e. the ceil((n+1)*coverage)/n empirical quantile."""
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration set")
    k = min(int(np.ceil((n + 1) * coverage)), n)
    return float(scores[k - 1])


def split_conformal(yhat_test, y_cal, yhat_cal, coverage: float = 0.8):
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)), coverage)
    yhat_test = np.asarray(yhat_test, float)
    return yhat_test - q, yhat_test + q, q


def normalized_conformal(yhat_test, sigma_test, y_cal, yhat_cal, sigma_cal,
                         coverage: float = 0.8, eps: float = 1e-6):
    sigma_cal = np.maximum(np.asarray(sigma_cal, float), eps)
    sigma_test = np.maximum(np.asarray(sigma_test, float), eps)
    scores = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)) / sigma_cal
    q = conformal_quantile(scores, coverage)
    yhat_test = np.asarray(yhat_test, float)
    return yhat_test - q * sigma_test, yhat_test + q * sigma_test, q


def cqr(yhat_test, lo_test, hi_test, y_cal, yhat_cal, lo_cal, hi_cal,
        coverage: float = 0.8):
    """lo_*/hi_* are predicted lower/upper quantiles of the residual y - yhat
    (in the same units as y). Returns the conformalized interval for y."""
    r_cal = np.asarray(y_cal, float) - np.asarray(yhat_cal, float)
    scores = np.maximum(np.asarray(lo_cal) - r_cal, r_cal - np.asarray(hi_cal))
    q = conformal_quantile(scores, coverage)
    yhat_test = np.asarray(yhat_test, float)
    lower = yhat_test + np.asarray(lo_test) - q
    upper = yhat_test + np.asarray(hi_test) + q
    return np.minimum(lower, upper), np.maximum(lower, upper), q
