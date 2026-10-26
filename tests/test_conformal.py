"""Unit tests for confidence/conformal.py (finite-sample coverage on
exchangeable synthetic data, and the width behaviour each method should have)."""

from __future__ import annotations

import numpy as np
import pytest

from conformal import conformal_quantile, cqr, normalized_conformal, split_conformal


def _hetero(n, rng):
    x = rng.integers(0, 2, n)            # "promo" flag
    sigma = np.where(x == 1, 3.0, 1.0)
    y = 10 + rng.normal(0, sigma)
    return x, sigma, y


def test_conformal_quantile_level():
    # n=9, coverage .8 -> ceil(10*.8)/9 = 8/9 -> 8th smallest of 1..9
    assert conformal_quantile(np.arange(1, 10), 0.8) == 8
    assert conformal_quantile(np.arange(1, 5), 0.99) == 4  # capped at max


@pytest.mark.parametrize("method", ["split", "normalized", "cqr"])
def test_marginal_coverage_on_exchangeable_data(method):
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(40):
        xc, sc, yc = _hetero(500, rng)
        xt, st, yt = _hetero(2000, rng)
        yhat_c, yhat_t = np.full(500, 10.0), np.full(2000, 10.0)
        if method == "split":
            lo, hi, _ = split_conformal(yhat_t, yc, yhat_c, 0.8)
        elif method == "normalized":
            lo, hi, _ = normalized_conformal(yhat_t, st, yc, yhat_c, sc, 0.8)
        else:
            lo, hi, _ = cqr(yhat_t, -1.28 * st, 1.28 * st, yc, yhat_c, -1.28 * sc, 1.28 * sc, 0.8)
        covs.append(np.mean((yt >= lo) & (yt <= hi)))
    assert np.mean(covs) >= 0.79


def test_split_conformal_width_is_constant_but_normalized_adapts():
    rng = np.random.default_rng(1)
    xc, sc, yc = _hetero(1000, rng)
    xt, st, _ = _hetero(1000, rng)
    lo, hi, _ = split_conformal(np.full(1000, 10.0), yc, np.full(1000, 10.0), 0.8)
    assert np.ptp(hi - lo) == pytest.approx(0.0)
    lo, hi, _ = normalized_conformal(np.full(1000, 10.0), st, yc, np.full(1000, 10.0), sc, 0.8)
    w = hi - lo
    assert w[xt == 1].mean() == pytest.approx(3 * w[xt == 0].mean())


def test_empty_calibration_raises():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.8)
