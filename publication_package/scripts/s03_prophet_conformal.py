"""
s03_prophet_conformal.py  (Task 3)

Post-hoc conformal recalibration of Prophet's prediction intervals and its
effect on the confidence-vs-error relationship.

Protocol (per series, no information from the test period is used):
  * fit window   : dates <= 70% quantile -> a *calibration-stage* Prophet is
                   trained here with the exact production configuration
                   (models/prophet_model.train_prophet_for_series with
                   test_frac=0.30). Its in-sample residuals train the
                   residual-difficulty models s(x) (normalized conformal) and
                   q10/q90(x) (CQR).
  * calib window : (70%, 85%] -> out-of-sample residuals of that Prophet at
                   the same 1..~145-day horizon the production model faces on
                   the test window. Conformal quantiles come from here.
  * test window  : (85%, 100%] -> the SAVED production Prophet's yhat (the
                   same forecasts evaluated everywhere else), with intervals
                   rebuilt by each method. Only this window is scored.

Difficulty features are calendar/context flags known at forecast time:
is_promo, is_holiday, is_weekend. Residuals are modelled relative to |yhat|
so one pooled model serves all 9 series despite their different scales.

Outputs
  publication_package/prophet_calibration_before_after.csv
  publication_package/prophet_reliability_by_confidence_bucket.csv
  publication_package/intermediate/test_predictions_prophet_conformal.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

from common import (PKG, INTER, ROOT, SEED, SERIES, cached, calibration_metrics,
                    config, confidence_vec, load_raw, ape)

sys.path.insert(0, str(ROOT / "confidence"))
from conformal import split_conformal, normalized_conformal, cqr  # noqa: E402

COVERAGE = 0.80  # Prophet's configured interval_width
FEATS = ["is_promo", "is_holiday", "is_weekend"]


def calib_stage(raw):
    """Fit the calibration-stage Prophet per series; return in-sample (fit) and
    calibration-window residual frames."""
    from prophet_model import train_prophet_for_series, _series_to_prophet_frame
    fit_rows, cal_rows = [], []
    for product, region in SERIES:
        df = _series_to_prophet_frame(raw, product, region)
        c70, c85 = df["ds"].quantile(0.70), df["ds"].quantile(0.85)
        np.random.seed(SEED)
        model, forecast, _ = train_prophet_for_series(raw, product, region, test_frac=0.30)
        fc = forecast.set_index("ds")
        cal = df[(df.ds > c70) & (df.ds <= c85)].copy()
        cal["yhat"] = fc.loc[cal.ds, "yhat"].values
        fit = df[df.ds <= c70].copy()
        np.random.seed(SEED)
        fit["yhat"] = model.predict(fit[["ds", "is_promo"]])["yhat"].values
        for frame, bucket in [(fit, fit_rows), (cal, cal_rows)]:
            frame["product_id"], frame["region"] = product, region
            bucket.append(frame)
    return pd.concat(fit_rows, ignore_index=True), pd.concat(cal_rows, ignore_index=True)


def add_flags(df, raw, date_col):
    flags = raw[["product_id", "region", "date", "is_holiday"]].rename(columns={"date": date_col})
    out = df.merge(flags, on=["product_id", "region", date_col], how="left", suffixes=("", "_r"))
    if "is_holiday_r" in out:
        out["is_holiday"] = out["is_holiday_r"]
        out = out.drop(columns=["is_holiday_r"])
    out["is_weekend"] = (pd.to_datetime(out[date_col]).dt.dayofweek >= 5).astype(int)
    return out


def main():
    raw = load_raw()
    alpha = config()["prophet"]["alpha"]
    test = cached("test_predictions_prophet", None).copy()
    test["is_weekend"] = (test.date.dt.dayofweek >= 5).astype(int)

    fit, cal = calib_stage(raw)
    fit = add_flags(fit, raw, "ds")
    cal = add_flags(cal, raw, "ds")
    for d in (fit, cal):
        d["rel_resid"] = (d.y - d.yhat) / np.abs(d.yhat)

    # --- difficulty model for normalized conformal: E|r|/|yhat| ~ flags ------
    lin = LinearRegression().fit(fit[FEATS], np.abs(fit.rel_resid))
    floor = 1e-3
    s_cal = np.maximum(lin.predict(cal[FEATS]), floor) * np.abs(cal.yhat)
    s_test = np.maximum(lin.predict(test[FEATS]), floor) * np.abs(test.point)

    # --- quantile models for CQR: q10/q90 of r/|yhat| ~ flags ---------------
    qlo = QuantileRegressor(quantile=0.10, alpha=0.0, solver="highs").fit(fit[FEATS], fit.rel_resid)
    qhi = QuantileRegressor(quantile=0.90, alpha=0.0, solver="highs").fit(fit[FEATS], fit.rel_resid)

    results = []
    variants = {}

    # BEFORE: native Prophet interval
    variants["native_prophet_interval (before)"] = (test.lower.values, test.upper.values)

    # Standard split conformal, per series (constant width within a series)
    lo = np.empty(len(test)); hi = np.empty(len(test))
    for product, region in SERIES:
        mt = (test.product_id == product) & (test.region == region)
        mc = (cal.product_id == product) & (cal.region == region)
        lo[mt.values], hi[mt.values], _ = split_conformal(test.point[mt], cal.y[mc], cal.yhat[mc], COVERAGE)
    variants["split_conformal_abs_residual"] = (lo, hi)

    # Normalized (locally weighted) split conformal, pooled calibration
    lo, hi, q_norm = normalized_conformal(test.point, s_test, cal.y, cal.yhat, s_cal, COVERAGE)
    variants["normalized_conformal_flags"] = (lo, hi)

    # CQR on residuals, pooled calibration
    lo, hi, q_cqr = cqr(test.point, qlo.predict(test[FEATS]) * np.abs(test.point),
                        qhi.predict(test[FEATS]) * np.abs(test.point),
                        cal.y, cal.yhat,
                        qlo.predict(cal[FEATS]) * np.abs(cal.yhat),
                        qhi.predict(cal[FEATS]) * np.abs(cal.yhat), COVERAGE)
    variants["cqr_flags"] = (lo, hi)

    bucket_rows = []
    for name, (lo, hi) in variants.items():
        m = calibration_metrics(test.point.values, lo, hi, test.actual.values, alpha, name)
        conf = np.round(confidence_vec(test.point.values, lo, hi, alpha), 10)
        e = ape(test.actual, test.point)
        m["mean_ape_conf_ge_0.85"] = float(e[conf >= 0.85].mean()) if (conf >= 0.85).any() else np.nan
        m["mean_ape_conf_lt_0.85"] = float(e[conf < 0.85].mean()) if (conf < 0.85).any() else np.nan
        promo = test.is_promo.values == 1
        m["mean_width_promo_days"] = float(np.mean((hi - lo)[promo]))
        m["mean_width_nonpromo_days"] = float(np.mean((hi - lo)[~promo]))
        # per-series Pearson (is the direction consistent across series?)
        per = []
        for product, region in SERIES:
            mt = ((test.product_id == product) & (test.region == region)).values
            per.append(np.corrcoef(conf[mt], e[mt])[0, 1])
        m["per_series_pearson_min"] = float(np.min(per))
        m["per_series_pearson_max"] = float(np.max(per))
        m["n_series_negative_pearson"] = int(np.sum(np.array(per) < 0))
        results.append(m)
        qs = pd.qcut(conf, 5, labels=False, duplicates="drop")
        for b in sorted(np.unique(qs)):
            sel = qs == b
            bucket_rows.append(dict(method=name, confidence_quintile=int(b) + 1,
                                    conf_min=conf[sel].min(), conf_max=conf[sel].max(),
                                    n=int(sel.sum()), mean_ape=float(e[sel].mean())))
        key = name.split(" ")[0]
        test[f"lower_{key}"], test[f"upper_{key}"] = lo, hi

    out = pd.DataFrame(results)
    out.insert(1, "documented_value_in_repo",
               ["+0.250 (threshold_config.yaml, calibrate.py)"] + [""] * (len(out) - 1))
    out["conformal_quantile"] = [np.nan, np.nan, q_norm, q_cqr]
    out["difficulty_model_coefs"] = ["", "",
                                     f"intercept={lin.intercept_:.4f}; " + "; ".join(f"{f}={c:.4f}" for f, c in zip(FEATS, lin.coef_)),
                                     "q10: " + "; ".join(f"{f}={c:.4f}" for f, c in zip(FEATS, qlo.coef_)) + f"; b0={qlo.intercept_:.4f} | q90: " + "; ".join(f"{f}={c:.4f}" for f, c in zip(FEATS, qhi.coef_)) + f"; b0={qhi.intercept_:.4f}"]
    out["n_fit_rows"] = len(fit)
    out["n_calibration_rows"] = len(cal)
    out.to_csv(PKG / "prophet_calibration_before_after.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(PKG / "prophet_reliability_by_confidence_bucket.csv", index=False)
    test.to_csv(INTER / "test_predictions_prophet_conformal.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.T.to_string())
    print(pd.DataFrame(bucket_rows).to_string(index=False))


if __name__ == "__main__":
    main()
