# Prophet confidence recalibration with conformal prediction (Task 3)

Script: `scripts/s03_prophet_conformal.py`; module: `confidence/conformal.py` (new, additive,
unit-tested in `tests/test_conformal.py`; not wired into the live gate).
Outputs: `prophet_calibration_before_after.csv`, `prophet_reliability_by_confidence_bucket.csv`.

## Why Prophet's score points the wrong way
Prophet's 80 % interval width is almost constant within a series (one global
`sigma_obs`), so `relative_width = width / |yhat|` is *smallest on high-`yhat` days* —
which in this data are promo/holiday days, the days with the *largest* errors. Confidence
is therefore highest exactly when the forecast is worst. The reliability table shows it
directly: the top native-confidence quintile has mean APE 5.64 % vs 3.52–3.73 % for the
other four.

## Protocol (no test-period information used)
| window | dates (per series) | used for |
|---|---|---|
| fit | ≤ 70 % date quantile | calibration-stage Prophet (exact production config via `train_prophet_for_series(test_frac=0.30)`); its in-sample residuals train the difficulty models |
| calibration | (70 %, 85 %] | out-of-sample residuals at the same 1–~145-day horizon the production model faces; conformal quantile |
| test | (85 %, 100 %] | **saved production Prophet** forecasts; only window scored |

Difficulty features are flags known at forecast time: `is_promo`, `is_holiday`,
`is_weekend`; residuals are modelled relative to `|yhat|` so one pooled model serves all 9
series. Target coverage 80 % (Prophet's configured `interval_width`); confidence uses the
configured `alpha = 1.0`.

Methods: (1) **split conformal** on |residual| (per series) — constant width; (2)
**normalized split conformal** with s(x) = linear model of |r|/|yhat|; (3) **CQR**
(Romano et al. 2019) with linear quantile regression (q10/q90) of r/|yhat|.

## Result (test set, n = 1,278)
| method | Pearson r(conf, APE) | Spearman ρ | AUROC (low conf → APE>10 %) | coverage (target 0.80) | mean width | series with r < 0 |
|---|---:|---:|---:|---:|---:|---:|
| native interval (before; repo documents +0.250) | **+0.253** | +0.094 | 0.284 | 0.862 | 16.70 | 1 / 9 |
| split conformal | +0.220 | +0.105 | 0.260 | 0.820 | 15.33 | 1 / 9 |
| normalized conformal | **−0.400** | −0.174 | **0.799** | 0.775 | 14.46 | 9 / 9 |
| CQR | −0.392 | −0.123 | 0.765 | 0.773 | 14.80 | 9 / 9 |

(All p < 0.001; exact values and per-series min/max in the CSV.)

## Honest reading
* **Direction fixed, but only by the methods that condition on context.** Plain split
  conformal does *not* fix it (+0.220): it rescales a constant width and cannot create
  heteroscedasticity. That negative result is itself worth reporting.
* The fix works because the difficulty model learns that promo days have ~2.3× the
  relative residual of normal days (fitted coefficient `is_promo = +0.0525` on an
  intercept of 0.0407). **This is exactly the heteroscedasticity the synthetic generator
  injects** (`noise_std × (1 + 1.8·is_promo + 1.2·is_holiday)`), so the improvement is
  partly by construction; on real data the gain depends on how predictable the error
  regime is.
* Rank correlation stays modest (Spearman −0.17). Normalized-conformal confidence takes
  essentially one value per flag combination (the reliability file has only 2 distinct
  quintile bins), so it is a good *detector of hard days* (AUROC 0.80) rather than a
  fine-grained ranking of every forecast.
* **Coverage is 77.5 %, below the 80 % target.** Split conformal's guarantee assumes
  exchangeable calibration/test data; the calibration window (Oct–Mar, incl. festive
  season) and test window (Mar–Aug) differ, and the production model is trained on more
  data than the calibration-stage model. The native interval over-covers (86.2 %).
* At the deployed 0.85 threshold, the share of Prophet decisions cleared for autonomy
  rises from 82.9 % to 92.4 %, and the cleared decisions have lower mean APE (3.67 % vs
  4.13 % under the native interval) — see `gate_baseline_comparison.csv`.
* Neither the live pipeline nor `threshold_config.yaml` was changed; adopting this would
  be a follow-up change.
