# Statistical rigour on forecasting metrics — method notes (Task 1)

Script: `scripts/s02_statistical_significance.py` (inputs cached by `scripts/s01_build_predictions.py`).
Outputs: `statistical_significance.csv`, `statistical_significance_block_sensitivity.csv`,
`statistical_seed_variance_xgboost.csv`, `lstm_seed_variance.csv` (from Task 4), `metric_reproduction_check.csv`.

## What is resampled
Retraining all three models hundreds of times is not cheap (Prophet × 9 series, LSTM × 9
series), so CIs come from a **moving-block bootstrap of the held-out residuals** of the
**saved** model artefacts — the same artefacts whose metrics are cited in the repo. Point
estimates reproduce `models/saved_models/*/metrics.json` to ≤ 1.5e-6
(`metric_reproduction_check.csv`).

* Test sets are each model's own original held-out split: XGBoost 1,233 rows
  (2025-03-18 → 2025-08-01), Prophet 1,278 rows (2025-03-13 → 2025-08-01), LSTM 1,251 rows
  (2025-03-16 → 2025-08-01) (`test_set_spans.csv`).
* Residuals of daily retail forecasts are autocorrelated (weekly cycle), so an i.i.d.
  bootstrap would understate uncertainty. Dates are resampled in **contiguous 7-day blocks**
  (Künsch 1989); all 9 series of a resampled date move together, preserving cross-series
  correlation. B = 2,000 resamples, `numpy.random.default_rng(42)`, percentile 95 % CIs.
* `statistical_significance_block_sensitivity.csv` repeats the aggregate CIs with block
  length 1 (i.i.d.), 7, 14, 28 days.
* Per-series CIs (`series = SKU_x__Region`) are the marginal of the same joint resample.

## Aggregation — an important correction
The README/`model_performance.csv` compares **XGBoost pooled RMSE (7.42)** with
**Prophet/LSTM mean-of-per-series RMSE (6.40 / 14.24)**. These are different statistics:
pooling squares errors across series of very different scale (SKU_A sells ~3× SKU_C), so
pooled RMSE > mean-of-series RMSE for every model. The CSV reports both for every model:
`ALL_pooled` and `ALL_mean_of_series` (MAPE is identical under both because every series
has the same number of test days). Like-for-like, mean-of-series RMSE is XGBoost 6.86,
Prophet 6.40, LSTM 14.27.

## Paired comparisons
Model differences are evaluated on the **137 dates common to all three test sets**
(2025-03-18 → 2025-08-01), with the *same* block resample applied to both models of a
pair (paired bootstrap). Rows `model = A_minus_B`, `metric = rmse_diff / mape_diff`,
`series = ALL_*_common_dates`. A CI excluding 0 indicates a difference at the 5 % level
(no multiple-comparison correction applied; three pairs × two metrics).

## Seed variation (supplementary)
* XGBoost point model retrained with `random_state` ∈ {0,1,2,3,4} (plus the original 42)
  on the identical split (`statistical_seed_variance_xgboost.csv`).
* LSTM: five independently initialised members trained with the unchanged production
  routine (Task 4, `lstm_seed_variance.csv`).
* Prophet's point forecast is a deterministic MAP fit; seeds only affect its sampled
  interval, so no seed study is reported for Prophet's RMSE/MAPE.

## Caveats
* The bootstrap captures sampling variability of the **test period** given fixed trained
  models; seed variation is reported separately and is not folded into the CIs.
* One test window (~4.5 months) per series; no rolling-origin backtest was run.
* The data are synthetic (`data/data_loader.py`, seed 42); the CIs say nothing about
  transfer to real POS data.
* `docs/report_data.json` cites numbers from a different training run than the saved
  artefacts (e.g. LSTM mean RMSE 14.237 vs 14.273; per-series differences up to 0.26);
  all numbers here are for the saved artefacts.
