# LSTM uncertainty: MC-Dropout vs deep ensemble (Task 4)

Script: `scripts/s04_lstm_uncertainty.py`. Outputs: `lstm_uncertainty_comparison.csv`,
`lstm_seed_variance.csv`.

## Setup
* Same held-out windows for every method: last 15 % of 21-day windows per series, 1,251
  forecasts total.
* **Current approach:** MC-Dropout on the *saved* LSTM models, 30 stochastic passes,
  80 % band = mean ± 1.28·std, `alpha = 0.5` (all from `threshold_config.yaml`).
* **Alternative:** deep ensemble of 5 LSTMs per series (45 models), trained with the
  **unchanged** production routine `train_lstm_for_series(epochs=25)`, seeds 0–4,
  TensorFlow op-determinism on. Uncertainty = std across members' deterministic forecasts.
* Also scored: MC-Dropout on each new member (separates the method from the particular
  weights) and ensemble + MC-Dropout (total variance = between + mean within).
* Evidential regression was not attempted.

## Result
| method | Pearson r(conf, APE) | Spearman ρ | AUROC (APE>10 %) | 80 % interval coverage | mean width | share conf ≥ 0.85 |
|---|---:|---:|---:|---:|---:|---:|
| MC-Dropout, saved models (current; repo documents −0.081) | **−0.105** | −0.135 | 0.549 | 0.233 | 6.83 | 1.000 |
| MC-Dropout, mean over 5 new members (sd of r = 0.015) | −0.091 | −0.133 | 0.553 | 0.228 | 6.66 | 1.000 |
| Deep ensemble (5) | **−0.033** | −0.021 | 0.531 | 0.062 | 1.49 | 1.000 |
| Deep ensemble + MC-Dropout | −0.094 | −0.140 | 0.556 | 0.251 | 7.12 | 1.000 |

## Plain conclusion
* **Neither method gives a strong confidence signal.** MC-Dropout is the better of the
  two (r = −0.105, AUROC 0.549 — barely above the 0.5 of a coin flip); the deep ensemble
  is *worse* (r = −0.033, AUROC 0.531). Combining them does not help. The repo's −0.081
  is reproduced in sign and magnitude (−0.091 to −0.105 here; the repo value came from
  re-trained 15-epoch models with 15 passes and a test-set-selected alpha).
* **The intervals are badly miscalibrated:** the nominal 80 % MC-Dropout band covers
  23.3 % of outcomes; the ensemble band covers 6.2 %. Members trained from different
  seeds converge to nearly the same function (pooled RMSE 15.44–15.57 across seeds), so
  between-member spread (mean width 1.49 units) is far smaller than the actual error
  (RMSE ≈ 15.5).
* **Practical consequence for the gate:** with `alpha = 0.5` every LSTM forecast scores
  ≥ 0.85, so on the LSTM path the deployed gate **never defers** (100 % autonomous at
  0.85, `gate_threshold_sweep.csv`), despite the LSTM having the largest errors of the
  three models (MAPE 7.99 %; 25.6 % of its forecasts exceed 10 % APE).
* Seed study (Task 1 supplement): mean-of-series RMSE across 5 seeds 14.18–14.28
  (saved model 14.27); the 5-member ensemble mean forecast gives 14.20.
* A likely remedy — not run here — is conformal recalibration of the LSTM interval
  (as in Task 3) or a larger `alpha`; both would need validation on held-out data.
* MC-Dropout is itself stochastic: before seeding was fixed (seed set *before* model
  load, TF op determinism on), three runs of the identical 30-pass procedure gave
  r = −0.1056, −0.1090 and −0.1048, i.e. Monte-Carlo noise of about ±0.004 in r. The
  reported −0.105 is the now-deterministic seeded value.
