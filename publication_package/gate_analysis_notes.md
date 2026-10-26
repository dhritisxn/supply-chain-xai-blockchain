# Decision-gate threshold sweep & baselines (Task 2)

Script: `scripts/s05_gate_analysis.py`. Outputs: `gate_threshold_sweep.csv`,
`gate_baseline_comparison.csv`, `gate_roc_curve.png/.svg`,
`gate_original_scan_realized_error.csv`.

**Setup.** Every held-out test forecast (one per series-day: XGBoost n = 1,233,
Prophet 1,278, LSTM 1,251) is treated as a reorder-required decision (the repo's
`below_rop` scenario) and routed through the real `gate.route()`. A decision "should have
been deferred" iff its realised APE > 10 % (5 % and 15 % also in the sweep). Positive class
= deferral; precision/recall/F1 are for deferring bad forecasts; "bad-execution rate" = share
of autonomously executed decisions whose forecast was bad. Difference CIs: 2,000-resample
7-day block bootstrap over dates.

## Summary (3–5 sentences)
For XGBoost, the deployed confidence gate (threshold 0.85, 71.9 % autonomous) executes
bad forecasts at a rate of **3.84 %**, vs **7.62 %** for both naive rules (always
autonomous ≡ autonomous-iff-reorder-required in this population) and for random gates at
77.8 % or at the matched 71.9 % rate — a reduction of 3.79 percentage points (95 % CI
[−4.76, −2.86]); at the repo's 77.8 % autonomy rate the gate reaches 4.17 % (−3.45 pp,
CI [−4.44, −2.54]). **This advantage is not unique to the ML signal:** a one-line rule
"defer iff promo or holiday" achieves 4.49 % at 92.2 % autonomy — higher precision
(0.448 vs 0.173) with far fewer deferrals — because those flags drive the synthetic
generator's noise; a trailing-volatility rule at the same deferral count reaches 5.87 %.
The gate is **worse than random** with Prophet's native confidence (6.80 % vs 6.18 %,
+0.62 pp, CI [+0.31, +1.10]) and **inert** with either LSTM signal (every LSTM confidence
≥ 0.85, so 100 % autonomous, 25.6 % / 27.2 % bad executions). With normalized-conformal
Prophet confidence (Task 3) the gate drops Prophet's bad-execution rate to 3.13 %
(−3.05 pp, CI [−4.07, −2.26]), equal to the promo/holiday rule.

## Key rows (APE tolerance 10 %, threshold 0.85 unless stated)
| confidence source | gate | autonomous | bad-exec rate | mean APE executed | AUROC |
|---|---|---:|---:|---:|---:|
| XGBoost q10/q90 | confidence @0.85 | 0.719 | 0.0384 | 0.0384 | 0.743 |
| XGBoost | confidence @ rate 0.778 (thr 0.8441) | 0.778 | 0.0417 | 0.0384 | |
| XGBoost | always autonomous | 1.000 | 0.0762 | 0.0433 | |
| XGBoost | random @0.778 (10k draws) | 0.778 | 0.0762 | 0.0433 | 0.5 |
| XGBoost | defer iff promo/holiday | 0.922 | 0.0449 | 0.0390 | |
| Prophet native | confidence @0.85 | 0.829 | 0.0680 | 0.0413 | 0.284 |
| Prophet normalized conformal | confidence @0.85 | 0.924 | 0.0313 | 0.0367 | 0.799 |
| LSTM MC-Dropout | confidence @0.85 | 1.000 | 0.2558 | 0.0782 | 0.549 |
| LSTM deep ensemble | confidence @0.85 | 1.000 | 0.2718 | 0.0794 | 0.531 |

**Threshold choice.** For XGBoost the F1 of deferral peaks at 0.80 (F1 0.439; 93.6 %
autonomous, bad-exec rate 4.85 %), not at 0.85 (F1 0.272); 0.85 trades precision for
recall (0.638). Above 0.90 almost nothing is autonomous (3.8 % at 0.90, 0 % at ≥ 0.95).

**The original 9-decision scan** (`gate_original_scan_realized_error.csv`): with the saved
artefacts the same 7 AUTO / 2 HUMAN split is reproduced, but the recomputed confidences
differ slightly from `docs/gate_accuracy_scan.csv` (e.g. 0.9055 vs 0.9137 for SKU_A/North),
indicating that CSV came from a different training run. Checked against the actual sales of
2025-08-01, the two deferred decisions had realised APE of 0.5 % and 1.3 % (both accurate),
while one autonomous decision (SKU_A/South) had 12.6 %. With n = 9 this is anecdotal, but
it means "22.2 % correctly deferred" is not supported by that scan's outcomes; the
test-set analysis above is the evidence that should be cited.
