# RESULTS_SUMMARY — publication package

Every value below is copied from a file in `publication_package/` produced by
`run_all_experiments.py` (fixed seeds). Column **source** names the file. "Before" = value
already cited in the repo (`docs/report_data.json`, `README.md`,
`docs/novelty_and_contribution.md`, `confidence/threshold_config.yaml`); "after" = value
produced here. Data: synthetic (`data/raw/synthetic_sales.csv`, 9 series × 944 days,
generator seed 42). Model artefacts are the saved ones in `models/saved_models/`, unchanged.

Conventions: CI = 95 % percentile interval, 2,000-resample 7-day moving-block bootstrap
over test dates. APE = |y − ŷ|/y. "Bad" forecast = APE > 10 %.

---

## 0. Provenance checks (read first)

| key | measures | before | after | demonstrates | source |
|---|---|---|---|---|---|
| `repro.saved_vs_metrics_json.max_abs_diff` | recomputed test metrics of saved artefacts vs `models/saved_models/*/metrics.json` | — | ≤ 1.5e-6 (all 21 rows) | the saved models are exactly the ones their metric files describe | `metric_reproduction_check.csv` |
| `repro.saved_vs_report_data_json` | same vs numbers quoted in `docs/report_data.json` | LSTM mean RMSE 14.237, MAPE 7.93 % | 14.273, 7.99 % (per-series Δ up to 0.258 RMSE); Prophet 6.402→6.4015; XGBoost 7.4210→7.4230 | `report_data.json` came from a different training run than the saved artefacts; all numbers here are for the saved artefacts | `metric_reproduction_check.csv` |
| `repro.gate_scan_confidence` | XGBoost confidences of the 9-decision scan, recomputed | 0.9137 … 0.8309 (`docs/gate_accuracy_scan.csv`) | 0.9055 … 0.8263; **same 7 AUTO / 2 HUMAN split** | headline 77.8 % split reproduces; individual confidences cited in the docs do not | `gate_original_scan_realized_error.csv` |
| `repro.solc_v1_bytecode` | committed v1 bytecode vs recompilation | — | identical executable bytecode (solc 0.8.19, optimizer 200, legacy codegen) | contract source in repo is what was deployed | `blockchain/scripts/compile.py`, `intermediate/log_compile.txt` |
| `repro.clean_env` | regenerate all outputs in a fresh venv from `requirements.txt` | — | **byte-identical** (22/22 CSVs, PNG, SVG); see §9 | the published numbers reproduce from a clean install | `reproducibility_check.md`, `intermediate/clean_env_compare.txt` |

---

## 1. Forecast accuracy with uncertainty (Task 1)

Test sets: XGBoost 1,233 rows (2025-03-18→08-01), Prophet 1,278 (03-13→08-01), LSTM 1,251
(03-16→08-01).

| key | measures | before | after (point [95 % CI]) | demonstrates | source |
|---|---|---|---|---|---|
| `xgboost.rmse.pooled` | RMSE over all test rows | 7.42 | 7.423 [6.939, 7.918] | | `statistical_significance.csv` |
| `xgboost.rmse.mean_of_series` | mean of 9 per-series RMSEs (the statistic used for Prophet/LSTM) | not reported | 6.861 [6.496, 7.187] | README compares XGBoost pooled (7.42) to Prophet mean-of-series (6.40); like-for-like XGBoost is 6.86 | `statistical_significance.csv` |
| `xgboost.mape` | MAPE | 4.34 % | 4.334 % [4.152, 4.548] | | `statistical_significance.csv` |
| `xgboost.r2.pooled` | R² | 0.977 | 0.9768 [0.9743, 0.9794] | | `statistical_significance.csv` |
| `prophet.rmse.mean_of_series` | | 6.40 | 6.402 [5.942, 6.818] | | `statistical_significance.csv` |
| `prophet.rmse.pooled` | | not reported | 6.894 [6.317, 7.501] | | `statistical_significance.csv` |
| `prophet.mape` | | 4.06 % | 4.057 % [3.869, 4.268] | | `statistical_significance.csv` |
| `lstm.rmse.mean_of_series` | | 14.24 | 14.273 [12.941, 15.588] | | `statistical_significance.csv` |
| `lstm.rmse.pooled` | | not reported | 15.558 [13.958, 17.388] | | `statistical_significance.csv` |
| `lstm.mape` | | 7.93 % | 7.991 % [7.572, 8.484] | | `statistical_significance.csv` |
| `diff.xgboost_minus_prophet.rmse` | paired, 137 common dates, mean-of-series | — | +0.468 [+0.230, +0.725] | Prophet is significantly more accurate than XGBoost (pooled: +0.531 [+0.197, +0.929]) | `statistical_significance.csv` |
| `diff.xgboost_minus_prophet.mape` | paired | — | +0.276 pp [+0.117, +0.441] | same, in MAPE | `statistical_significance.csv` |
| `diff.xgboost_minus_lstm.rmse` | paired, mean-of-series | — | −7.497 [−8.546, −6.282] | LSTM significantly worst | `statistical_significance.csv` |
| `diff.prophet_minus_lstm.rmse` | paired, mean-of-series | — | −7.965 [−9.015, −6.761] | | `statistical_significance.csv` |
| `xgboost.seed_variance.rmse` | point model retrained, seeds 0–4 | single run | mean 7.445, SD 0.034 (range 7.408–7.474) | seed effect ≪ bootstrap CI width (0.98) | `statistical_seed_variance_xgboost.csv` |
| `lstm.seed_variance.rmse_mean_of_series` | 5 retrained seeds | single run | 14.182–14.283 (saved 14.273) | | `lstm_seed_variance.csv` |
| `per_series` | 9 series × 3 models × {RMSE, MAPE} with CIs | point only | all in CSV | | `statistical_significance.csv` |
| `block_length_sensitivity` | aggregate CIs at block length 1/7/14/28 days | — | in CSV | | `statistical_significance_block_sensitivity.csv` |

---

## 2. Decision gate: threshold sweep and baselines (Task 2)

Population: every test forecast as a reorder-required decision (`below_rop`), routed by
the real `gate.route()`.

| key | measures | before | after | demonstrates | source |
|---|---|---|---|---|---|
| `gate.scan.auto_rate` | repo headline | 77.8 % AUTO / 22.2 % HUMAN on **n = 9** | reproduced (7/2) | headline was a single-threshold, n = 9 number | `gate_original_scan_realized_error.csv` |
| `gate.scan.deferred_realized_ape` | actual APE of the 2 "correctly deferred" decisions | "correctly deferred" | 0.52 % and 1.28 %; one AUTO decision had 12.58 % | the "correctly deferred" claim is not supported by that scan's outcomes (anecdotal, n = 9) | `gate_original_scan_realized_error.csv` |
| `gate.xgb.auto_rate@0.85` | share autonomous on test set | 77.8 % (n = 9) | 71.86 % (n = 1,233) | | `gate_baseline_comparison.csv` |
| `gate.xgb.bad_exec_rate@0.85` | share of executed decisions with APE > 10 % | not measured | **3.84 %** | | `gate_baseline_comparison.csv` |
| `gate.baseline.always_autonomous` | naive rule (≡ autonomous iff reorder_required here) | — | 7.62 % | gate halves bad executions vs naive rule | `gate_baseline_comparison.csv` |
| `gate.baseline.random@77.8%` | random gate, 10,000 draws | — | 7.62 % (95 % range in file) | | `gate_baseline_comparison.csv` |
| `gate.xgb.diff_vs_random_matched` | gate − random at same rate | — | −3.79 pp [−4.76, −2.86] | confidence gate significantly better than random | `gate_baseline_comparison.csv` |
| `gate.xgb.diff_vs_random@77.8%` | gate thresholded to 77.8 % autonomy (thr 0.8441) | — | 4.17 %; −3.45 pp [−4.44, −2.54] | holds at the repo's autonomy rate | `gate_baseline_comparison.csv` |
| `gate.xgb.mean_ape_executed` | mean APE of executed decisions | — | 3.84 % vs 4.33 % (always-auto) | | `gate_baseline_comparison.csv` |
| `gate.xgb.precision/recall@0.85` | deferral precision / recall | — | 0.173 / 0.638 (F1 0.272) | | `gate_threshold_sweep.csv` |
| `gate.xgb.best_f1_threshold` | sweep 0.50–0.99 | 0.85 assumed | 0.80 (F1 0.439, 93.6 % autonomous, bad-exec 4.85 %) | 0.85 is not F1-optimal; it favours recall | `gate_threshold_sweep.csv` |
| `gate.xgb.auroc` | low confidence as detector of APE > 10 % | — | 0.743 | | `gate_threshold_sweep.csv`, `gate_roc_curve.png` |
| `gate.heuristic.promo_holiday` | non-ML rule "defer iff promo or holiday" | — | 4.49 % bad-exec at 92.2 % autonomy, precision 0.448 | **a trivial rule nearly matches the ML gate** — the synthetic noise is driven by these flags; must be reported as a limitation | `gate_baseline_comparison.csv` |
| `gate.heuristic.volatility` | defer highest trailing-7-day CV, same deferral count | — | 5.87 % | ML gate beats a volatility heuristic | `gate_baseline_comparison.csv` |
| `gate.prophet_native.bad_exec@0.85` | | — | 6.80 % vs 6.18 % random: +0.62 pp [+0.31, +1.10] | with Prophet's native confidence the gate is **worse than random** | `gate_baseline_comparison.csv` |
| `gate.prophet_conformal.bad_exec@0.85` | with Task 3 recalibration | — | 3.13 %; −3.05 pp [−4.07, −2.26]; AUROC 0.799 | recalibration turns Prophet's gate from harmful to useful | `gate_baseline_comparison.csv` |
| `gate.lstm.auto_rate@0.85` | MC-Dropout and ensemble | — | 100 % (every confidence ≥ 0.85); bad-exec 25.6 % / 27.2 % | **the LSTM gate never defers** at the deployed threshold | `gate_threshold_sweep.csv` |
| tolerance sensitivity | APE tolerance 5 % / 15 % | — | in CSV | | `gate_threshold_sweep.csv` |

---

## 3. Prophet confidence calibration (Task 3)

| key | measures | before | after | demonstrates | source |
|---|---|---|---|---|---|
| `prophet.corr_conf_ape.pearson` | Pearson r(confidence, APE), alpha = 1.0, test n = 1,278 | +0.250 (documented); +0.253 (reproduced) | split conformal **+0.220**; normalized conformal **−0.400**; CQR −0.392 | direction fixed only by context-conditioned conformal; plain split conformal does not fix it | `prophet_calibration_before_after.csv` |
| `prophet.corr_conf_ape.spearman` | rank correlation (alpha-invariant) | +0.094 | +0.105 / **−0.174** / −0.123 | improvement is real but modest in rank terms | `prophet_calibration_before_after.csv` |
| `prophet.auroc` | detect APE > 10 % | 0.284 | 0.260 / **0.799** / 0.765 | | `prophet_calibration_before_after.csv` |
| `prophet.coverage` | empirical coverage of nominal 80 % interval | 0.862 | 0.820 / **0.775** / 0.773 | conformal under-covers by 2.5 pp: calibration and test windows are not exchangeable | `prophet_calibration_before_after.csv` |
| `prophet.series_with_negative_r` | of 9 series | 1 | 1 / 9 / 9 | consistent across series | `prophet_calibration_before_after.csv` |
| `prophet.difficulty_model` | fitted relative-residual model | — | intercept 0.0407, is_promo +0.0525, is_holiday +0.0061, is_weekend −0.0041 | gain comes from learning promo heteroscedasticity, which the generator injects by construction | `prophet_calibration_before_after.csv` |

---

## 4. LSTM uncertainty (Task 4)

| key | measures | before | after | demonstrates | source |
|---|---|---|---|---|---|
| `lstm.mc_dropout.pearson` | saved models, 30 passes, alpha 0.5, n = 1,251 | −0.081 (documented) | −0.105 (Spearman −0.135, AUROC 0.549; MC noise ±0.004 across unseeded runs) | weak, correct direction — reproduced | `lstm_uncertainty_comparison.csv` |
| `lstm.mc_dropout.members.pearson` | MC-Dropout on 5 newly trained members | — | mean −0.091, SD over seeds (in file) | | `lstm_uncertainty_comparison.csv` |
| `lstm.deep_ensemble.pearson` | 5-member ensemble std | — | **−0.033** (Spearman −0.021, AUROC 0.531) | **ensemble is worse than MC-Dropout** | `lstm_uncertainty_comparison.csv` |
| `lstm.ensemble_plus_mc.pearson` | combined variance | — | −0.094 (AUROC 0.556) | no gain from combining | `lstm_uncertainty_comparison.csv` |
| `lstm.coverage` | nominal 80 % band | not reported | MC-Dropout 0.233; ensemble 0.062 | both intervals severely over-confident | `lstm_uncertainty_comparison.csv` |
| `lstm.ensemble.rmse_mean_of_series` | accuracy of ensemble mean | 14.27 (saved) | 14.20 | | `lstm_seed_variance.csv` |
| verdict | | | **neither method gives a usable confidence signal**; MC-Dropout is marginally better | negative result | `lstm_uncertainty_notes.md` |

---

## 5. Smart-contract security (Task 5)

| key | measures | before (v1) | after (v2) | demonstrates | source |
|---|---|---|---|---|---|
| `contract.access_control` | who can call `logDecision` / `autoExecutePO` | anyone | owner-managed allow-list (`onlyAuthorizedCaller`) | unauthorised write/execute now reverts | `security_review.md`, `tests/test_contract.py` |
| `contract.double_execution` | POs per AUTONOMOUS decision | unbounded (test executes 2) | at most 1 | closes a replay hole | `tests/test_contract.py` |
| `contract.policy_consistency` | AUTO record with confidence < threshold, or threshold < 0.85 | accepted | reverts (on-chain floor `minAutonomousThreshold`, owner-set) | backend cannot weaken the autonomy policy | `tests/test_contract.py` |
| `contract.reentrancy` | | no external calls → not exploitable | `nonReentrant` added (defence in depth) | CEI holds trivially | `security_review.md` |
| `contract.oracle_risk` | is the confidence computation verified on-chain? | no | **no** | a compromised backend can still forge a high-confidence AUTO decision; named as a limitation | `security_review.md` |
| `contract.shap_hash` | what `shapHash` contains | claimed: hash of SHAP explanation | actually SHA-256 of `confidence_metadata` (constant `{"quantiles":[0.1,0.9]}` for XGBoost) | **the "SHAP hash on-chain" claim in `novelty_and_contribution.md` §3.4 is not true of the code**; not changed here | `security_review.md` |
| `contract.gas.logDecision` | mean gas, 9 replayed decisions | ~300,000 (doc); 311,007 (re-measured v1) | 314,946 (+1.3 %) | | `gas_measurements.csv` |
| `contract.gas.autoExecutePO` | mean gas, 7 POs | ~226,000 (doc); 228,768 (re-measured v1) | 255,610 (+11.7 %) | security costs ~27k gas per PO | `gas_measurements.csv` |
| `contract.gas.deploy` | | 1,495,969 | 1,962,276 | | `gas_measurements.csv` |

---

## 6. Tests (Task 6)

| key | before | after | source |
|---|---|---|---|
| `tests.count` | 0 automated tests | **84 passed, 0 failed** (Python 61, Solidity-via-web3 23) | `test_report.md`, `intermediate/pytest_output.txt` |
| `tests.coverage.target_modules` | — | 84 % raw; 100 % excluding `__main__` demo blocks (`confidence_estimator.py`, `gate.py`, `ss_rop_calculator.py`) | `test_report.md` |
| `tests.hardhat` | — | **21 passed, 0 failed** (mocha port of the contract suite, `blockchain/test/`) | `test_report.md`, `intermediate/log_hardhat_coverage.txt` |
| `tests.coverage.solidity` | — | `ReplenishmentLedger.sol`: **100 % statements, 100 % lines, 100 % functions, 95.65 % branches** (44/46 arms; both missing arms are the unreachable `nonReentrant` revert) | `test_report.md`, `intermediate/log_hardhat_coverage.txt` |

---

## 7. Diagrams (Task 7)

`class_diagram.puml/.svg` and `sequence_diagram.puml/.svg` (PlantUML 1.2024.7, Smetana
layout). They show the code **as implemented**, including divergences from the narrative.
Each divergence is a claim the paper should not make:
- each call uses **one** model chosen by `model_type`; nothing ensembles the three;
- Prophet and LSTM are **re-trained on every** `evaluate_series()` call (LSTM with 15
  epochs, not the saved 25-epoch models);
- explainability modules are **not invoked** on the decision path;
- the dashboard calls `pipeline.run_pipeline` in-process (not via FastAPI);
- `backend/schemas.py` is unused;
- the approve endpoint writes nothing on-chain.

Both SVGs were rasterised and inspected at full resolution. No clipped, overlapping or
unreadable elements; one arrowhead touches the `inventory` frame title. Fixed: four
inline `'` comments in `class_diagram.puml` were rendered as literal characters (PlantUML
only treats `'` as a comment at line start). They are now `//` in the `.puml`, and the
same four text nodes were patched in the `.svg`. A full PlantUML re-render was blocked by
the sandbox's policy on running downloaded code (the jar), so the `.svg` is the text-patched
1.2024.7 render, not a fresh one. Re-render with `PLANTUML_JAR=… run_all_experiments.py`.

---

## 8. Gas cost in USD (Task 8) — assumption-based arithmetic

| key | assumptions | value | source |
|---|---|---|---|
| `usd.per_decision.mainnet` | v2, 498,620 gas expected per decision, ETH $3,500 | $34.90 (20 gwei) / $87.26 (50) / $174.52 (100) / $0.87 (0.5) | `gas_cost_table.csv` |
| `usd.per_year.10k_sku_loc` | 3.65 M decisions/yr, ETH $3,500 | $127.4 M (20 gwei) … $637.0 M (100 gwei); $3.18 M at 0.5 gwei | `gas_cost_scale.csv` |
| `usd.per_decision.l2` | assumed 0.01–0.1 gwei rollup / 50–200 gwei POL @ $0.25 | $0.017–$0.175 / $0.006–$0.025 | `gas_cost_l2_scenarios.csv` |

ETH and gas prices are assumptions. Re-check them before publication (`gas_cost_analysis.md`).

---

## 9. Reproducibility (Task 9)

| key | value | source |
|---|---|---|
| `repro.command` | `bash publication_package/run_all_experiments.sh` (or `python publication_package/run_all_experiments.py`) | `run_all_experiments.py` |
| `repro.requirements` | all 21 packages pinned with `==` (previously 5 were `>=`; added scipy, matplotlib, pytest, pytest-cov, coverage) | `requirements.txt` |
| `repro.seeds` | numpy/python/keras 42; LSTM members 0–4 + TF op determinism; bootstrap `default_rng(42)`; Prophet `np.random.seed(42)` | `scripts/common.py` |
| `repro.clean_env_result` | **CONFIRMED.** Fresh venv (Python 3.13.2, Windows 11, no pip cache): `pip install -r requirements.txt` exit 0; all generated outputs deleted from a repo copy, then the one-command script run: every step exit 0, 84/84 tests pass, all 16 top-level and 6 intermediate CSVs **byte-identical** (max rel. diff 0.0), PNG and SVG byte-identical (after fixing the SVG's timestamp/ID salt). About 40 unpinned transitive deps resolved to newer versions without effect. Not tested on other OS/Python/CPU. | `reproducibility_check.md`, `intermediate/clean_env_*` |

---

## 10. Limitations / what did NOT improve (for the Discussion section)

1. **LSTM confidence is not fixed.** The deep ensemble did worse than MC-Dropout. Both
   signals are near-random (AUROC 0.53–0.56) and their intervals badly under-cover. At 0.85
   the LSTM path auto-executes every decision.
2. **Prophet fix is partial.** Pearson r becomes −0.40 only with context-conditioned
   conformal. Plain split conformal stays wrong (+0.22). Coverage falls 2.5 pp below
   target, and the gain relies on the promo heteroscedasticity the synthetic generator
   builds in.
3. **The ML gate's advantage over a trivial promo/holiday rule is small** on this
   synthetic data. Real-data validation is required before claiming the confidence score
   adds value beyond known calendar flags.
4. **The 0.85 threshold is not optimal.** F1 peaks at 0.80 for XGBoost. The threshold is
   one global value, although the three models' confidence scales differ widely.
5. **The original 77.8 %/22.2 % claim rests on n = 9.** Its two deferred cases were
   accurate forecasts.
6. **The README model comparison mixes pooled and mean-of-series RMSE.** On a like-for-like
   basis Prophet is significantly better than XGBoost.
7. **The ML computation is not verified on-chain** (oracle risk). `shapHash` does not
   hash SHAP output. Approvals are off-chain. Production would need a key-management
   (HSM/KMS) and multisig setup that is not implemented.
8. **Mainnet cost** is prohibitive at 20–100 gwei. L2 figures are scenario arithmetic,
   not measurements.
9. **Scope of the evidence.** All data is synthetic, there is one test window per series,
   and there is no rolling-origin backtest.
10. **Infeasible / not done:** evidential regression for the LSTM (the deep ensemble was
    the method tried). A fresh PlantUML re-render of the diagrams was blocked; the SVGs were
    inspected and text-patched instead. Reproducibility is verified on one OS/Python only.
    Done since the first draft: Solidity coverage (Hardhat), clean-env reproduction, visual
    diagram check, and git history (baseline v1 commit followed by a separate
    contract-hardening commit).
