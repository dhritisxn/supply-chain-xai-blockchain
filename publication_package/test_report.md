# Automated test suite report (Task 6)

Environment: Windows 11, Python 3.13.2, pytest 8.4.2, pytest-cov 7.0.0, web3 8.0.0,
Ganache v7.9.2 (Node 22.19.0), solc 0.8.19. Raw outputs:
`intermediate/pytest_output.txt`, `intermediate/pytest_junit.xml` (written by
`run_all_experiments.py`).

## Result: 84 passed, 0 failed, 0 skipped (≈ 35–40 s)

| file | tests | covers |
|---|---:|---|
| `tests/test_confidence_estimator.py` | 18 | `exp(-alpha·width/|forecast|)` against hand-computed values; alpha scaling; zero width → 1.0; monotonicity; forecast = 0 and near 0 (1e-12 … 1e-6, eps floor); crossed bounds; [0,1] range over 500 random draws; threshold from YAML; inclusive boundary; `to_dict` rounding; XGBoost / Prophet / LSTM estimators (LSTM inverse-scaling of the MC band via a monkeypatched `predict_with_uncertainty`) |
| `tests/test_gate.py` | 17 | all three routing branches (parametrised), inclusive threshold boundary, default threshold from config, custom thresholds, record contents, model-type override |
| `tests/test_ss_rop_calculator.py` | 12 | SS = 1.645·σ·√7 and ROP = D·7 + SS vs hand-computed values (e.g. σ = 10 → SS 43.52260906701252; D = 100 → ROP 743.5226090670125); population-std trailing window; order-up-to quantity; inventory exactly at ROP; simulated-inventory ranges |
| `tests/test_conformal.py` | 6 | finite-sample quantile rank; ≥ 79 % mean coverage for split / normalized / CQR on exchangeable synthetic data; constant vs adaptive widths |
| `tests/test_integration.py` | 8 | README scenarios A/B/C end-to-end through `gate.evaluate_series()` with the saved XGBoost models; Prophet and LSTM paths; facade |
| `tests/test_contract.py` | 23 | Solidity, via web3.py on a private Ganache (port 8546): `logDecision()` storage + events, `autoExecutePO()` happy path and every revert, **unauthorised callers rejected for both functions**, owner grant/revoke, non-owner admin rejected, double-execution rejected, confidence/threshold checks and on-chain floor, ownership transfer, zero-address and missing-id reverts, v1 vulnerability demonstration, unchanged `interact.py` bridge against v2 |

The contract tests are skipped automatically if the `ganache` CLI is missing.

## Line coverage (`--cov=confidence --cov=decision_gate --cov=inventory`)

| module | raw | excluding `if __name__ == "__main__":` demo blocks |
|---|---:|---:|
| `confidence/confidence_estimator.py` | 78 % | **100 %** |
| `decision_gate/gate.py` | 89 % | **100 %** |
| `inventory/ss_rop_calculator.py` | 89 % | **100 %** |
| `confidence/conformal.py` (new) | 100 % | 100 % |
| `confidence/calibrate.py` (not in scope; slow retraining script) | 0 % | 0 % |
| **total of the three target modules** | 84 % (291 stmts, 47 missed) | **100 %** (241 stmts) |
| total incl. `calibrate.py` | 64 % | 74 % |

Every uncovered line in the three target modules is in a `__main__` demo block.

## Solidity coverage (Hardhat + solidity-coverage)

`blockchain/` now contains a Hardhat 2 project (`hardhat.config.js`, `package.json` +
`package-lock.json`, pinned: hardhat 2.22.17, solidity-coverage 0.8.14, ethers 6.13.4,
chai 4.5.0) used only for testing. `blockchain/test/ReplenishmentLedger.test.js` is a
mocha port of `tests/test_contract.py`: 21 tests. The v1 vulnerability demo and the
`interact.py` bridge check stay Python-only. Compiler: solc 0.8.19, optimizer 200 (same as
`scripts/compile.py`). Result: **21 passing, 0 failing**.

| file | statements | branches | functions | lines |
|---|---:|---:|---:|---:|
| `ReplenishmentLedger.sol` | **100 %** | **95.65 %** (44/46 arms) | **100 %** | **100 %** |

The two uncovered branch arms are the same revert path, counted twice: the
`"Reentrant call"` failure of the `nonReentrant` guard, once in the modifier (line 160) and
once where it is applied to `autoExecutePO` (line 287). It cannot be reached because the
contract makes no external call through which to re-enter. Coverage-only setting:
`.solcover.js` enables the Yul optimizer for the instrumented build. Without it,
instrumentation pushes `logDecision()` (9 parameters) into "Stack too deep". The deployed
artefact is unaffected. Raw output: `intermediate/log_hardhat_coverage.txt`.

Bug found by the suite while it was being written: `conformal_quantile()` first used
`np.quantile(..., method="higher")`, which returns the ⌈(n+1)(1−α)⌉+1-th order statistic
for some n (slightly too-wide intervals). Fixed before any reported numbers were produced.

## How to re-run
```bash
pip install -r requirements.txt
npm install -g ganache            # only for tests/test_contract.py
pytest                            # all 84 tests (pytest.ini sets testpaths = tests)
pytest --cov=confidence --cov=decision_gate --cov=inventory --cov-report=term-missing
pytest tests/test_contract.py     # Solidity tests only (web3.py + Ganache)
cd blockchain && npm ci && npx hardhat test       # 21 Hardhat tests
npx hardhat coverage                              # Solidity line/branch coverage
```
