# Clean-environment reproducibility check (Task 9): status CONFIRMED (2026-10-26)

## Procedure
1. Fresh venv on Windows 11 / Python 3.13.2, no pip cache:
   `python -m venv cleanvenv && cleanvenv/Scripts/pip install --no-cache-dir -r requirements.txt`
   finished with exit 0. Installed set: `intermediate/clean_env_pip_freeze.txt`.
2. Copied the repo (code, raw data, saved models) to a separate directory, **deleted every
   generated output** in its `publication_package/` (all CSVs, the ROC figures, every
   `intermediate/` CSV, the trained LSTM ensemble, all logs), then ran the one-command
   script with the clean interpreter:
   `PYTHON=cleanvenv/Scripts/python.exe bash publication_package/run_all_experiments.sh`.
3. Compared the regenerated outputs with the originals using
   `scripts/compare_outputs.py` plus a byte comparison of the figures.

## Result
- Every step exited 0 (`intermediate/clean_env_run_all.log`): compile check (recompiled
  bytecode identical to the committed ABI/bytecode), s01–s07, and pytest
  **84 passed** (`intermediate/clean_env_pytest_output.txt`). Wall time ≈ 23 min, of which
  s04 (training 45 LSTMs) took 17 min.
- **All 16 top-level CSVs and all 6 `intermediate/` CSVs are byte-identical** to the
  originals: worst relative difference 0.0 (`intermediate/clean_env_compare.txt`).
- `gate_roc_curve.png` is byte-identical. `gate_roc_curve.svg` first differed only in
  matplotlib's embedded timestamp and random element IDs. `s05_gate_analysis.py` now fixes
  `svg.hashsalt` and drops the date. After that change the SVG is byte-identical across
  runs and across the two environments.

## What "clean" did and did not vary
- `requirements.txt` pins the 21 top-level packages exactly, and they matched. About 40
  **unpinned transitive** dependencies resolved to newer patch/minor releases than in the
  original environment (`requirements-lock.txt`), e.g. numba 0.67.0→0.68.0,
  protobuf 7.36.1→7.36.2, multidict 6.7.1→7.2.1. The outputs were unaffected. For an exact
  replica of the original environment, install `requirements-lock.txt` instead.
- Same machine, OS and Python version. Not tested: Linux/macOS, other CPUs, other Python
  versions. TensorFlow/XGBoost floating-point results can differ across platforms. Expect
  agreement within tolerance there, not byte identity; check with
  `compare_outputs.py --rtol 1e-6`.
- Needs the `ganache` CLI on PATH (s06, contract tests). The Hardhat coverage step runs only
  if `blockchain/node_modules` exists (`cd blockchain && npm ci`). The PlantUML render runs
  only if `PLANTUML_JAR` is set.
