"""
run_all_experiments.py  (Task 9)

Regenerates every CSV / figure / number in publication_package/ from the raw
data and the saved model artefacts, with fixed seeds, in one command:

    python publication_package/run_all_experiments.py            # full run (~10-15 min, CPU)
    python publication_package/run_all_experiments.py --keep-cache  # reuse trained LSTM ensemble

Steps
  0. recompile ReplenishmentLedger.sol and check it matches the committed ABI
  1. s01  cache test-set predictions of the saved models (+ metric reproduction check)
  2. s02  Task 1  bootstrap CIs, paired model differences, XGBoost seed variance
  3. s03  Task 3  Prophet conformal recalibration
  4. s04  Task 4  LSTM MC-Dropout vs 5-member deep ensemble (trains 45 LSTMs)
  5. s05  Task 2  gate threshold sweep, baselines, ROC/PR figure
  6. s06  Task 5/8 gas measurement on a private Ganache (needs `ganache` CLI)
  7. s07  Task 8  gas -> USD tables
  8. pytest suite with coverage (Task 6)  -> intermediate/pytest_*.txt
  9. Hardhat tests + Solidity line coverage (Task 6), only if blockchain/node_modules
     exists (`cd blockchain && npm ci`) -> intermediate/log_hardhat_coverage.txt
 10. PlantUML -> SVG (Task 7), only if PLANTUML_JAR points to a plantuml.jar
 11. reproducibility_manifest.csv (SHA-256 of every output)
 12. s08  consolidated results_bundle/ (RESULTS_ALL.xlsx / .md, FIGURES_ALL.pdf, archive zip)

Random seeds: numpy/python/keras seed 42 everywhere (common.SEED); LSTM
ensemble members seeds 0-4 with TensorFlow op determinism enabled; bootstrap
RNG numpy default_rng(42); Prophet interval sampling np.random.seed(42).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
SCRIPTS = PKG / "scripts"
INTER = PKG / "intermediate"
PY = sys.executable

STEPS = ["s01_build_predictions.py", "s02_statistical_significance.py", "s03_prophet_conformal.py",
         "s04_lstm_uncertainty.py", "s05_gate_analysis.py", "s06_gas_measurement.py",
         "s07_gas_cost_table.py"]


def run(cmd, cwd, log_name):
    t0 = time.time()
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    env = dict(os.environ, PYTHONHASHSEED="0", TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")
    with open(INTER / log_name, "w", encoding="utf-8") as log:
        p = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"    exit={p.returncode}  ({time.time() - t0:.0f}s)  log: intermediate/{log_name}", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-cache", action="store_true", help="reuse intermediate/lstm_ensemble_members.csv")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    INTER.mkdir(exist_ok=True)

    if not args.keep_cache:
        for f in INTER.glob("*.csv"):
            f.unlink()
        shutil.rmtree(INTER / "lstm_ensemble", ignore_errors=True)

    failures = []
    # 0. contract compile check
    tmp = INTER / "compiled_check.json"
    if run([PY, str(ROOT / "blockchain" / "scripts" / "compile.py"), "--out", str(tmp)], ROOT, "log_compile.txt") == 0:
        same = json.loads(tmp.read_text()) == json.loads((ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json").read_text())
        print(f"    compiled artefact identical to committed ABI/bytecode: {same}")
        if not same:
            failures.append("compile-mismatch")
    else:
        failures.append("compile")

    for s in STEPS:
        if run([PY, s], SCRIPTS, f"log_{s.replace('.py', '')}.txt") != 0:
            failures.append(s)

    if not args.skip_tests:
        rc = run([PY, "-m", "pytest", "--cov=confidence", "--cov=decision_gate", "--cov=inventory",
                  "--cov-report=term", f"--junitxml={INTER / 'pytest_junit.xml'}", "-q"], ROOT, "pytest_output.txt")
        if rc != 0:
            failures.append("pytest")

    npx = shutil.which("npx")
    if npx and (ROOT / "blockchain" / "node_modules").exists():
        if run([npx, "hardhat", "coverage"], ROOT / "blockchain", "log_hardhat_coverage.txt") != 0:
            failures.append("hardhat-coverage")
    else:
        print("\n(skip) Solidity coverage: run `npm ci` in blockchain/ to enable `npx hardhat coverage`")

    jar = os.environ.get("PLANTUML_JAR")
    if jar and Path(jar).exists():
        run(["java", "-jar", jar, "-tsvg", "class_diagram.puml", "sequence_diagram.puml"], PKG, "log_plantuml.txt")
    else:
        print("\n(skip) PlantUML render: set PLANTUML_JAR=/path/to/plantuml.jar (v1.2024.7 used) to re-render SVGs")

    rows = ["file,sha256,bytes"]
    for f in sorted(PKG.glob("*")):
        if f.is_file() and f.suffix in {".csv", ".png", ".svg", ".md", ".puml"} and f.name != "reproducibility_manifest.csv":
            rows.append(f"{f.name},{hashlib.sha256(f.read_bytes()).hexdigest()},{f.stat().st_size}")
    (PKG / "reproducibility_manifest.csv").write_text("\n".join(rows) + "\n")

    if run([PY, "s08_bundle_results.py"], SCRIPTS, "log_s08_bundle_results.txt") != 0:
        failures.append("s08_bundle_results.py")

    print("\nFAILED STEPS:", failures if failures else "none")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
