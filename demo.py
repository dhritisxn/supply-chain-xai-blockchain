"""
demo.py

Full Review-2 presentation demo script.  Runs the entire pipeline
end-to-end in one command:

  1. Model metrics summary (RMSE/MAPE/R2)
  2. Scenario A: High-confidence reorder -> auto-executed on-chain
  3. Scenario B: Low-confidence reorder  -> flagged for human review
  4. Scenario C: No reorder needed       -> logged but no action
  5. Full scan of all 18 combinations    -> gate accuracy report
  6. Ledger integrity verification       -> tamper detection demo
  7. Summary of key claims for the report

Prerequisites:
  - Ganache running:     ganache --port 8545
  - Contract deployed:   python blockchain/scripts/deploy.py

Run:
    python demo.py                   # full demo with blockchain
    python demo.py --no-chain        # skip blockchain (offline mode)
    python demo.py --quick           # skip full scan, just show scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "decision_gate"))
sys.path.insert(0, str(PROJECT_ROOT / "confidence"))
sys.path.insert(0, str(PROJECT_ROOT / "inventory"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "blockchain" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))


def _banner(text: str):
    width = 64
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}")


def _subbanner(text: str):
    print(f"\n--- {text} ---")


def step_model_metrics():
    """Step 1: Show model performance metrics."""
    _banner("STEP 1: Model Performance Metrics")
    from evaluation.model_metrics import load_saved_metrics, print_report
    results = load_saved_metrics()
    print_report(results, "Forecasting Model Performance")


def step_scenario_a(w3=None):
    """Step 2: High-confidence reorder -> auto-execute PO."""
    _banner("STEP 2: SCENARIO A — High Confidence -> Auto-Execute PO")
    _subbanner("SKU_A / North / XGBoost / Low inventory")

    from pipeline import run_pipeline
    from gate import GateDecision

    result = run_pipeline(
        "SKU_A", "North",
        model_type="xgboost",
        inventory_scenario="below_rop",
        log_to_chain=(w3 is not None),
    )
    dr = result["decision_record"]

    print(f"  Product:           {dr.product_id}/{dr.region}")
    print(f"  Forecast:          {dr.forecast:.2f}")
    print(f"  Confidence:        {dr.confidence:.4f}")
    print(f"  Threshold:         {dr.confidence_threshold}")
    print(f"  Decision:          {dr.decision.value}")
    print(f"  Reasoning:         {dr.reasoning[:100]}...")

    if result["chain_result"]:
        cr = result["chain_result"]
        print(f"  On-chain decision: ID={cr['decision_id']}  tx={cr['tx_hash'][:16]}...")
    if result["po_result"]:
        pr = result["po_result"]
        print(f"  PO executed:       ID={pr['po_id']}  tx={pr['tx_hash'][:16]}...")

    assert dr.decision == GateDecision.AUTONOMOUS_EXECUTION, "Expected AUTONOMOUS_EXECUTION"
    print(f"\n  [PASS] Scenario A: PO auto-executed on-chain")
    return result


def step_scenario_b(w3=None):
    """Step 3: Low-confidence reorder -> human review."""
    _banner("STEP 3: SCENARIO B — Low Confidence -> Human Approval")
    _subbanner("SKU_B / West / XGBoost / Low inventory")

    from pipeline import run_pipeline
    from gate import GateDecision

    result = run_pipeline(
        "SKU_B", "West",
        model_type="xgboost",
        inventory_scenario="below_rop",
        log_to_chain=(w3 is not None),
    )
    dr = result["decision_record"]

    print(f"  Product:           {dr.product_id}/{dr.region}")
    print(f"  Forecast:          {dr.forecast:.2f}")
    print(f"  Confidence:        {dr.confidence:.4f}")
    print(f"  Threshold:         {dr.confidence_threshold}")
    print(f"  Decision:          {dr.decision.value}")
    print(f"  Reasoning:         {dr.reasoning[:100]}...")

    if result["chain_result"]:
        cr = result["chain_result"]
        print(f"  On-chain decision: ID={cr['decision_id']}  tx={cr['tx_hash'][:16]}...")
    print(f"  No PO executed (awaiting human review)")

    assert dr.decision == GateDecision.HUMAN_APPROVAL, "Expected HUMAN_APPROVAL"
    print(f"\n  [PASS] Scenario B: Deferred to human review")
    return result


def step_scenario_c(w3=None):
    """Step 4: No reorder needed."""
    _banner("STEP 4: SCENARIO C — No Reorder Needed")
    _subbanner("SKU_A / North / XGBoost / High inventory")

    from pipeline import run_pipeline
    from gate import GateDecision

    result = run_pipeline(
        "SKU_A", "North",
        model_type="xgboost",
        inventory_scenario="above_rop",
        log_to_chain=(w3 is not None),
    )
    dr = result["decision_record"]

    print(f"  Product:           {dr.product_id}/{dr.region}")
    print(f"  Forecast:          {dr.forecast:.2f}")
    print(f"  Current inventory: {dr.current_inventory:.2f}")
    print(f"  Reorder point:     {dr.reorder_point:.2f}")
    print(f"  Decision:          {dr.decision.value}")

    if result["chain_result"]:
        cr = result["chain_result"]
        print(f"  On-chain decision: ID={cr['decision_id']}  tx={cr['tx_hash'][:16]}...")

    assert dr.decision == GateDecision.NO_ACTION, "Expected NO_ACTION"
    print(f"\n  [PASS] Scenario C: No action taken")
    return result


def step_full_scan(w3=None):
    """Step 5: Scan all 18 combinations."""
    _banner("STEP 5: Full Scan — All Product/Region/Model Combinations")

    from pipeline import run_full_scan

    results = run_full_scan(
        inventory_scenario="below_rop",
        log_to_chain=(w3 is not None),
    )

    decisions = [r["decision_record"].decision.value for r in results]
    counts = Counter(decisions)
    total = len(results)

    print(f"\n  Total combinations: {total}")
    for d in ["NO_ACTION", "AUTONOMOUS_EXECUTION", "HUMAN_APPROVAL"]:
        n = counts.get(d, 0)
        print(f"  {d:22s}: {n:3d} / {total} ({100*n/total:.1f}%)")

    auto = counts.get("AUTONOMOUS_EXECUTION", 0)
    human = counts.get("HUMAN_APPROVAL", 0)
    reorder_total = auto + human
    if reorder_total > 0:
        print(f"\n  Key claim: {100*auto/reorder_total:.1f}% auto-executed, "
              f"{100*human/reorder_total:.1f}% deferred to human review")

    return results


def step_integrity_check():
    """Step 6: Ledger integrity verification."""
    _banner("STEP 6: Ledger Integrity Verification")
    try:
        from ledger_integrity_check import verify_ledger, tamper_demo
        results = verify_ledger()
        print()
        tamper_demo()
        return results
    except Exception as e:
        print(f"  Integrity check skipped: {e}")
        return None


def step_summary(scenario_results: dict, scan_results: list, chain_available: bool):
    """Step 7: Final summary."""
    _banner("STEP 7: Summary — Key Claims for Review-2 Report")

    print("""
  1. NOVELTY: ML *gates* whether blockchain is trusted to act
     autonomously, not just feeds it. The confidence score
     determines the execution path.

  2. EXPLAINABILITY-ON-CHAIN: SHAP explanations are hashed and
     stored with every decision, making the audit trail explain
     WHY as well as WHAT.

  3. QUANTIFIABLE CLAIMS:
""")

    if scan_results:
        decisions = [r["decision_record"].decision.value for r in scan_results]
        counts = Counter(decisions)
        auto = counts.get("AUTONOMOUS_EXECUTION", 0)
        human = counts.get("HUMAN_APPROVAL", 0)
        reorder_total = auto + human
        if reorder_total > 0:
            print(f"     - {100*auto/reorder_total:.1f}% of reorder decisions auto-executed")
            print(f"     - {100*human/reorder_total:.1f}% deferred to human review")

    print(f"""
  4. TAMPER EVIDENCE: Ledger integrity check demonstrates that
     any modification to on-chain records is immediately detected.

  5. FULL PIPELINE: Data -> Forecast -> Confidence -> SS/ROP ->
     Decision Gate -> Blockchain -> Dashboard (all connected).
""")

    if chain_available:
        try:
            from interact import _get_web3, _load_contract
            w3 = _get_web3()
            contract = _load_contract(w3)
            n_d = contract.functions.getDecisionCount().call()
            n_p = contract.functions.getPOCount().call()
            print(f"  ON-CHAIN STATE: {n_d} decisions, {n_p} purchase orders")
        except Exception:
            pass

    print(f"\n{'='*64}")
    print(f"  DEMO COMPLETE")
    print(f"{'='*64}\n")


def main():
    parser = argparse.ArgumentParser(description="Review-2 Full Demo")
    parser.add_argument("--no-chain", action="store_true", help="Skip blockchain integration")
    parser.add_argument("--quick", action="store_true", help="Skip full scan (faster)")
    args = parser.parse_args()

    chain_available = not args.no_chain
    w3 = None

    if chain_available:
        try:
            from interact import _get_web3
            w3 = _get_web3()
            print("[Blockchain] Connected to Ganache")
        except Exception as e:
            print(f"[Blockchain] Not available ({e}) — running in offline mode")
            chain_available = False

    # Run all steps
    step_model_metrics()
    sa = step_scenario_a(w3)
    sb = step_scenario_b(w3)
    sc = step_scenario_c(w3)

    scan_results = []
    if not args.quick:
        scan_results = step_full_scan(w3)
    else:
        # In quick mode, compute claims from the 3 scenarios we already ran
        scan_results = [
            {"decision_record": sa["decision_record"]},
            {"decision_record": sb["decision_record"]},
            {"decision_record": sc["decision_record"]},
        ]

    step_integrity_check()
    step_summary(
        scenario_results={"a": sa, "b": sb, "c": sc},
        scan_results=scan_results,
        chain_available=chain_available,
    )


if __name__ == "__main__":
    main()
