"""
pipeline.py

End-to-end integration:  data  ->  forecast  ->  confidence  ->  SS/ROP
->  decision gate  ->  blockchain logging  ->  PO execution (if applicable)

This is the single entry point the FastAPI backend and dashboard will call.
It eliminates the need for separate manual wiring between gate.py and
interact.py.

Usage:
    # Run one SKU/region through the full pipeline
    result = run_pipeline("SKU_A", "North", model_type="xgboost")

    # Run all 18 combinations and collect results
    results = run_full_scan(inventory_scenario="below_rop")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "decision_gate"))
sys.path.insert(0, str(PROJECT_ROOT / "confidence"))
sys.path.insert(0, str(PROJECT_ROOT / "inventory"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "explainability"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "blockchain" / "scripts"))

from gate import evaluate_series, GateDecision, DecisionRecord

PRODUCTS = ["SKU_A", "SKU_B", "SKU_C"]
REGIONS = ["North", "South", "West"]
MODEL_TYPES = ["xgboost", "prophet", "lstm"]


def _get_w3():
    """Lazy Web3 connection (only created if blockchain is reachable)."""
    try:
        from interact import _get_web3
        return _get_web3()
    except Exception:
        return None


def run_pipeline(
    product_id: str,
    region: str,
    model_type: str = "xgboost",
    inventory_scenario: str = "random",
    current_inventory: Optional[float] = None,
    log_to_chain: bool = True,
) -> dict:
    """
    Run the full ML -> Confidence -> SS/ROP -> Decision Gate -> Blockchain
    pipeline for a single product/region/model combination.

    Returns a dict with:
      - decision_record: the gate.py DecisionRecord (full context)
      - chain_result: dict with on-chain tx details (or None if chain unavailable)
      - po_result: dict with PO execution details (or None)
    """
    # Step 1-4: ML pipeline + decision gate
    record = evaluate_series(
        product_id, region,
        model_type=model_type,
        inventory_scenario=inventory_scenario,
        current_inventory=current_inventory,
    )

    result = {
        "decision_record": record,
        "chain_result": None,
        "po_result": None,
    }

    if not log_to_chain:
        return result

    # Step 5-6: Blockchain integration
    w3 = _get_w3()
    if w3 is None:
        return result  # chain unavailable, return pipeline result without on-chain logging

    from interact import log_decision as _log_decision, execute_po as _execute_po

    chain_result = _log_decision(record, w3=w3)
    result["chain_result"] = chain_result

    if record.decision == GateDecision.AUTONOMOUS_EXECUTION:
        po_result = _execute_po(chain_result["decision_id"], w3=w3)
        result["po_result"] = po_result

    return result


def run_full_scan(
    inventory_scenario: str = "below_rop",
    model_types: list[str] | None = None,
    log_to_chain: bool = True,
) -> list[dict]:
    """
    Run every (product, region, model) combination through the full pipeline.
    Returns a list of result dicts with on-chain outcomes.
    """
    if model_types is None:
        model_types = MODEL_TYPES

    results = []
    for product in PRODUCTS:
        for region in REGIONS:
            for model_type in model_types:
                try:
                    r = run_pipeline(
                        product, region,
                        model_type=model_type,
                        inventory_scenario=inventory_scenario,
                        log_to_chain=log_to_chain,
                    )
                    dr = r["decision_record"]
                    print(
                        f"{product} {region:6s} {model_type:8s}  "
                        f"conf={dr.confidence:.3f}  decision={dr.decision.value}"
                        f"{'  chain=' + r['chain_result']['tx_hash'][:12] + '...' if r['chain_result'] else ''}"
                    )
                    results.append(r)
                except Exception as e:
                    print(f"{product} {region:6s} {model_type:8s}  ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="End-to-end pipeline runner")
    parser.add_argument("--product", default=None, help="Product ID (e.g. SKU_A)")
    parser.add_argument("--region", default=None, help="Region (e.g. North)")
    parser.add_argument("--model", default="xgboost", help="Model type")
    parser.add_argument("--scenario", default="below_rop",
                        choices=["below_rop", "above_rop", "random"])
    parser.add_argument("--scan", action="store_true", help="Run all 18 combos")
    parser.add_argument("--no-chain", action="store_true", help="Skip blockchain logging")
    args = parser.parse_args()

    if args.scan:
        results = run_full_scan(
            inventory_scenario=args.scenario,
            log_to_chain=not args.no_chain,
        )
        decisions = [r["decision_record"].decision.value for r in results]
        from collections import Counter
        counts = Counter(decisions)
        print(f"\n=== Pipeline scan complete: {len(results)} combinations ===")
        for d in ["NO_ACTION", "AUTONOMOUS_EXECUTION", "HUMAN_APPROVAL"]:
            print(f"  {d:22s}: {counts.get(d, 0)}")
    elif args.product and args.region:
        r = run_pipeline(
            args.product, args.region,
            model_type=args.model,
            inventory_scenario=args.scenario,
            log_to_chain=not args.no_chain,
        )
        print(json.dumps(r["decision_record"].to_dict(), indent=2))
        if r["chain_result"]:
            print(f"On-chain: {json.dumps(r['chain_result'], indent=2)}")
        if r["po_result"]:
            print(f"PO: {json.dumps(r['po_result'], indent=2)}")
    else:
        parser.print_help()
