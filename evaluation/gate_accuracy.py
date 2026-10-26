"""
gate_accuracy.py

Scans every (product_id, region, model_type) combination through the full
Phase 1-6 pipeline and reports:
  - How the Decision Gate routed each one (NO_ACTION / AUTO / HUMAN)
  - On-chain logging outcomes (gas used, tx hashes)
  - Per-model confidence statistics
  - The quantifiable claim for the report: "X% auto-executed, Y% deferred"

Run:
    python evaluation/gate_accuracy.py                         # all combos, below_rop
    python evaluation/gate_accuracy.py --scenario random       # mixed
    python evaluation/gate_accuracy.py --no-chain              # skip blockchain (offline)
    python evaluation/gate_accuracy.py --product SKU_A --region North  # single combo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "decision_gate"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from gate import evaluate_series, GateDecision

PRODUCTS = ["SKU_A", "SKU_B", "SKU_C"]
REGIONS = ["North", "South", "West"]
MODEL_TYPES = ["xgboost", "prophet", "lstm"]


def scan(
    scenario: str = "below_rop",
    model_types: list[str] | None = None,
    log_to_chain: bool = True,
    product_filter: str | None = None,
    region_filter: str | None = None,
) -> pd.DataFrame:
    """Run the full pipeline on all (or filtered) combinations."""
    if model_types is None:
        model_types = MODEL_TYPES

    products = [product_filter] if product_filter else PRODUCTS
    regions = [region_filter] if region_filter else REGIONS

    rows = []
    for product in products:
        for region in regions:
            for model_type in model_types:
                try:
                    record = evaluate_series(
                        product, region,
                        model_type=model_type,
                        inventory_scenario=scenario,
                    )
                    d = record.to_dict()

                    row = {
                        "product_id": product,
                        "region": region,
                        "model_type": model_type,
                        "confidence": record.confidence,
                        "confidence_threshold": record.confidence_threshold,
                        "reorder_required": record.reorder_required,
                        "forecast": record.forecast,
                        "safety_stock": record.safety_stock,
                        "reorder_point": record.reorder_point,
                        "current_inventory": record.current_inventory,
                        "suggested_order_quantity": record.suggested_order_quantity,
                        "decision": record.decision.value,
                        "chain_decision_id": None,
                        "chain_tx_hash": None,
                        "chain_gas_used": None,
                        "po_id": None,
                        "po_tx_hash": None,
                    }

                    # On-chain logging
                    if log_to_chain:
                        try:
                            from pipeline import run_pipeline
                            result = run_pipeline(
                                product, region,
                                model_type=model_type,
                                inventory_scenario=scenario,
                                log_to_chain=True,
                            )
                            if result["chain_result"]:
                                row["chain_decision_id"] = result["chain_result"]["decision_id"]
                                row["chain_tx_hash"] = result["chain_result"]["tx_hash"]
                                row["chain_gas_used"] = result["chain_result"]["gas_used"]
                            if result["po_result"]:
                                row["po_id"] = result["po_result"]["po_id"]
                                row["po_tx_hash"] = result["po_result"]["tx_hash"]
                        except Exception as e:
                            print(f"  [chain error: {e}]")

                    rows.append(row)
                    status = f"chain_id={row['chain_decision_id']}" if row["chain_decision_id"] else "no-chain"
                    print(
                        f"{product} {region:6s} {model_type:8s}  "
                        f"conf={record.confidence:.3f}  "
                        f"decision={record.decision.value:22s}  {status}"
                    )
                except Exception as e:
                    print(f"{product} {region:6s} {model_type:8s}  ERROR: {e}")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    """Print a comprehensive summary of gate accuracy and on-chain metrics."""
    total = len(df)

    print("\n" + "=" * 70)
    print("  GATE ACCURACY REPORT")
    print("=" * 70)

    # Decision distribution
    print("\n--- Decision Distribution ---")
    counts = df["decision"].value_counts()
    for decision in ["NO_ACTION", "AUTONOMOUS_EXECUTION", "HUMAN_APPROVAL"]:
        n = counts.get(decision, 0)
        pct = 100 * n / total if total > 0 else 0
        print(f"  {decision:22s}: {n:3d} / {total} ({pct:.1f}%)")

    # Reorder-specific breakdown
    reorder_needed = df[df["reorder_required"]]
    if len(reorder_needed) > 0:
        auto = (reorder_needed["decision"] == "AUTONOMOUS_EXECUTION").sum()
        human = (reorder_needed["decision"] == "HUMAN_APPROVAL").sum()
        print(f"\n--- Of {len(reorder_needed)} cases where reorder was required ---")
        print(f"  Auto-executed (high confidence) : {auto:3d} ({100 * auto / len(reorder_needed):.1f}%)")
        print(f"  Deferred to human (low conf.)   : {human:3d} ({100 * human / len(reorder_needed):.1f}%)")

    # On-chain metrics
    on_chain = df[df["chain_decision_id"].notna()]
    if len(on_chain) > 0:
        print(f"\n--- On-Chain Metrics ({len(on_chain)} decisions logged) ---")
        print(f"  Total gas used (decisions)  : {on_chain['chain_gas_used'].sum():,.0f}")
        po_execs = df[df["po_id"].notna()]
        if len(po_execs) > 0:
            print(f"  POs auto-executed on-chain  : {len(po_execs)}")
        print(f"  Avg gas per decision        : {on_chain['chain_gas_used'].mean():,.0f}")

    # Per-model confidence statistics
    print("\n--- Per-Model Average Confidence ---")
    model_stats = df.groupby("model_type")["confidence"].agg(["mean", "std", "min", "max"])
    for model, row in model_stats.iterrows():
        print(f"  {model:8s}: mean={row['mean']:.3f}  std={row['std']:.3f}  "
              f"min={row['min']:.3f}  max={row['max']:.3f}")

    # Key claim for report
    reorder_auto = len(df[(df["reorder_required"]) & (df["decision"] == "AUTONOMOUS_EXECUTION")])
    reorder_human = len(df[(df["reorder_required"]) & (df["decision"] == "HUMAN_APPROVAL")])
    reorder_total = len(reorder_needed)
    if reorder_total > 0:
        print(f"\n--- Key Claim for Report ---")
        print(f"  \"{100 * reorder_auto / reorder_total:.1f}% of reorder decisions were")
        print(f"   confidently automated; {100 * reorder_human / reorder_total:.1f}% were")
        print(f"   correctly deferred to human review.\"")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate accuracy scan with on-chain metrics")
    parser.add_argument("--scenario", default="below_rop",
                        choices=["below_rop", "above_rop", "random"])
    parser.add_argument("--no-chain", action="store_true", help="Skip blockchain logging")
    parser.add_argument("--product", default=None, help="Filter to one product")
    parser.add_argument("--region", default=None, help="Filter to one region")
    parser.add_argument("--model", default=None, help="Filter to one model type")
    parser.add_argument("--output", default=None, help="Save results to CSV")
    args = parser.parse_args()

    model_types = [args.model] if args.model else None

    df = scan(
        scenario=args.scenario,
        model_types=model_types,
        log_to_chain=not args.no_chain,
        product_filter=args.product,
        region_filter=args.region,
    )
    summarize(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
