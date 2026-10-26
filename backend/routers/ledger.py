"""
ledger.py

FastAPI router for the immutable audit ledger.
Provides endpoints to:
  - View all on-chain decisions and POs
  - Get ledger summary (counts, gas usage)
  - Run integrity verification
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blockchain" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

router = APIRouter(prefix="/ledger", tags=["ledger"])

FIXED_POINT = 10**18


def _from_fixed(val: int) -> float:
    return val / FIXED_POINT


def _get_contract():
    """Get the deployed contract instance."""
    try:
        from interact import _get_web3, _load_contract
        w3 = _get_web3()
        return _load_contract(w3)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Blockchain unavailable: {e}")


@router.get("/summary")
def ledger_summary():
    """Get counts of decisions and POs on-chain."""
    contract = _get_contract()
    n_decisions = contract.functions.getDecisionCount().call()
    n_pos = contract.functions.getPOCount().call()
    return {
        "total_decisions": n_decisions,
        "total_pos": n_pos,
    }


@router.get("/decisions")
def list_decisions():
    """Return all DecisionRecords stored on-chain."""
    contract = _get_contract()
    decision_ids = contract.functions.getAllDecisionIds().call()

    STATUS_NAMES = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}
    decisions = []
    for did in decision_ids:
        rec = contract.functions.getDecision(did).call()
        decisions.append({
            "id": rec[0],
            "sku": rec[1],
            "region": rec[2],
            "order_quantity": _from_fixed(rec[3]),
            "model_type": rec[4],
            "forecast_value": _from_fixed(rec[5]),
            "confidence_score": _from_fixed(rec[6]),
            "confidence_threshold": _from_fixed(rec[7]),
            "shap_hash": rec[8].hex() if isinstance(rec[8], bytes) else str(rec[8]),
            "status": STATUS_NAMES.get(rec[9], str(rec[9])),
            "status_int": rec[9],
            "timestamp": rec[10],
        })

    return {"decisions": decisions, "count": len(decisions)}


@router.get("/decisions/{decision_id}")
def get_decision(decision_id: int):
    """Get a specific DecisionRecord by ID."""
    contract = _get_contract()
    STATUS_NAMES = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}

    try:
        rec = contract.functions.getDecision(decision_id).call()
    except Exception:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")

    if rec[0] == 0:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")

    return {
        "id": rec[0],
        "sku": rec[1],
        "region": rec[2],
        "order_quantity": _from_fixed(rec[3]),
        "model_type": rec[4],
        "forecast_value": _from_fixed(rec[5]),
        "confidence_score": _from_fixed(rec[6]),
        "confidence_threshold": _from_fixed(rec[7]),
        "shap_hash": rec[8].hex() if isinstance(rec[8], bytes) else str(rec[8]),
        "status": STATUS_NAMES.get(rec[9], str(rec[9])),
        "status_int": rec[9],
        "timestamp": rec[10],
    }


@router.get("/purchase-orders")
def list_pos():
    """Return all PurchaseOrders stored on-chain."""
    contract = _get_contract()
    po_ids = contract.functions.getAllPOIds().call()

    STATUS_NAMES = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}
    pos = []
    for pid in po_ids:
        po = contract.functions.getPO(pid).call()
        pos.append({
            "id": po[0],
            "decision_id": po[1],
            "sku": po[2],
            "region": po[3],
            "quantity": _from_fixed(po[4]),
            "timestamp": po[5],
            "status": STATUS_NAMES.get(po[6], str(po[6])),
        })

    return {"purchase_orders": pos, "count": len(pos)}


@router.get("/purchase-orders/{po_id}")
def get_po(po_id: int):
    """Get a specific PurchaseOrder by ID."""
    contract = _get_contract()
    STATUS_NAMES = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}

    try:
        po = contract.functions.getPO(po_id).call()
    except Exception:
        raise HTTPException(status_code=404, detail=f"PO #{po_id} not found")

    if po[0] == 0:
        raise HTTPException(status_code=404, detail=f"PO #{po_id} not found")

    return {
        "id": po[0],
        "decision_id": po[1],
        "sku": po[2],
        "region": po[3],
        "quantity": _from_fixed(po[4]),
        "timestamp": po[5],
        "status": STATUS_NAMES.get(po[6], str(po[6])),
    }


@router.post("/verify")
def verify_integrity():
    """Run integrity verification on all on-chain records."""
    try:
        from ledger_integrity_check import verify_ledger
        results = verify_ledger()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
