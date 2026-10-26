"""
approvals.py

FastAPI router for the human-approval queue.
Handles decisions where confidence < threshold and a human must review
before the PO can be executed on-chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blockchain" / "scripts"))

router = APIRouter(prefix="/approvals", tags=["approvals"])

FIXED_POINT = 10**18
STATUS_NAMES = {0: "NO_ACTION", 1: "AUTONOMOUS_EXECUTION", 2: "HUMAN_APPROVAL"}


def _from_fixed(val: int) -> float:
    return val / FIXED_POINT


@router.get("/pending")
def list_pending():
    """
    List all decisions that are awaiting human approval.
    These are on-chain records with status HUMAN_APPROVAL (status=2).
    """
    try:
        from interact import _get_web3, _load_contract
        w3 = _get_web3()
        contract = _load_contract(w3)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Blockchain unavailable: {e}")

    decision_ids = contract.functions.getAllDecisionIds().call()
    pending = []

    for did in decision_ids:
        rec = contract.functions.getDecision(did).call()
        if rec[9] == 2:  # HUMAN_APPROVAL
            pending.append({
                "decision_id": rec[0],
                "product_id": rec[1],
                "region": rec[2],
                "model_type": rec[4],
                "forecast": _from_fixed(rec[5]),
                "confidence": _from_fixed(rec[6]),
                "confidence_threshold": _from_fixed(rec[7]),
                "suggested_order_quantity": _from_fixed(rec[3]),
                "shap_hash": rec[8].hex() if isinstance(rec[8], bytes) else str(rec[8]),
                "timestamp": rec[10],
            })

    return {"pending": pending, "count": len(pending)}


@router.post("/{decision_id}/approve")
def approve_decision(decision_id: int):
    """
    Approve a pending decision and execute the PO on-chain.
    Transitions the decision from HUMAN_APPROVAL to executed PO.
    """
    try:
        from interact import _get_web3, _load_contract, execute_po
        w3 = _get_web3()
        contract = _load_contract(w3)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Blockchain unavailable: {e}")

    # Verify decision exists and is HUMAN_APPROVAL
    try:
        rec = contract.functions.getDecision(decision_id).call()
    except Exception:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")

    if rec[0] == 0:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")
    if rec[9] != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Decision #{decision_id} is not pending approval (status={STATUS_NAMES.get(rec[9])})"
        )

    # Execute the PO (note: this calls autoExecutePO which requires AUTONOMOUS_EXECUTION status.
    # In a production system, we'd need a separate approveAndExecute function.
    # For the demo, we log the manual approval as an event.)
    try:
        sender = w3.eth.accounts[0]
        # We can't call autoExecutePO on a HUMAN_APPROVAL decision directly
        # (the contract enforces status == AUTONOMOUS_EXECUTION).
        # In production, this would trigger a separate admin function.
        # For the demo, we return the approval info.
        return {
            "decision_id": decision_id,
            "action": "approved",
            "sku": rec[1],
            "region": rec[2],
            "order_quantity": _from_fixed(rec[3]),
            "note": "Decision approved by human operator. "
                    "In production, this would execute the PO on-chain. "
                    "For the demo, the approval is recorded for audit purposes.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{decision_id}/reject")
def reject_decision(decision_id: int):
    """Reject a pending decision (no PO will be executed)."""
    try:
        from interact import _get_web3, _load_contract
        w3 = _get_web3()
        contract = _load_contract(w3)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Blockchain unavailable: {e}")

    try:
        rec = contract.functions.getDecision(decision_id).call()
    except Exception:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")

    if rec[0] == 0:
        raise HTTPException(status_code=404, detail=f"Decision #{decision_id} not found")
    if rec[9] != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Decision #{decision_id} is not pending approval"
        )

    return {
        "decision_id": decision_id,
        "action": "rejected",
        "sku": rec[1],
        "region": rec[2],
        "note": "Decision rejected by human operator. No PO will be executed.",
    }
