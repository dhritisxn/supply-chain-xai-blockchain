"""
Solidity-side tests for ReplenishmentLedger (web3.py against a private
Ganache, see conftest.py). Covers logDecision(), autoExecutePO(), the v2
access-control modifier, the one-PO-per-decision rule, the on-chain
confidence/threshold consistency checks, and a non-regression check that
the existing interact.py bridge still works unchanged against v2.
"""

from __future__ import annotations

import pytest
from web3.exceptions import ContractLogicError, Web3RPCError

# Ganache (vmErrorsOnRPCResponse) reports reverts as an RPC error carrying the
# revert reason; other nodes raise ContractLogicError. Either means "reverted".
REVERT = (ContractLogicError, Web3RPCError)

FP = 10**18
NO_ACTION, AUTO, HUMAN = 0, 1, 2
SHAP = b"\x11" * 32


def _log(c, sender, status=AUTO, conf=0.91, thr=0.85, qty=442.83):
    return c.functions.logDecision(
        "SKU_A", "North", int(qty * FP), "xgboost", int(183.23 * FP),
        int(round(conf * FP)), int(round(thr * FP)), SHAP, status,
    ).transact({"from": sender, "gas": 2_000_000})


def _did(c, w3, tx):
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    return c.events.DecisionLogged().process_receipt(rcpt)[0]["args"]["decisionId"], rcpt


# ---------------------------------------------------------------- deployment

def test_deployer_is_owner_and_authorized(ledger, w3):
    owner = w3.eth.accounts[0]
    assert ledger.functions.owner().call() == owner
    assert ledger.functions.authorizedCallers(owner).call() is True
    assert ledger.functions.authorizedCallers(w3.eth.accounts[1]).call() is False
    assert ledger.functions.minAutonomousThreshold().call() == 85 * 10**16


# ---------------------------------------------------------------- logDecision

def test_log_decision_stores_full_record(ledger, w3):
    did, _ = _did(ledger, w3, _log(ledger, w3.eth.accounts[0]))
    rec = ledger.functions.getDecision(did).call()
    assert rec[0] == did == 1
    assert tuple(rec[1:3]) == ("SKU_A", "North")
    assert rec[3] == int(442.83 * FP)
    assert rec[4] == "xgboost"
    assert rec[6] == int(round(0.91 * FP))
    assert rec[8] == SHAP
    assert rec[9] == AUTO
    assert ledger.functions.getDecisionCount().call() == 1
    assert list(ledger.functions.getAllDecisionIds().call()) == [1]


def test_human_approval_emits_secondary_event(ledger, w3):
    _, rcpt = _did(ledger, w3, _log(ledger, w3.eth.accounts[0], status=HUMAN, conf=0.83))
    ev = ledger.events.HumanApprovalRequired().process_receipt(rcpt)
    assert len(ev) == 1 and ev[0]["args"]["confidenceScore"] == int(round(0.83 * FP))


def test_no_action_and_auto_do_not_emit_human_event(ledger, w3):
    for status in (NO_ACTION, AUTO):
        _, rcpt = _did(ledger, w3, _log(ledger, w3.eth.accounts[0], status=status))
        assert ledger.events.HumanApprovalRequired().process_receipt(rcpt) == ()


def test_log_decision_rejects_unauthorized_caller(ledger, w3):
    with pytest.raises(REVERT, match="Caller is not authorized"):
        _log(ledger, w3.eth.accounts[1])
    assert ledger.functions.getDecisionCount().call() == 0


def test_autonomous_requires_confidence_at_or_above_threshold(ledger, w3):
    with pytest.raises(REVERT, match="Confidence below threshold"):
        _log(ledger, w3.eth.accounts[0], status=AUTO, conf=0.84, thr=0.85)
    # boundary: confidence == threshold is allowed (matches gate.py's >=)
    _did(ledger, w3, _log(ledger, w3.eth.accounts[0], status=AUTO, conf=0.85, thr=0.85))
    # a low-confidence record is fine when it is NOT marked autonomous
    _did(ledger, w3, _log(ledger, w3.eth.accounts[0], status=HUMAN, conf=0.10, thr=0.85))


def test_autonomous_threshold_cannot_undercut_onchain_floor(ledger, w3):
    with pytest.raises(REVERT, match="Threshold below on-chain minimum"):
        _log(ledger, w3.eth.accounts[0], status=AUTO, conf=0.60, thr=0.50)


def test_confidence_above_one_rejected(ledger, w3):
    with pytest.raises(REVERT, match="Confidence above 1.0"):
        _log(ledger, w3.eth.accounts[0], status=HUMAN, conf=1.5)


# ---------------------------------------------------------------- autoExecutePO

def test_auto_execute_po_creates_linked_po(ledger, w3):
    owner = w3.eth.accounts[0]
    did, _ = _did(ledger, w3, _log(ledger, owner))
    rcpt = w3.eth.wait_for_transaction_receipt(
        ledger.functions.autoExecutePO(did).transact({"from": owner, "gas": 2_000_000}))
    ev = ledger.events.OrderExecuted().process_receipt(rcpt)[0]["args"]
    po = ledger.functions.getPO(ev["poId"]).call()
    assert po[1] == did and tuple(po[2:4]) == ("SKU_A", "North") and po[4] == int(442.83 * FP) and po[6] == AUTO
    assert ledger.functions.poForDecision(did).call() == ev["poId"] == 1
    assert ledger.functions.getPOCount().call() == 1


def test_auto_execute_po_rejects_unauthorized_caller(ledger, w3):
    did, _ = _did(ledger, w3, _log(ledger, w3.eth.accounts[0]))
    with pytest.raises(REVERT, match="Caller is not authorized"):
        ledger.functions.autoExecutePO(did).transact({"from": w3.eth.accounts[1], "gas": 2_000_000})
    assert ledger.functions.getPOCount().call() == 0


def test_auto_execute_po_rejects_double_execution(ledger, w3):
    owner = w3.eth.accounts[0]
    did, _ = _did(ledger, w3, _log(ledger, owner))
    ledger.functions.autoExecutePO(did).transact({"from": owner, "gas": 2_000_000})
    with pytest.raises(REVERT, match="PO already executed for decision"):
        ledger.functions.autoExecutePO(did).transact({"from": owner, "gas": 2_000_000})
    assert ledger.functions.getPOCount().call() == 1


@pytest.mark.parametrize("status", [NO_ACTION, HUMAN])
def test_auto_execute_po_rejects_non_autonomous_decisions(ledger, w3, status):
    did, _ = _did(ledger, w3, _log(ledger, w3.eth.accounts[0], status=status, conf=0.5))
    with pytest.raises(REVERT, match="not marked AUTONOMOUS_EXECUTION"):
        ledger.functions.autoExecutePO(did).transact({"from": w3.eth.accounts[0], "gas": 2_000_000})


def test_auto_execute_po_rejects_missing_decision_and_zero_quantity(ledger, w3):
    owner = w3.eth.accounts[0]
    with pytest.raises(REVERT, match="Decision does not exist"):
        ledger.functions.autoExecutePO(999).transact({"from": owner, "gas": 2_000_000})
    did, _ = _did(ledger, w3, _log(ledger, owner, qty=0))
    with pytest.raises(REVERT, match="Order quantity is zero"):
        ledger.functions.autoExecutePO(did).transact({"from": owner, "gas": 2_000_000})


def test_reentrancy_lock_is_released_after_each_call(ledger, w3):
    """autoExecutePO makes no external calls, so re-entry cannot be provoked
    from a test; this checks the guard does not wedge the contract."""
    owner = w3.eth.accounts[0]
    for _ in range(3):
        did, _ = _did(ledger, w3, _log(ledger, owner))
        ledger.functions.autoExecutePO(did).transact({"from": owner, "gas": 2_000_000})
    assert ledger.functions.getPOCount().call() == 3


# ---------------------------------------------------------------- administration

def test_owner_can_grant_and_revoke_callers(ledger, w3):
    owner, backend = w3.eth.accounts[0], w3.eth.accounts[2]
    ledger.functions.setAuthorizedCaller(backend, True).transact({"from": owner})
    did, _ = _did(ledger, w3, _log(ledger, backend))
    ledger.functions.autoExecutePO(did).transact({"from": backend, "gas": 2_000_000})
    ledger.functions.setAuthorizedCaller(backend, False).transact({"from": owner})
    with pytest.raises(REVERT, match="Caller is not authorized"):
        _log(ledger, backend)


def test_non_owner_cannot_administer(ledger, w3):
    attacker = w3.eth.accounts[3]
    with pytest.raises(REVERT, match="Caller is not the owner"):
        ledger.functions.setAuthorizedCaller(attacker, True).transact({"from": attacker})
    with pytest.raises(REVERT, match="Caller is not the owner"):
        ledger.functions.setMinAutonomousThreshold(0).transact({"from": attacker})
    with pytest.raises(REVERT, match="Caller is not the owner"):
        ledger.functions.transferOwnership(attacker).transact({"from": attacker})


def test_threshold_floor_admin(ledger, w3):
    owner = w3.eth.accounts[0]
    with pytest.raises(REVERT, match="Threshold above 1.0"):
        ledger.functions.setMinAutonomousThreshold(FP + 1).transact({"from": owner})
    ledger.functions.setMinAutonomousThreshold(90 * 10**16).transact({"from": owner})
    with pytest.raises(REVERT, match="Threshold below on-chain minimum"):
        _log(ledger, owner, status=AUTO, conf=0.95, thr=0.85)


def test_transfer_ownership(ledger, w3):
    a0, a1 = w3.eth.accounts[0], w3.eth.accounts[1]
    ledger.functions.transferOwnership(a1).transact({"from": a0})
    assert ledger.functions.owner().call() == a1
    with pytest.raises(REVERT, match="Caller is not the owner"):
        ledger.functions.setAuthorizedCaller(a0, False).transact({"from": a0})


# ---------------------------------------------------------------- before/after evidence

def test_v1_contract_had_no_access_control_or_double_spend_protection(ledger_v1, w3):
    """Documents the pre-hardening vulnerability: any account could write a
    fabricated AUTONOMOUS record with confidence below its own threshold and
    execute the same decision's PO repeatedly."""
    attacker = w3.eth.accounts[5]
    did, _ = _did(ledger_v1, w3, _log(ledger_v1, attacker, status=AUTO, conf=0.10, thr=0.0))
    for _ in range(2):
        ledger_v1.functions.autoExecutePO(did).transact({"from": attacker, "gas": 2_000_000})
    assert ledger_v1.functions.getPOCount().call() == 2


# ---------------------------------------------------------------- non-regression of the Python bridge

def test_interact_bridge_works_unchanged_against_v2(ledger, w3, monkeypatch):
    import interact
    monkeypatch.setattr(interact, "_load_address", lambda: ledger.address)
    record = {
        "product_id": "SKU_B", "region": "West", "model_type": "xgboost",
        "forecast": 87.55, "confidence": 0.9137, "confidence_threshold": 0.85,
        "suggested_order_quantity": 406.37, "decision": "AUTONOMOUS_EXECUTION",
        "confidence_metadata": {"quantiles": [0.1, 0.9]},
    }
    res = interact.log_decision(record, w3=w3)
    po = interact.execute_po(res["decision_id"], w3=w3)
    assert po["po_id"] == 1
    human = dict(record, decision="HUMAN_APPROVAL", confidence=0.8309)
    assert interact.log_decision(human, w3=w3)["decision_id"] == 2


def test_zero_address_rejected_in_admin(ledger, w3):
    zero = "0x" + "00" * 20
    with pytest.raises(REVERT, match="Zero address"):
        ledger.functions.setAuthorizedCaller(zero, True).transact({"from": w3.eth.accounts[0]})
    with pytest.raises(REVERT, match="Zero address"):
        ledger.functions.transferOwnership(zero).transact({"from": w3.eth.accounts[0]})


def test_getters_revert_for_missing_ids(ledger):
    with pytest.raises(REVERT, match="Decision does not exist"):
        ledger.functions.getDecision(1).call()
    with pytest.raises(REVERT, match="Purchase order does not exist"):
        ledger.functions.getPO(1).call()
