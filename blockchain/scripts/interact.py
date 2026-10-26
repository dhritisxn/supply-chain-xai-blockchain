"""
interact.py

Python ↔ web3.py bridge that connects the Decision Gate (gate.py) output
to the on-chain ReplenishmentLedger contract.

Two main entry points:
  - log_decision(record)      → calls contract.logDecision() (always)
  - execute_po(decision_id)   → calls contract.autoExecutePO() (only when gate passes)

Both functions accept a gate.py DecisionRecord (or a dict with the same keys)
and return the transaction receipt.

Usage (standalone demo):
    python blockchain/scripts/interact.py          # runs Scenario A/B/C demos

Usage (from gate.py pipeline):
    from blockchain.scripts.interact import log_decision, execute_po
    receipt = log_decision(decision_record)
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Union

from web3 import Web3

# Suppress harmless MismatchedABI warnings from process_receipt when
# multiple events are emitted in one transaction (DecisionLogged +
# HumanApprovalRequired). The actual event decoding works correctly.
warnings.filterwarnings("ignore", category=UserWarning, module="eth_utils")

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ABI_PATH = PROJECT_ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json"
CONFIG_PATH = PROJECT_ROOT / "blockchain" / "ganache_config.json"
ADDRESS_PATH = PROJECT_ROOT / "blockchain" / "deployed_address.json"

# Fixed-point precision: 1e18 = 1.0 (standard Solidity convention)
FIXED_POINT = 10**18


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_fixed(val: float) -> int:
    """Convert a Python float to Solidity uint256 fixed-point (1e18)."""
    return int(round(val * FIXED_POINT))


def _shap_hash(metadata: dict) -> bytes:
    """
    Produce a bytes32 SHA-256 hash of the SHAP explanation payload.
    If metadata is empty, returns a zero hash.
    """
    if not metadata:
        return b"\x00" * 32
    payload = json.dumps(metadata, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).digest()


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _load_address() -> str:
    with open(ADDRESS_PATH) as f:
        return json.load(f)["address"]


def _get_web3() -> Web3:
    cfg = _load_config()
    rpc_url = cfg.get("rpc_url", "http://127.0.0.1:8545")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(
            f"Cannot connect to Ganache at {rpc_url}. "
            "Start Ganache first:  ganache --port 8545"
        )
    return w3


def _load_contract(w3: Web3):
    """Return a Web3 contract instance for the deployed ReplenishmentLedger."""
    with open(ABI_PATH) as f:
        abi = json.load(f)["abi"]
    address = _load_address()
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)


# ---------------------------------------------------------------------------
#  Core interface
# ---------------------------------------------------------------------------

# Decision status mapping (matches gate.py GateDecision → Solidity enum)
_STATUS_MAP = {
    "NO_ACTION": 0,
    "AUTONOMOUS_EXECUTION": 1,
    "HUMAN_APPROVAL": 2,
}


def log_decision(record, w3: Web3 | None = None) -> dict:
    """
    Log a DecisionRecord to the on-chain ledger.  Always called regardless
    of gate outcome — this is the immutable audit trail.

    Parameters
    ----------
    record : gate.py DecisionRecord or dict with matching keys
    w3     : optional Web3 instance (created if None)

    Returns
    -------
    dict with keys: decision_id, tx_hash, gas_used, status
    """
    if w3 is None:
        w3 = _get_web3()
    contract = _load_contract(w3)

    # Accept both DecisionRecord objects and plain dicts
    d = record.to_dict() if hasattr(record, "to_dict") else record

    status_int = _STATUS_MAP.get(d["decision"])
    if status_int is None:
        raise ValueError(f"Unknown decision status: {d['decision']}")

    shap_hash = _shap_hash(d.get("confidence_metadata", {}))

    sender = w3.eth.accounts[0]

    tx = contract.functions.logDecision(
        d["product_id"],                    # sku
        d["region"],                        # region
        _to_fixed(d["suggested_order_quantity"]),  # orderQuantity
        d["model_type"],                    # modelType
        _to_fixed(d["forecast"]),           # forecastValue
        _to_fixed(d["confidence"]),         # confidenceScore
        _to_fixed(d["confidence_threshold"]),  # confidenceThreshold
        shap_hash,                          # shapHash
        status_int,                         # status
    ).transact({"from": sender, "gas": 2_000_000})

    receipt = w3.eth.wait_for_transaction_receipt(tx, timeout=60)

    # Decode the DecisionLogged event to get the on-chain decision ID
    logs = contract.events.DecisionLogged().process_receipt(receipt)
    decision_id = logs[0]["args"]["decisionId"] if logs else None

    result = {
        "decision_id": decision_id,
        "tx_hash": receipt.transactionHash.hex(),
        "gas_used": receipt.gasUsed,
        "status": d["decision"],
    }
    print(f"[log_decision] ID={decision_id}  status={d['decision']}  "
          f"tx={result['tx_hash'][:16]}...")
    return result


def execute_po(decision_id: int, w3: Web3 | None = None) -> dict:
    """
    Auto-execute a Purchase Order on-chain for an AUTONOMOUS_EXECUTION decision.

    Parameters
    ----------
    decision_id : the on-chain ID returned by log_decision()
    w3          : optional Web3 instance

    Returns
    -------
    dict with keys: po_id, tx_hash, gas_used
    """
    if w3 is None:
        w3 = _get_web3()
    contract = _load_contract(w3)
    sender = w3.eth.accounts[0]

    tx = contract.functions.autoExecutePO(decision_id).transact({
        "from": sender,
        "gas": 2_000_000,
    })
    receipt = w3.eth.wait_for_transaction_receipt(tx, timeout=60)

    logs = contract.events.OrderExecuted().process_receipt(receipt)
    po_id = logs[0]["args"]["poId"] if logs else None

    result = {
        "po_id": po_id,
        "tx_hash": receipt.transactionHash.hex(),
        "gas_used": receipt.gasUsed,
    }
    print(f"[execute_po] PO_ID={po_id}  decision={decision_id}  "
          f"tx={result['tx_hash'][:16]}...")
    return result


def get_decision(decision_id: int, w3: Web3 | None = None) -> dict:
    """Read a DecisionRecord back from the chain."""
    if w3 is None:
        w3 = _get_web3()
    contract = _load_contract(w3)
    return contract.functions.getDecision(decision_id).call()


def get_po(po_id: int, w3: Web3 | None = None) -> dict:
    """Read a PurchaseOrder back from the chain."""
    if w3 is None:
        w3 = _get_web3()
    contract = _load_contract(w3)
    return contract.functions.getPO(po_id).call()


# ---------------------------------------------------------------------------
#  Demo / standalone runner
# ---------------------------------------------------------------------------

def _run_demo():
    """
    Replay the three Decision Gate scenarios against the live Ganache chain.
    Requires: ganache running + contract deployed (run deploy.py first).
    """
    # Import gate logic (resolve paths from project root)
    sys.path.insert(0, str(PROJECT_ROOT / "decision_gate"))
    sys.path.insert(0, str(PROJECT_ROOT / "confidence"))
    sys.path.insert(0, str(PROJECT_ROOT / "inventory"))
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    sys.path.insert(0, str(PROJECT_ROOT / "explainability"))
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

    from gate import evaluate_series, GateDecision

    scenarios = [
        ("Scenario A: Reorder + High Confidence -> AUTO", "SKU_A", "North", "xgboost", "below_rop"),
        ("Scenario B: Reorder + Low Confidence  -> HUMAN", "SKU_B", "West",  "xgboost", "below_rop"),
        ("Scenario C: No Reorder                -> NO_ACTION", "SKU_A", "North", "xgboost", "above_rop"),
    ]

    w3 = _get_web3()

    for label, sku, region, model, inv_scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        record = evaluate_series(sku, region, model_type=model, inventory_scenario=inv_scenario)
        d = record.to_dict()
        print(f"  Product:     {d['product_id']}/{d['region']}")
        print(f"  Forecast:    {d['forecast']:.2f}")
        print(f"  Confidence:  {d['confidence']:.4f}  (threshold={d['confidence_threshold']})")
        print(f"  Decision:    {d['decision']}")

        # Always log to the ledger
        result = log_decision(record, w3=w3)

        # If autonomous, also execute the PO
        if record.decision == GateDecision.AUTONOMOUS_EXECUTION:
            po_result = execute_po(result["decision_id"], w3=w3)
            print(f"  => PO executed on-chain (PO_ID={po_result['po_id']})")
        elif record.decision == GateDecision.HUMAN_APPROVAL:
            print(f"  => Logged for human review (decision_id={result['decision_id']})")
        else:
            print(f"  => No action logged")

    # Summary
    contract = _load_contract(w3)
    n_decisions = contract.functions.getDecisionCount().call()
    n_pos = contract.functions.getPOCount().call()
    print(f"\n{'='*60}")
    print(f"  LEDGER SUMMARY: {n_decisions} decisions, {n_pos} POs on-chain")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        _run_demo()
    except ConnectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Start Ganache first:  ganache --port 8545", file=sys.stderr)
        sys.exit(1)
