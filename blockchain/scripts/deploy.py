"""
deploy.py

Deploys the ReplenishmentLedger smart contract to a local Ganache instance.

Usage:
    1. Start Ganache:  ganache --port 8545
    2. Run:            python blockchain/scripts/deploy.py

The script reads the compiled ABI + bytecode from blockchain/abi/ and
deploys to the Ganache RPC endpoint configured in ganache_config.json.
On success it prints the contract address and saves it for interact.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from web3 import Web3

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ABI_PATH = PROJECT_ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json"
CONFIG_PATH = PROJECT_ROOT / "blockchain" / "ganache_config.json"
ADDRESS_PATH = PROJECT_ROOT / "blockchain" / "deployed_address.json"


# ---------------------------------------------------------------------------
#  Load compiled artefact
# ---------------------------------------------------------------------------

def load_artifact():
    with open(ABI_PATH) as f:
        artifact = json.load(f)
    return artifact["abi"], bytes.fromhex(artifact["bytecode"])


# ---------------------------------------------------------------------------
#  Connect + deploy
# ---------------------------------------------------------------------------

def get_web3() -> Web3:
    """Read the RPC URL from ganache_config.json and return a connected Web3."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    rpc_url = cfg.get("rpc_url", "http://127.0.0.1:8545")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(
            f"Cannot connect to Ganache at {rpc_url}. "
            "Start Ganache first:  ganache --port 8545"
        )
    return w3


def deploy(w3: Web3 | None = None) -> str:
    """
    Deploy ReplenishmentLedger and return the hex contract address.
    Also persists the address to deployed_address.json for interact.py.
    """
    if w3 is None:
        w3 = get_web3()

    abi, bytecode = load_artifact()

    deployer = w3.eth.accounts[0]
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    print(f"Deploying from account: {deployer}")

    tx_hash = Contract.constructor().transact({
        "from": deployer,
        "gas": 6_000_000,
    })
    print(f"Transaction sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    contract_address = receipt.contractAddress

    print(f"Contract deployed at: {contract_address}")
    print(f"Gas used: {receipt.gasUsed}")

    # Persist for interact.py
    address_data = {
        "address": contract_address,
        "deployer": deployer,
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
    }
    with open(ADDRESS_PATH, "w") as f:
        json.dump(address_data, f, indent=2)
    print(f"Address saved to {ADDRESS_PATH}")

    return contract_address


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        deploy()
    except ConnectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
