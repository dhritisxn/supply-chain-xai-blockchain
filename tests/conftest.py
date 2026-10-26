"""
Shared pytest fixtures.

Python unit tests need nothing beyond requirements.txt. The Solidity tests
(tests/test_contract.py) start a private, throw-away Ganache instance on
port 8546 (so they never touch a developer's long-running :8545 chain) and
are skipped automatically if the `ganache` CLI is not installed
(`npm install -g ganache`).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ["models", "data", "confidence", "inventory", "decision_gate",
            "explainability", "blockchain/scripts"]:
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

GANACHE_PORT = 8546


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def w3():
    exe = shutil.which("ganache")
    if exe is None:
        pytest.skip("ganache CLI not installed (npm install -g ganache)")
    from web3 import Web3

    proc = None
    if not _port_open(GANACHE_PORT):
        proc = subprocess.Popen(
            [exe, "--port", str(GANACHE_PORT), "--wallet.deterministic",
             "--chain.chainId", "1337", "--chain.vmErrorsOnRPCResponse", "true",
             "--logging.quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 60
        while not _port_open(GANACHE_PORT):
            if time.time() > deadline:
                proc.kill()
                pytest.fail("ganache did not start within 60s")
            time.sleep(0.5)
    w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{GANACHE_PORT}"))
    assert w3.is_connected()
    yield w3
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _deploy(w3, artifact_path: Path):
    art = json.loads(artifact_path.read_text())
    C = w3.eth.contract(abi=art["abi"], bytecode=bytes.fromhex(art["bytecode"]))
    tx = C.constructor().transact({"from": w3.eth.accounts[0], "gas": 6_000_000})
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    return w3.eth.contract(address=rcpt.contractAddress, abi=art["abi"]), rcpt


@pytest.fixture
def ledger(w3):
    """A freshly deployed v2 (hardened) ReplenishmentLedger."""
    c, _ = _deploy(w3, ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json")
    return c


@pytest.fixture
def ledger_v1(w3):
    """The original, unhardened contract (kept for before/after evidence)."""
    c, _ = _deploy(w3, ROOT / "blockchain" / "legacy" / "ReplenishmentLedger_v1.json")
    return c
