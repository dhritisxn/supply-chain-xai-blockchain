"""
compile.py

Compiles ReplenishmentLedger.sol into blockchain/abi/ReplenishmentLedger.json
({"abi": [...], "bytecode": "<hex, no 0x>"}), the artefact deploy.py and
interact.py read.

Settings are pinned to the ones that reproduce the original v1 artefact
(blockchain/legacy/ReplenishmentLedger_v1.json): solc 0.8.19, legacy
codegen, optimizer enabled, runs=200. With these settings the v1 source
compiles to identical executable bytecode; the trailing CBOR metadata hash
additionally matches byte-for-byte only if the source is decoded as cp1252,
which is evidently how the original build read the file (its comments
contain non-ASCII arrows). This script reads UTF-8.

Usage:
    python blockchain/scripts/compile.py                       # current contract
    python blockchain/scripts/compile.py --source X.sol --out Y.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = PROJECT_ROOT / "blockchain" / "contracts" / "ReplenishmentLedger.sol"
OUT = PROJECT_ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json"

UNIT_NAME = "ReplenishmentLedger.sol"
SOLC_VERSION = "0.8.19"
SETTINGS = {
    "optimizer": {"enabled": True, "runs": 200},
    "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}},
}


def compile_contract(source: Path = SOURCE, contract: str = "ReplenishmentLedger") -> dict:
    import solcx
    if SOLC_VERSION not in [str(v) for v in solcx.get_installed_solc_versions()]:
        solcx.install_solc(SOLC_VERSION)
    out = solcx.compile_standard(
        {"language": "Solidity",
         # fixed unit name: it is hashed into the bytecode metadata
         "sources": {UNIT_NAME: {"content": source.read_text(encoding="utf-8")}},
         "settings": SETTINGS},
        solc_version=SOLC_VERSION,
    )
    c = out["contracts"][UNIT_NAME][contract]
    return {"abi": c["abi"], "bytecode": c["evm"]["bytecode"]["object"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    artifact = compile_contract(args.source)
    args.out.write_text(json.dumps(artifact), encoding="utf-8")
    print(f"Compiled {args.source.name} -> {args.out} ({len(artifact['bytecode']) // 2} bytes)")
