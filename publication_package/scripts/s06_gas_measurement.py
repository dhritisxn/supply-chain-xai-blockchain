"""
s06_gas_measurement.py  (Task 5/8 support)

Re-measures gas on a private Ganache (port 8547, deterministic wallet) for
the ORIGINAL v1 contract and the hardened v2 contract, replaying the 9
decisions from docs/gate_accuracy_scan.csv through the unchanged
blockchain/scripts/interact.py bridge (log_decision for every row,
execute_po for AUTONOMOUS_EXECUTION rows).

Output: publication_package/gas_measurements.csv
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time

import pandas as pd

from common import PKG, ROOT

PORT = 8547


def _open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    exe = shutil.which("ganache")
    if exe is None:
        raise SystemExit("ganache CLI not found (npm install -g ganache)")
    from web3 import Web3
    import interact

    proc = subprocess.Popen([exe, "--port", str(PORT), "--wallet.deterministic", "--chain.chainId", "1337",
                             "--logging.quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not _open(PORT):
            if time.time() - t0 > 60:
                raise SystemExit("ganache did not start")
            time.sleep(0.5)
        w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{PORT}"))
        hardfork = "ganache default"
        scan = pd.read_csv(ROOT / "docs" / "gate_accuracy_scan.csv")
        rows = []
        for version, path in [("v1_original", ROOT / "blockchain" / "legacy" / "ReplenishmentLedger_v1.json"),
                              ("v2_hardened", ROOT / "blockchain" / "abi" / "ReplenishmentLedger.json")]:
            art = json.loads(path.read_text())
            C = w3.eth.contract(abi=art["abi"], bytecode=bytes.fromhex(art["bytecode"]))
            rc = w3.eth.wait_for_transaction_receipt(
                C.constructor().transact({"from": w3.eth.accounts[0], "gas": 6_000_000}))
            rows.append(dict(contract=version, operation="deploy", decision="", gas_used=rc.gasUsed,
                             bytecode_bytes=len(art["bytecode"]) // 2))
            addr = rc.contractAddress
            interact._load_address = lambda a=addr: a
            interact.ABI_PATH = path
            for _, r in scan.iterrows():
                rec = {"product_id": r.Product, "region": r.Region, "model_type": r.Model,
                       "forecast": float(r.Forecast), "confidence": float(r.Confidence),
                       "confidence_threshold": float(r.Threshold),
                       "suggested_order_quantity": float(r.Order_Quantity), "decision": r.Decision,
                       "confidence_metadata": {"quantiles": [0.1, 0.9]}}
                res = interact.log_decision(rec, w3=w3)
                rows.append(dict(contract=version, operation="logDecision", decision=r.Decision,
                                 series=f"{r.Product}/{r.Region}", gas_used=res["gas_used"]))
                if r.Decision == "AUTONOMOUS_EXECUTION":
                    po = interact.execute_po(res["decision_id"], w3=w3)
                    rows.append(dict(contract=version, operation="autoExecutePO", decision=r.Decision,
                                     series=f"{r.Product}/{r.Region}", gas_used=po["gas_used"]))
        df = pd.DataFrame(rows)
        df["chain"] = f"ganache {subprocess.run([exe, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1]} ({hardfork})"
        df.to_csv(PKG / "gas_measurements.csv", index=False)
        summ = df.groupby(["contract", "operation"]).gas_used.agg(["count", "mean", "min", "max"])
        print(summ.to_string())
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
