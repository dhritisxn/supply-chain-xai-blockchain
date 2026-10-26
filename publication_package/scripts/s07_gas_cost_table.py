"""
s07_gas_cost_table.py  (Task 8)

Pure arithmetic: converts measured gas (gas_measurements.csv) into USD at
assumed gas prices and ETH prices, per decision and per year at scale.
All prices are ASSUMPTIONS (inputs below), not market data.

    cost_USD = gas_used * gas_price_gwei * 1e-9 * ETH_USD

Output: publication_package/gas_cost_table.csv, gas_cost_scale.csv
"""

from __future__ import annotations

import pandas as pd

from common import PKG

GAS_PRICES_GWEI = [0.5, 5, 20, 50, 100]      # 20/50/100 as requested; 0.5/5 = recent low-fee regime
ETH_USD = [2500, 3500, 4500]
AUTO_RATE = 0.718573                          # XGBoost gate @0.85 on the test set (gate_baseline_comparison.csv)
SCALES = {"9 series x 1/day (this study)": 9 * 365,
          "1,000 SKU-locations x 1/day": 1_000 * 365,
          "10,000 SKU-locations x 1/day": 10_000 * 365}


def main():
    g = pd.read_csv(PKG / "gas_measurements.csv")
    mean = g.groupby(["contract", "operation"]).gas_used.mean()
    rows = []
    for contract in ["v1_original", "v2_hardened"]:
        log, po = mean[(contract, "logDecision")], mean[(contract, "autoExecutePO")]
        per_decision = log + AUTO_RATE * po
        for op, gas in [("logDecision", log), ("autoExecutePO", po),
                        ("expected per decision (log + 0.7186*PO)", per_decision),
                        ("deploy (one-off)", mean[(contract, "deploy")])]:
            for gp in GAS_PRICES_GWEI:
                r = dict(contract=contract, operation=op, gas=round(gas), gas_price_gwei=gp)
                for eth in ETH_USD:
                    r[f"usd_at_eth_{eth}"] = gas * gp * 1e-9 * eth
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(PKG / "gas_cost_table.csv", index=False)

    sc = []
    per = t[(t.contract == "v2_hardened") & (t.operation.str.startswith("expected"))]
    for label, n in SCALES.items():
        for _, r in per.iterrows():
            sc.append(dict(scale=label, decisions_per_year=n, gas_price_gwei=r.gas_price_gwei,
                           usd_per_year_at_eth_3500=n * r.usd_at_eth_3500))
    pd.DataFrame(sc).to_csv(PKG / "gas_cost_scale.csv", index=False)

    # L2 / sidechain scenarios: ASSUMED effective all-in gas prices and native-token prices.
    gas = mean[("v2_hardened", "logDecision")] + AUTO_RATE * mean[("v2_hardened", "autoExecutePO")]
    l2 = []
    for name, gp, px in [("Arbitrum-style rollup, 0.01 gwei effective, ETH $3,500", 0.01, 3500),
                         ("Arbitrum-style rollup, 0.1 gwei effective, ETH $3,500", 0.1, 3500),
                         ("Polygon PoS-style, 50 gwei in POL, POL $0.25", 50, 0.25),
                         ("Polygon PoS-style, 200 gwei in POL, POL $0.25", 200, 0.25),
                         ("Permissioned EVM (Besu/Quorum), gas price 0", 0, 0)]:
        per = gas * gp * 1e-9 * px
        l2.append({"scenario (assumed prices)": name, "gas_per_decision": round(gas), "usd_per_decision": per,
                   **{f"usd_per_year_{k}": per * n for k, n in
                      zip(["9_series", "1k_sku_loc", "10k_sku_loc"], SCALES.values())}})
    pd.DataFrame(l2).to_csv(PKG / "gas_cost_l2_scenarios.csv", index=False)
    print(t.to_string(index=False))
    print(pd.DataFrame(sc).to_string(index=False))


if __name__ == "__main__":
    main()
