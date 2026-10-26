# Gas cost in real-world terms (Task 8)

Inputs: `gas_measurements.csv` (re-measured by `scripts/s06_gas_measurement.py` on Ganache
v7.9.2, replaying the 9 decisions of `docs/gate_accuracy_scan.csv` through the unchanged
`interact.py`). Arithmetic: `scripts/s07_gas_cost_table.py` → `gas_cost_table.csv`,
`gas_cost_scale.csv`, `gas_cost_l2_scenarios.csv`.

`USD = gas_used × gas_price_gwei × 1e-9 × ETH_USD`

**All prices are assumptions, not market data.** ETH/USD values of $2,500 / $3,500 /
$4,500 bracket the range of price snapshots found during this work (no dated October-2026
quote could be verified). Gas prices of 20 / 50 / 100 gwei are the requested scenarios;
0.5 and 5 gwei are added because ethereum.org reported mainnet standard gas around
0.15 gwei with daily averages near 0.5 gwei in April–May 2026 (ethereum.org, "Building on
Ethereum in 2026"). Re-check both inputs on the date the paper is finalised.

## Measured gas (mean over the 9 replayed decisions; 7 POs)
| operation | v1 original | v2 hardened | Δ |
|---|---:|---:|---:|
| `logDecision` | 311,007 (min 308,221, max 325,345) | 314,946 | +3,939 (+1.3 %) |
| `autoExecutePO` | 228,768 (min 226,325, max 243,425) | 255,610 | +26,842 (+11.7 %) |
| deployment (one-off) | 1,495,969 | 1,962,276 | +466,307 |
| expected per decision = log + 0.7186 × PO | 475,393 | 498,620 | +4.9 % |

The v1 measurements reproduce the repo's "~300K per log / ~226K per PO" figures (the
first write of each run costs more, because it initialises fresh storage slots). 0.7186 is
the XGBoost gate's autonomous rate at threshold 0.85 on the test set
(`gate_baseline_comparison.csv`).

## Ethereum mainnet, v2 contract (USD)
| operation | gas | 20 gwei @ $3,500 | 50 gwei @ $3,500 | 100 gwei @ $3,500 | 0.5 gwei @ $3,500 |
|---|---:|---:|---:|---:|---:|
| `logDecision` | 314,946 | $22.05 | $55.12 | $110.23 | $0.55 |
| `autoExecutePO` | 255,610 | $17.89 | $44.73 | $89.46 | $0.45 |
| expected per decision | 498,620 | $34.90 | $87.26 | $174.52 | $0.87 |
| deployment (one-off) | 1,962,276 | $137.36 | $343.40 | $686.80 | $3.43 |

Range across ETH $2,500–$4,500 for the expected per-decision cost: $24.93–$44.88 at
20 gwei, $62.33–$112.19 at 50 gwei, $124.66–$224.38 at 100 gwei (`gas_cost_table.csv`,
which also has the v1 rows).

## At scale (v2, expected per decision, ETH $3,500, USD / year)
| scale | decisions / yr | 0.5 gwei | 5 gwei | 20 gwei | 50 gwei | 100 gwei |
|---|---:|---:|---:|---:|---:|---:|
| 9 series × 1/day (this study) | 3,285 | $2,866 | $28,664 | $114,658 | $286,644 | $573,289 |
| 1,000 SKU-locations × 1/day | 365,000 | $318,494 | $3,184,937 | $12,739,750 | $31,849,374 | $63,698,749 |
| 10,000 SKU-locations × 1/day | 3,650,000 | $3,184,937 | $31,849,374 | $127,397,497 | $318,493,743 | $636,987,487 |

## Discussion: L2 / permissioned deployment
At the requested 20–100 gwei, mainnet costs $35–$175 per decision, which already
dominates the value of a small replenishment order. A retailer running 10,000
SKU-locations daily would pay $127M–$637M a year, so per-decision mainnet logging is not
viable at those prices. Even at the sub-gwei fees seen in 2026 the cost is $0.87 per
decision ($3.2M/yr at 10,000 SKU-locations). Because the contract is plain EVM bytecode,
it can be deployed unchanged to an EVM rollup or sidechain. With an *assumed* effective
all-in price of 0.01–0.1 gwei on an Arbitrum-style rollup (ETH $3,500), a decision costs
$0.017–$0.17, or $64k–$637k/yr at 10,000 SKU-locations. On a Polygon-PoS-style chain at an
*assumed* 50–200 gwei priced in POL at $0.25, it costs $0.006–$0.025 per decision, or
$23k–$91k/yr (`gas_cost_l2_scenarios.csv`). On a permissioned EVM chain (Hyperledger
Besu / GoQuorum with zero gas price) the marginal monetary cost is zero and the cost
becomes node operation. That chain also trades public verifiability for a consortium
trust model, which weakens the "independently verifiable audit trail" argument.
Separately from the chain choice, most of `logDecision`'s cost comes from writing a full
struct plus three strings (SKU, region, model type) to contract storage. Storing only a
32-byte commitment to the decision context and emitting the details in events should be
much cheaper, but that alternative was not implemented or measured here, so no saving is
claimed. The L2
figures are scenario arithmetic, not measurements, and those chains' fee levels and token
prices should be checked when the paper is written.
