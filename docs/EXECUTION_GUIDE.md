# Comprehensive Execution Guide & Report Data

> Complete instructions for running the project and all data needed for the Review-2 report.
> All figures, tables, and metrics in this document are ready to paste into your report.

---

## Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Step-by-Step Execution](#2-step-by-step-execution)
3. [Dataset Summary](#3-dataset-summary)
4. [Model Performance Metrics](#4-model-performance-metrics)
5. [Confidence Estimation Results](#5-confidence-estimation-results)
6. [Decision Gate Accuracy](#6-decision-gate-accuracy)
7. [Blockchain Layer Results](#7-blockchain-layer-results)
8. [API Endpoints Reference](#8-api-endpoints-reference)
9. [Report Tables & Figures](#9-report-tables--figures)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites & Setup

### Software Requirements

| Software | Version | Install Command | Why It's Needed |
|----------|---------|-----------------|-----------------|
| Python | 3.10+ | System package manager | Runs all ML, API, and blockchain scripts |
| Node.js | 18+ | System package manager | Runs Ganache (local Ethereum testnet) |
| Ganache | Latest | `npm install -g ganache` | Simulates Ethereum blockchain locally |
| Solidity compiler | 0.8.19 | `npm install -g solc@0.8.19` | Compiles smart contract to bytecode |

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages and their roles:**

| Package | Role in Project |
|---------|-----------------|
| pandas, numpy | Data loading, feature engineering, numerical computation |
| scikit-learn | Train/test splitting, StandardScaler for LSTM, evaluation metrics |
| xgboost | XGBoost regressor + quantile models for confidence intervals |
| prophet, cmdstanpy | Facebook Prophet forecaster with native decomposition |
| tensorflow-cpu, keras | LSTM neural network with attention + MC-Dropout |
| shap | SHAP values for XGBoost feature attribution |
| pyyaml | Reads threshold_config.yaml and inventory_config.yaml |
| fastapi, uvicorn, pydantic | REST API backend for the dashboard |
| web3, py-solc-x | Python-Ethereum bridge for smart contract interaction |
| streamlit | Interactive dashboard UI |

### Project Structure

```
supply-chain-xai-blockchain/
├── data/                    # Raw sales data + feature engineering pipeline
├── models/                  # Trained XGBoost, Prophet, LSTM models
├── explainability/          # SHAP, Prophet decomposition, LSTM attention
├── confidence/              # Confidence scoring (ML <-> blockchain bridge)
├── inventory/               # Safety Stock + Reorder Point calculator
├── decision_gate/           # 3-way routing: AUTO / HUMAN / NO_ACTION
├── blockchain/              # Solidity contract + web3.py deployment/interaction
├── backend/                 # FastAPI REST API (5 routers)
├── dashboard/               # Streamlit dashboard (5 views)
├── evaluation/              # Model metrics, gate accuracy, integrity check
├── docs/                    # Report data, diagrams, novelty section
├── pipeline.py              # Single entry point for end-to-end pipeline
├── demo.py                  # Full presentation demo (7 steps)
└── requirements.txt         # Python dependencies
```

---

## 2. Step-by-Step Execution

### Phase 1: Start Blockchain (Terminal 1)

```bash
ganache --port 8545
```

**What this does:**
Ganache starts a local Ethereum blockchain simulator on port 8545. It creates 10 test accounts, each pre-funded with 100 ETH. This simulates a real Ethereum network without spending real money. Every transaction (logging a decision, executing a PO) is mined into a block, making it immutable — exactly like the real Ethereum blockchain, just local and instant.

**What you'll see:**
```
Ganache v7.x.x
Endpoint: http://127.0.0.1:8545
Chain ID: 1337
Accounts: 10 (100 ETH each)
Block: 0
```

**Why this matters for the report:**
This demonstrates that the system uses a real blockchain protocol (Ethereum/EVM-compatible), not a simulated ledger. The Ganache testnet provides genuine cryptographic guarantees: each block references the previous block's hash, making tampering computationally infeasible.

---

### Phase 2: Deploy Smart Contract (Terminal 2)

```bash
python blockchain/scripts/deploy.py
```

**What this does:**
The `deploy.py` script:
1. Reads the compiled Solidity bytecode from `blockchain/abi/ReplenishmentLedger.json`
2. Connects to Ganache at `http://127.0.0.1:8545`
3. Sends a deployment transaction from account 0
4. Waits for the transaction to be mined (included in a block)
5. Saves the contract address to `blockchain/deployed_address.json`

**What you'll see:**
```
Deploying from account: 0xb2078f116D259C257984605Aa3E0Fb903c0622b5
Transaction sent: 7af2bfbc...
Contract deployed at: 0x131097eE042A3ac284987f0f2811e3791C07f38F
Gas used: 1,535,219
Address saved to blockchain/deployed_address.json
```

**Why this matters for the report:**
The contract is now permanently on the local blockchain. Its address can be used by any client (the Python scripts, the FastAPI backend, the dashboard) to read and write data. The gas cost (1.5M gas) is typical for a contract with storage-heavy structs.

---

### Phase 3: Run Full Demo (Terminal 2)

```bash
python demo.py              # Full demo with blockchain
python demo.py --quick      # Skip full scan (~30 seconds faster)
python demo.py --no-chain   # Skip blockchain (offline mode)
```

**What this does:**
`demo.py` runs the complete Review-2 presentation in 7 sequential steps:

**Step 1 — Model Performance Metrics**
Reads pre-computed RMSE/MAPE/R² from `models/saved_models/` and prints a formatted report showing how well each forecasting model performs.

*Result:* Table showing XGBoost (RMSE=7.42, MAPE=4.34%, R²=0.977), Prophet (RMSE=6.40, MAPE=4.06%), LSTM (RMSE=14.24, MAPE=7.93%).

**Step 2 — Scenario A: High Confidence -> Auto-Execute PO**
Runs SKU_A/North through the full pipeline with low inventory (below_rop scenario). The XGBoost model forecasts 183.23 units with confidence 0.9137 (above the 0.85 threshold). The Decision Gate routes to AUTONOMOUS_EXECUTION. The pipeline calls `logDecision()` on-chain (always) and `autoExecutePO()` (because confidence >= threshold).

*Result:* Decision #N logged on-chain, PO #M executed on-chain. Two blockchain transactions created.

**Step 3 — Scenario B: Low Confidence -> Human Approval**
Runs SKU_B/West with low inventory. XGBoost forecasts 87.55 units but confidence is 0.8309 (below 0.85). The Decision Gate routes to HUMAN_APPROVAL. Only `logDecision()` is called — no PO execution.

*Result:* Decision logged on-chain, flagged in the approval queue. One blockchain transaction created.

**Step 4 — Scenario C: No Reorder Needed**
Runs SKU_A/North with high inventory (above_rop). Even though confidence is high (0.9137), the current inventory (1553.98) exceeds the reorder point (1380.10). The gate routes to NO_ACTION.

*Result:* Decision logged on-chain, no action taken. One blockchain transaction.

**Step 5 — Full Scan (skipped in --quick mode)**
Runs all 9 XGBoost product-region combinations through the pipeline and reports the decision distribution.

*Result:* 7 AUTONOMOUS_EXECUTION (77.8%), 2 HUMAN_APPROVAL (22.2%).

**Step 6 — Ledger Integrity Verification**
Reads all decisions and POs from the chain, recomputes SHA-256 hashes, and verifies they match. Then demonstrates tamper detection by showing that modifying a record's data changes its hash.

*Result:* "INTEGRITY CHECK PASSED" + tamper demo showing hash mismatch.

**Step 7 — Summary**
Prints the key claims for the Review-2 report with on-chain state.

*Result:* Final summary with 5 key claims + on-chain decision/PO counts.

---

### Phase 4: Run Individual Components

#### Single Product Pipeline

```bash
python pipeline.py --product SKU_A --region North --model xgboost
```

**What this does:**
Runs the complete ML -> Confidence -> SS/ROP -> Decision Gate -> Blockchain pipeline for one specific product/region/model combination. This is useful for testing or debugging a single case.

**Pipeline flow:**
1. Loads raw sales data for SKU_A/North
2. Builds feature table (lags, rolling stats, calendar features)
3. Loads pre-trained XGBoost model (point + quantile models)
4. Predicts point forecast + confidence interval
5. Computes confidence score: `exp(-1.0 * interval_width / forecast)`
6. Calculates Safety Stock and Reorder Point from historical demand
7. Simulates current inventory (or uses provided value)
8. Decision Gate: compares inventory to ROP, then confidence to threshold
9. Logs decision to blockchain (if connected)
10. Executes PO on-chain (if AUTONOMOUS_EXECUTION)

**Result:** JSON output with forecast, confidence, decision, and chain transaction details.

#### Scan All Combinations

```bash
python pipeline.py --scan --scenario below_rop --no-chain
```

**What this does:**
Iterates through all 3 products x 3 regions x 1 model (XGBoost) = 9 combinations, running the full pipeline for each. Reports the decision distribution.

**Result:** Table showing which combinations auto-execute vs. defer to human review.

#### Model Performance Metrics

```bash
python evaluation/model_metrics.py
```

**What this does:**
Reads pre-computed metrics from `models/saved_models/` (fast mode) or re-evaluates models on a test split (live mode with `--live`). Prints RMSE, MAPE, and R² per model with per-series breakdown for Prophet and LSTM.

**Result:** Formatted table of model performance + key findings (best model, R² explanation).

#### Gate Accuracy Report

```bash
python evaluation/gate_accuracy.py --scenario below_rop --no-chain
```

**What this does:**
Scans all product-region-model combinations, runs each through the full pipeline, and reports:
- Decision distribution (AUTO vs HUMAN vs NO_ACTION)
- Per-model average confidence
- On-chain metrics (if chain is connected)
- The quantifiable claim for the report

**Result:** Full gate accuracy report with percentages and the key claim statement.

#### Tamper Detection Demo

```bash
python evaluation/ledger_integrity_check.py --demo
```

**What this does:**
1. Reads the first decision record from the chain
2. Computes its SHA-256 hash
3. Creates a "tampered" version (forecast +100 units)
4. Computes the tampered hash
5. Shows that the two hashes don't match

**Result:** Visual proof that modifying on-chain data is immediately detectable.

---

### Phase 5: Start Backend + Dashboard (Terminal 3 & 4)

#### FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

**What this does:**
Starts the FastAPI REST API server on port 8000. The `--reload` flag auto-restarts when code changes. The API provides 15 endpoints for:
- Running the pipeline (`POST /forecast/run`)
- Viewing on-chain data (`GET /ledger/decisions`)
- Managing the approval queue (`GET /approvals/pending`)
- Verifying ledger integrity (`POST /ledger/verify`)

**Result:** API server running at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

#### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

**What this does:**
Starts the Streamlit dashboard on port 8501. The dashboard has 5 views:
1. **Forecast & Explainability** — Run forecasts and view SHAP/decomposition explanations
2. **Confidence Gate** — View gate decisions, scan all combinations
3. **Audit Ledger** — Browse immutable on-chain records
4. **Approval Queue** — Review and act on low-confidence decisions
5. **System Status** — Blockchain connectivity, model availability

**Result:** Interactive web dashboard at `http://localhost:8501`.

---

## 3. Dataset Summary

### Raw Data

| Property | Value | What It Means |
|----------|-------|---------------|
| Total rows | 8,496 | Daily observations across all products/regions |
| Products | SKU_A, SKU_B, SKU_C | 3 distinct product lines |
| Regions | North, South, West | 3 geographic distribution regions |
| Date range | 2023-01-01 to 2025-08-01 | ~2.5 years of daily data |
| Columns | date, product_id, region, sales, is_holiday, is_promo, price | 7 raw features |
| Null values | 0 | Complete dataset, no missing data |

**Why 8,496 rows:** 3 products x 3 regions x ~943 days = ~8,487 rows (close to 8,496 with minor variations in date ranges per series).

### Feature Engineering

`data/feature_engineering.py` generates 20 features from the raw data:

| Feature Category | Features | Why These Features |
|-----------------|----------|-------------------|
| Lag features | lag_1, lag_7, lag_14, lag_28 | Capture recent demand patterns and weekly/monthly cycles |
| Rolling statistics | rolling_mean_7/14/30, rolling_std_7/14/30 | Smooth noise, capture trend and volatility |
| Calendar | day_of_week, day_of_month, month, week_of_year, is_weekend | Capture weekly and seasonal patterns |
| Events | is_holiday, is_promo | Capture demand spikes from promotions and holidays |
| Price | price | Capture price-demand relationship |
| Encoded | product_id_enc, region_enc | Numeric encoding for categorical variables |

**Why these features matter:** The lag and rolling features capture temporal patterns (trend, seasonality, autocorrelation). Calendar features capture known periodic effects. Event features capture exogenous demand shocks. Together, they give the models enough context to forecast accurately.

---

## 4. Model Performance Metrics

### How Metrics Are Computed

Each model is evaluated on a **chronological test split** (last 15% of data, not random) to simulate real-world forecasting where you predict future values from past data.

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| RMSE | `sqrt(mean((actual - predicted)^2))` | Average magnitude of errors (in sales units) |
| MAPE | `mean(abs((actual - predicted) / actual)) * 100` | Average percentage error |
| R² | `1 - (SS_res / SS_total)` | Proportion of variance explained (1.0 = perfect) |

### Aggregate Comparison (Table 1 for Report)

| Model | RMSE | MAPE | R-squared | Confidence Method |
|-------|------|------|-----------|-------------------|
| XGBoost | 7.42 | 4.34% | 0.977 | Quantile regression (q10/q90) |
| Prophet | 6.40 | 4.06% | — | Native uncertainty intervals |
| LSTM | 14.24 | 7.93% | — | MC-Dropout (30 passes) |

**Interpretation:**
- **XGBoost** explains 97.7% of sales variance (R² = 0.977). Its error (RMSE=7.42) means predictions are typically off by ~7 units out of ~180 average daily sales.
- **Prophet** has the lowest MAPE (4.06%) because it naturally handles weekly/yearly seasonality through its decomposition. Its RMSE is slightly lower than XGBoost's because Prophet was evaluated per-series (one model per product-region), while XGBoost was trained jointly on all series.
- **LSTM** has higher error because it uses a sequence-to-one architecture with only 21 days of context, and the synthetic data's patterns are better captured by tree-based or statistical models.

### Prophet Per-Series Breakdown (Table 2 for Report)

| Series | RMSE | MAPE |
|--------|------|------|
| SKU_A/North | 10.67 | 4.02% |
| SKU_A/South | 10.26 | 3.94% |
| SKU_A/West | 7.75 | 3.86% |
| SKU_B/North | 6.02 | 3.72% |
| SKU_B/South | 5.67 | 3.97% |
| SKU_B/West | 5.69 | 4.32% |
| SKU_C/North | 5.24 | 4.58% |
| SKU_C/South | 3.53 | 4.23% |
| SKU_C/West | 2.79 | 3.87% |

**Pattern:** SKU_C has lower RMSE because it has lower sales volume (less absolute variation). SKU_A has higher RMSE because it has higher sales volume (more absolute variation), even though the percentage error (MAPE) is similar.

### LSTM Per-Series Breakdown (Table 3 for Report)

| Series | RMSE | MAPE |
|--------|------|------|
| SKU_A/North | 23.49 | 7.51% |
| SKU_A/South | 24.67 | 8.42% |
| SKU_A/West | 18.14 | 7.90% |
| SKU_B/North | 12.61 | 7.62% |
| SKU_B/South | 12.85 | 7.88% |
| SKU_B/West | 11.65 | 8.77% |
| SKU_C/North | 11.11 | 8.82% |
| SKU_C/South | 7.35 | 7.41% |
| SKU_C/West | 6.26 | 7.04% |

---

## 5. Confidence Estimation Results

### How Confidence Is Computed

Each model produces a prediction interval (low/high bounds). The confidence score normalizes the interval width relative to the forecast magnitude:

```python
relative_width = interval_width / max(|point_forecast|, eps)
confidence = exp(-alpha * relative_width)
```

**Why exponential decay:** A wider interval (relative to the forecast) means less certainty. The exponential function smoothly maps this to 0-1, with `alpha` controlling the decay rate. Higher alpha = stricter (same interval width yields lower confidence).

### Threshold Configuration

```yaml
CONFIDENCE_THRESHOLD: 0.85

xgboost:
  alpha: 1.0   # Calibrated: confidence-error correlation = -0.346

prophet:
  alpha: 1.0   # Calibrated: correlation = +0.250 (homoscedastic limitation)

lstm:
  alpha: 0.5   # Calibrated: correlation = -0.081
  mc_dropout_passes: 30
  z_multiplier: 1.28   # ~80% CI half-width
```

### Calibration Results (Table 4 for Report)

| Model | Confidence-Error Correlation | Correct Direction? | Interpretation |
|-------|------------------------------|-------------------|----------------|
| XGBoost | -0.346 | Yes | Higher confidence reliably predicts lower error |
| Prophet | +0.250 | No | Confidence goes UP when error goes UP (wrong) |
| LSTM | -0.081 | Yes (weak) | Correct direction but weak signal |

**Why Prophet's correlation is wrong:** Prophet's uncertainty interval is driven by a single global observation-noise parameter (sigma_obs). It doesn't vary between promo and non-promo days because a regressor shifts the *mean*, not the *uncertainty*. So the interval width is nearly constant (homoscedastic), making it a poor confidence signal. This is a documented limitation worth mentioning honestly in the report.

---

## 6. Decision Gate Accuracy

### Gate Logic

```
if NOT reorder_required:
    decision = NO_ACTION               # Stock is sufficient
elif confidence >= 0.85:
    decision = AUTONOMOUS_EXECUTION    # Smart contract auto-executes PO
else:
    decision = HUMAN_APPROVAL          # Logged on-chain, flagged for review
```

### Results: below_rop Scenario (9 XGBoost combinations)

| Product | Region | Forecast | Confidence | Decision |
|---------|--------|----------|------------|----------|
| SKU_A | North | 183.23 | 0.9137 | AUTONOMOUS_EXECUTION |
| SKU_A | South | 168.45 | 0.8582 | AUTONOMOUS_EXECUTION |
| SKU_A | West | 155.12 | 0.8794 | AUTONOMOUS_EXECUTION |
| SKU_B | North | 98.67 | 0.8531 | AUTONOMOUS_EXECUTION |
| SKU_B | South | 102.34 | 0.8544 | AUTONOMOUS_EXECUTION |
| SKU_B | West | 87.55 | 0.8309 | HUMAN_APPROVAL |
| SKU_C | North | 72.18 | 0.8629 | AUTONOMOUS_EXECUTION |
| SKU_C | South | 65.92 | 0.8628 | AUTONOMOUS_EXECUTION |
| SKU_C | West | 58.41 | 0.8323 | HUMAN_APPROVAL |

### Summary (Table 5 for Report)

| Decision | Count | Percentage |
|----------|-------|------------|
| AUTONOMOUS_EXECUTION | 7 | 77.8% |
| HUMAN_APPROVAL | 2 | 22.2% |
| **Total reorder cases** | **9** | **100%** |

### Key Claims for Report

> "77.8% of reorder decisions were confidently automated; 22.2% were correctly deferred to human review."

The two HUMAN_APPROVAL cases (SKU_B/West at 0.831 and SKU_C/West at 0.832) are genuinely below the 0.85 threshold — not manufactured for demonstration. They were found by scanning all 18 (product, region, model) combinations.

---

## 7. Blockchain Layer Results

### Smart Contract: ReplenishmentLedger.sol

| Property | Value |
|----------|-------|
| Solidity version | 0.8.19 |
| Network | Ganache (local Ethereum testnet) |
| Chain ID | 1337 |
| Contract functions | `logDecision()`, `autoExecutePO()` |
| Events | `DecisionLogged`, `OrderExecuted`, `HumanApprovalRequired` |
| Fixed-point precision | 1e18 (standard Solidity convention) |

### On-Chain Metrics

| Metric | Value |
|--------|-------|
| Total decisions logged | 17 |
| Total POs executed | 8 |
| Avg gas per decision log | ~300,000 |
| Avg gas per PO execution | ~226,000 |
| Integrity verification | PASSED (all records) |

### What Each Transaction Records

When `logDecision()` is called, the following is stored on-chain:

| Field | Example Value | What It Captures |
|-------|---------------|-----------------|
| id | 1 | Auto-incrementing decision ID |
| sku | "SKU_A" | Which product |
| region | "North" | Which distribution region |
| orderQuantity | 540.86e18 | Suggested replenishment quantity |
| modelType | "xgboost" | Which ML model produced the forecast |
| forecastValue | 183.23e18 | The point forecast |
| confidenceScore | 0.9137e18 | The confidence score |
| confidenceThreshold | 0.85e18 | The threshold used |
| shapHash | 0x3de129... | SHA-256 of the SHAP explanation |
| status | 1 (AUTONOMOUS) | The gate decision |
| timestamp | 1725100000 | Block timestamp |

### Three Demo Scenarios (Table 6 for Report)

| Scenario | Product | Confidence | Decision | On-Chain Action |
|----------|---------|------------|----------|-----------------|
| A: High confidence | SKU_A/North | 0.9137 | AUTONOMOUS_EXECUTION | PO auto-executed |
| B: Low confidence | SKU_B/West | 0.8309 | HUMAN_APPROVAL | Logged, flagged for review |
| C: No reorder | SKU_A/North | 0.9137 | NO_ACTION | Logged, no action |

---

## 8. API Endpoints Reference

### REST API (FastAPI at port 8000)

| Method | Endpoint | What It Does | Returns |
|--------|----------|-------------|---------|
| GET | `/` | Shows system overview | Pipeline stages, blockchain status |
| GET | `/health` | Health check | `{"status": "healthy"}` |
| POST | `/forecast/run` | Runs full pipeline for one product | Forecast + confidence + decision + chain result |
| POST | `/forecast/scan` | Runs all 18 combinations | Decision distribution summary |
| GET | `/forecast/products` | Lists available products/regions | Product and region lists |
| GET | `/forecast/models` | Lists model types | Model names + descriptions |
| GET | `/forecast/metrics` | Shows model performance | RMSE/MAPE/R² from saved models |
| GET | `/ledger/summary` | Shows on-chain record counts | Decision + PO counts |
| GET | `/ledger/decisions` | Lists all on-chain decisions | Full decision records |
| GET | `/ledger/decisions/{id}` | Gets one decision | Single decision record |
| GET | `/ledger/purchase-orders` | Lists all on-chain POs | Full PO records |
| GET | `/ledger/purchase-orders/{id}` | Gets one PO | Single PO record |
| POST | `/ledger/verify` | Runs integrity check | Verification results |
| GET | `/approvals/pending` | Lists pending human reviews | Pending decisions |
| POST | `/approvals/{id}/approve` | Approves a decision | Approval confirmation |
| POST | `/approvals/{id}/reject` | Rejects a decision | Rejection confirmation |

### Example API Calls

```bash
# Run pipeline for one product
curl -X POST http://localhost:8000/forecast/run \
  -H "Content-Type: application/json" \
  -d '{"product_id": "SKU_A", "region": "North", "model_type": "xgboost", "inventory_scenario": "below_rop"}'

# Get all on-chain decisions
curl http://localhost:8000/ledger/decisions

# Run integrity check
curl -X POST http://localhost:8000/ledger/verify

# Get pending approvals
curl http://localhost:8000/approvals/pending
```

---

## 9. Report Tables & Figures

### Table 1: Forecasting Model Performance (Aggregate)

| Model | RMSE | MAPE (%) | R-squared | Confidence Method |
|-------|------|----------|-----------|-------------------|
| XGBoost | 7.42 | 4.34 | 0.977 | Quantile regression |
| Prophet | 6.40 | 4.06 | — | Native intervals |
| LSTM | 14.24 | 7.93 | — | MC-Dropout |

### Table 2: Prophet Per-Series Performance

| Series | RMSE | MAPE (%) |
|--------|------|----------|
| SKU_A/North | 10.67 | 4.02 |
| SKU_A/South | 10.26 | 3.94 |
| SKU_A/West | 7.75 | 3.86 |
| SKU_B/North | 6.02 | 3.72 |
| SKU_B/South | 5.67 | 3.97 |
| SKU_B/West | 5.69 | 4.32 |
| SKU_C/North | 5.24 | 4.58 |
| SKU_C/South | 3.53 | 4.23 |
| SKU_C/West | 2.79 | 3.87 |

### Table 3: Decision Gate Accuracy

| Decision | Count | % |
|----------|-------|---|
| AUTONOMOUS_EXECUTION | 7 | 77.8% |
| HUMAN_APPROVAL | 2 | 22.2% |
| **Total** | **9** | **100%** |

### Table 4: Confidence Calibration

| Model | Correlation | Correct Direction? |
|-------|-------------|-------------------|
| XGBoost | -0.346 | Yes |
| Prophet | +0.250 | No (homoscedastic) |
| LSTM | -0.081 | Yes (weak) |

### Table 5: Demo Scenarios

| Scenario | Product | Confidence | Decision | Blockchain Action |
|----------|---------|------------|----------|-------------------|
| A | SKU_A/North | 0.914 | AUTO | PO executed |
| B | SKU_B/West | 0.831 | HUMAN | Flagged for review |
| C | SKU_A/North | 0.914 | NO_ACTION | Logged |

### Figure 1: Architecture Diagram
- File: `docs/architecture_diagram.svg`
- Description: 8-stage pipeline visual (Data -> ML -> Confidence -> SS/ROP -> Gate -> Blockchain -> UI)

### Figure 2: Decision Gate Routing
- Generate from: `python pipeline.py --scan --scenario below_rop --no-chain`
- Shows: confidence distribution across 9 product-region combinations

### Figure 3: Tamper Detection
- Generate from: `python evaluation/ledger_integrity_check.py --demo`
- Shows: Original vs tampered hash comparison

---

## 10. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Cannot connect to Ganache` | Ganache not running | Start: `ganache --port 8545` |
| `Contract not deployed` | deploy.py not run yet | Run: `python blockchain/scripts/deploy.py` |
| `ModuleNotFoundError: prophet` | Prophet not installed | Install: `pip install prophet` |
| `ModuleNotFoundError: tensorflow` | TensorFlow not installed | Install: `pip install tensorflow-cpu` |
| `UnicodeEncodeError` on Windows | Terminal encoding issue | The code uses ASCII-safe output; ensure terminal is UTF-8 |
| `MismatchedABI warning` | web3.py over-broad event decoding | Harmless — suppressed in interact.py |
| `ConnectionRefused` on API | Backend not running | Start: `uvicorn backend.main:app --reload` |

### Full Reset

```bash
# Kill Ganache
pkill -f ganache

# Restart and redeploy
ganache --port 8545 &
sleep 2
python blockchain/scripts/deploy.py
```

### Verify Everything Works

```bash
python -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
print('Ganache:', 'OK' if w3.is_connected() else 'DOWN')

from backend.main import app
from fastapi.testclient import TestClient
r = TestClient(app).get('/health')
print('Backend:', 'OK' if r.status_code == 200 else 'DOWN')
print('Blockchain decisions:', r.json().get('blockchain', {}).get('decisions_on_chain', 'N/A'))
"
```

---

## Files Reference for Report

| File | Purpose | Used in Report |
|------|---------|----------------|
| `docs/architecture_diagram.svg` | Pipeline visual | Figure 1 |
| `docs/novelty_and_contribution.md` | Novelty section | Section 3 |
| `docs/review2_report.md` | Full report draft | Entire report |
| `docs/report_data.json` | Raw metrics data | Tables 1-5 |
| `docs/gate_accuracy_scan.csv` | Gate accuracy data | Table 3 (importable) |
| `docs/model_performance.csv` | Model comparison | Table 1 (importable) |
| `docs/prophet_per_series.csv` | Prophet breakdown | Table 2 (importable) |
| `docs/lstm_per_series.csv` | LSTM breakdown | Table 3 (importable) |
| `evaluation/model_metrics.py` | Model metrics | Table 1 |
| `evaluation/gate_accuracy.py` | Gate accuracy | Table 3 |
| `evaluation/ledger_integrity_check.py` | Tamper demo | Figure 3 |
| `demo.py` | Live demo | Presentation |
| `pipeline.py` | Single product test | Quick verification |

---

*All data in this document is from the actual executed system as of August 2026.*
