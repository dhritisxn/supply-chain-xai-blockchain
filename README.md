# Blockchain-Verified, Confidence-Gated Explainable AI for Autonomous Supply Chain Replenishment

> ML doesn't just feed blockchain — it **gates** whether blockchain is trusted to act autonomously.

---

## System Overview

A complete pipeline that combines tri-model forecasting, explainability, confidence scoring, and blockchain auditing into an autonomous supply chain replenishment system:

```
Data (Sales/Promo/Holiday)
        |
        v
+---------------------------+
|  Tri-Model Forecast Engine |  XGBoost | Prophet | LSTM
+---------------------------+
        |  forecast + SHAP/decomposition explanation
        v
+---------------------------+
|  Confidence Estimation     |  quantile spread / prediction interval /
|  Module                    |  MC-dropout variance -> confidence score
+---------------------------+
        |
        v
+---------------------------+
|  SS / ROP Inventory Calc   |  Safety Stock + Reorder Point
+---------------------------+
        |
        v
+---------------------------+
|  Decision Gate             |  confidence >= 0.85?
+-----------+-------+-------+
            |       |
      YES   |       |  NO
            v       v
+----------------+  +------------------+
| Smart Contract  |  | Ledger log +     |
| auto-executes   |  | flagged for      |
| PO              |  | human approval   |
+----------------+  +------------------+
            |               |
            v               v
+-----------------------------------+
|  Immutable Audit Ledger            |
|  (forecast, SHAP hash, confidence, |
|   decision, action, timestamp)     |
+-----------------------------------+
            |
            v
     FastAPI Backend + Dashboard
```

## Project Structure

```
supply-chain-xai-blockchain/
|
+-- data/                          # Data layer
|   +-- raw/                       # Synthetic sales/promo/holiday datasets
|   +-- processed/                 # Feature-engineered datasets
|   +-- data_loader.py             # Data loading and preprocessing
|   +-- feature_engineering.py     # Lags, rolling stats, calendar features
|
+-- models/                        # Tri-model forecasting
|   +-- xgboost_model.py           # XGBoost + quantile regression (q10/q90)
|   +-- prophet_model.py           # Prophet with native decomposition
|   +-- lstm_model.py              # LSTM + additive attention
|   +-- train.py                   # Unified training orchestrator
|   +-- saved_models/              # Trained models + metrics (43 files)
|
+-- explainability/                # XAI layer
|   +-- shap_explainer.py          # SHAP for XGBoost
|   +-- prophet_decomposition.py   # Trend/seasonality/holiday breakdown
|   +-- lstm_attention.py          # Attention-weight approximation
|
+-- confidence/                    # Confidence estimation (ML <-> blockchain bridge)
|   +-- confidence_estimator.py    # Unified 0-1 score across all models
|   +-- calibrate.py               # Validates confidence tracks real error
|   +-- threshold_config.yaml      # Tunable threshold (0.85) + per-model alpha
|
+-- inventory/                     # Inventory control
|   +-- ss_rop_calculator.py       # Safety Stock + Reorder Point formulas
|   +-- inventory_config.yaml      # Service level, lead time config
|
+-- decision_gate/                 # Routing logic
|   +-- gate.py                    # 3-way decision: AUTO / HUMAN / NO_ACTION
|
+-- blockchain/                    # On-chain audit ledger
|   +-- contracts/
|   |   +-- ReplenishmentLedger.sol   # Solidity smart contract
|   +-- scripts/
|   |   +-- deploy.py                 # Deploy to Ganache
|   |   +-- interact.py               # web3.py bridge (log + execute PO)
|   +-- abi/                          # Compiled ABI + bytecode
|   +-- ganache_config.json
|   +-- deployed_address.json
|
+-- backend/                       # FastAPI REST API
|   +-- main.py                    # App entry point
|   +-- schemas.py                 # Pydantic models
|   +-- routers/
|       +-- forecast.py            # Forecast + pipeline endpoints
|       +-- ledger.py              # On-chain record viewer
|       +-- approvals.py           # Human approval queue
|
+-- dashboard/                     # Streamlit UI
|   +-- app.py                     # 5 views: Forecast, Confidence, Ledger, Approvals, Status
|
+-- evaluation/                    # Metrics + integrity
|   +-- model_metrics.py           # RMSE/MAPE/R2 per model
|   +-- gate_accuracy.py           # Decision gate scan + on-chain metrics
|   +-- ledger_integrity_check.py  # Tamper detection demo
|
+-- docs/                          # Documentation
|   +-- architecture_diagram.svg   # Pipeline visual
|   +-- novelty_and_contribution.md # Review-2 novelty section
|
+-- pipeline.py                    # End-to-end pipeline (single entry point)
+-- demo.py                        # Full presentation demo script
+-- requirements.txt               # Python dependencies
+-- README.md
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Node.js (for Ganache)
npm install -g ganache solc@0.8.19
```

### 1. Start Blockchain

```bash
# Terminal 1: Start local Ethereum testnet
ganache --port 8545

# Terminal 2: Deploy the smart contract
python blockchain/scripts/deploy.py
```

### 2. Run the Full Demo

```bash
# Complete Review-2 presentation demo (all 7 steps)
python demo.py

# Quick mode (skip full scan)
python demo.py --quick

# Offline mode (no blockchain)
python demo.py --no-chain
```

### 3. Run Individual Components

```bash
# Single product through full pipeline
python pipeline.py --product SKU_A --region North --model xgboost

# Scan all 18 combinations
python pipeline.py --scan --scenario below_rop

# Model performance metrics
python evaluation/model_metrics.py

# Gate accuracy report
python evaluation/gate_accuracy.py --scenario below_rop

# Tamper detection demo
python evaluation/ledger_integrity_check.py --demo
```

### 4. Start Backend + Dashboard

```bash
# Terminal 1: FastAPI server
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Streamlit dashboard
streamlit run dashboard/app.py
```

API docs available at: http://localhost:8000/docs

## Decision Gate Logic

```
NOT reorder_required                                    -> NO_ACTION
reorder_required AND confidence >= 0.85                 -> AUTONOMOUS_EXECUTION
reorder_required AND confidence <  0.85                 -> HUMAN_APPROVAL
```

- **AUTONOMOUS_EXECUTION**: Smart contract auto-executes the Purchase Order on-chain
- **HUMAN_APPROVAL**: Logged on-chain, flagged in dashboard for human review
- **NO_ACTION**: Stock sufficient, no reorder needed

## Model Performance

| Model | RMSE | MAPE | R-squared | Confidence Method |
|-------|------|------|-----------|-------------------|
| XGBoost | 7.42 | 4.34% | 0.977 | Quantile regression (q10/q90) |
| Prophet | 6.40 | 4.06% | — | Native uncertainty intervals |
| LSTM | 14.24 | 7.93% | — | MC-Dropout (30 passes) |

**Key claim**: XGBoost explains 97.7% of sales variance. Prophet achieves the best MAPE (4.06%). All three models are used for diversity; confidence scoring handles the variance.

## Gate Accuracy (18 combinations, below_rop scenario)

| Decision | Count | % |
|----------|-------|---|
| AUTONOMOUS_EXECUTION | 7 | 77.8% |
| HUMAN_APPROVAL | 2 | 22.2% |
| NO_ACTION | 0 | 0% (excluded by scenario) |

**Key claim**: "77.8% of reorder decisions were confidently automated; 22.2% were correctly deferred to human review."

## Blockchain Layer

- **Smart Contract**: `ReplenishmentLedger.sol` on Ganache (local Ethereum testnet)
- **DecisionRecord**: SKU, region, model, forecast, confidence, SHAP hash, status, timestamp
- **PurchaseOrder**: Linked to originating decision, auto-executed when confidence gate passes
- **Events**: `DecisionLogged`, `OrderExecuted`, `HumanApprovalRequired`
- **Integrity**: SHA-256 hash verification, tamper detection demo included

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System overview + blockchain status |
| GET | `/health` | Health check |
| POST | `/forecast/run` | Full pipeline for one product |
| POST | `/forecast/scan` | All 18 combinations |
| GET | `/forecast/products` | List products and regions |
| GET | `/ledger/decisions` | All on-chain decisions |
| GET | `/ledger/purchase-orders` | All on-chain POs |
| POST | `/ledger/verify` | Integrity check |
| GET | `/approvals/pending` | Pending human approvals |
| POST | `/approvals/{id}/approve` | Approve decision |
| POST | `/approvals/{id}/reject` | Reject decision |

## Dashboard Views

1. **Forecast & Explainability** — Run forecasts with SHAP/prophet decomposition explanations
2. **Confidence Gate** — View gate decisions, scan all combinations, confidence distribution
3. **Audit Ledger** — Browse immutable on-chain decisions and purchase orders
4. **Approval Queue** — Review and act on low-confidence decisions
5. **System Status** — Blockchain connectivity, model availability, configuration

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Forecasting | XGBoost, Prophet, LSTM (TensorFlow/Keras) |
| Explainability | SHAP, Prophet decomposition, LSTM attention |
| Confidence | Quantile regression, native intervals, MC-Dropout |
| Blockchain | Solidity, Ganache, web3.py |
| Backend | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit |
| Evaluation | scikit-learn, numpy, pandas |

## License

Academic project — Review-2 submission.
