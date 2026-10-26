# Blockchain-Verified, Confidence-Gated Explainable AI for Autonomous Supply Chain Replenishment
### Full Project Structure & Execution Plan (Review-2 Build)

---

## 1. System Overview

The revised pipeline has ML *govern* blockchain behavior instead of just feeding it:

```
Sales/Promo/Holiday Data
        │
        ▼
┌───────────────────────────┐
│  Tri-Model Forecast Engine │  XGBoost | Prophet | LSTM
└──────────────┬─────────────┘
               │  forecast + SHAP/decomposition explanation
               ▼
┌───────────────────────────┐
│  Confidence Estimation     │  quantile spread / prediction interval /
│  Module                    │  MC-dropout variance → confidence score
└──────────────┬─────────────┘
               │
               ▼
┌───────────────────────────┐
│  SS / ROP Inventory Calc   │  Safety Stock + Reorder Point
└──────────────┬─────────────┘
               │
               ▼
┌───────────────────────────┐
│  Decision Gate             │  confidence ≥ threshold?
└──────┬───────────────┬────┘
       │ YES            │ NO
       ▼                ▼
┌─────────────┐   ┌─────────────────────┐
│ Smart        │   │ Ledger-only log +   │
│ Contract     │   │ flagged for human   │
│ auto-executes│   │ approval            │
│ PO           │   │                     │
└──────┬───────┘   └──────────┬──────────┘
       │                      │
       ▼                      ▼
┌───────────────────────────────────┐
│  Immutable Audit Ledger            │
│  (forecast, SHAP hash, confidence, │
│   decision, action, timestamp)     │
└───────────────────────────────────┘
               │
               ▼
        FastAPI Backend + Dashboard
   (forecasts, explanations, confidence,
    pending approvals, audit trail viewer)
```

**Where ML lives (this is your "ML integration" answer for the report):**
1. Tri-model forecasting (existing)
2. SHAP/decomposition explainability (existing)
3. **Confidence estimation module (new)** — the actual bridge between ML and blockchain: it decides whether the smart contract is allowed to act autonomously

**Where blockchain lives:**
1. Smart contract that executes a PO when confidence-gated approval passes
2. Immutable ledger that logs every forecast → explanation → decision → action, regardless of gate outcome

---

## 2. Repository Structure

```
supply-chain-xai-blockchain/
│
├── data/
│   ├── raw/                       # original sales/promo/holiday datasets
│   ├── processed/                 # feature-engineered datasets
│   └── data_loader.py
│
├── models/
│   ├── xgboost_model.py
│   ├── prophet_model.py
│   ├── lstm_model.py
│   ├── train.py                   # trains + logs RMSE/MAPE for all three
│   └── saved_models/
│
├── explainability/
│   ├── shap_explainer.py          # SHAP for XGBoost
│   ├── prophet_decomposition.py   # trend/seasonality/holiday breakdown
│   └── lstm_attention.py          # attention-weight approximation
│
├── confidence/
│   ├── confidence_estimator.py    # core new module — see §3
│   └── threshold_config.yaml      # tunable confidence threshold(s)
│
├── inventory/
│   └── ss_rop_calculator.py       # Safety Stock + Reorder Point formulas
│
├── decision_gate/
│   └── gate.py                    # routes to smart contract vs. manual queue
│
├── blockchain/
│   ├── contracts/
│   │   └── ReplenishmentLedger.sol   # Solidity smart contract
│   ├── scripts/
│   │   ├── deploy.py                 # deploys contract to Ganache
│   │   └── interact.py               # web3.py calls: log(), autoExecutePO()
│   ├── ganache_config.json
│   └── abi/                          # compiled contract ABI
│
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── routers/
│   │   ├── forecast.py
│   │   ├── ledger.py
│   │   └── approvals.py           # human-review queue for low-confidence cases
│   └── schemas.py
│
├── dashboard/
│   ├── src/                       # React (or Streamlit if time-constrained)
│   │   ├── ForecastView.jsx
│   │   ├── ExplainabilityPanel.jsx
│   │   ├── ConfidenceGateStatus.jsx
│   │   └── AuditLedgerViewer.jsx
│   └── package.json
│
├── evaluation/
│   ├── model_metrics.py           # RMSE, MAPE, R²
│   ├── gate_accuracy.py           # how often confidence gating was "correct"
│   └── ledger_integrity_check.py  # tamper-detection demo for report
│
├── docs/
│   ├── architecture_diagram.png
│   ├── novelty_and_contribution.md
│   └── review2_report.docx
│
├── requirements.txt
└── README.md
```

---

## 3. Confidence Estimation Module (the key new ML piece)

This is the part reviewers will scrutinize most, since it's what makes ML and blockchain *structurally* connected rather than sequential.

| Model | Confidence Signal | Method |
|---|---|---|
| XGBoost | Prediction interval width | Quantile regression (train separate low/high quantile models) or bootstrap ensembling |
| Prophet | Native uncertainty interval | Built-in `yhat_lower` / `yhat_upper` — already available, just needs to be surfaced |
| LSTM | Predictive variance | MC-Dropout (multiple stochastic forward passes) or a small ensemble of LSTMs |

Normalize all three into a single **0–1 confidence score**, e.g.:

```
confidence = 1 - (interval_width / mean_forecast_value)
```

Then define a threshold (start with something like 0.8, tune empirically):
- `confidence ≥ threshold` → decision gate passes → smart contract auto-executes PO
- `confidence < threshold` → ledger logs the forecast + explanation + reason, item enters a manual-approval queue in the dashboard

This gives you a genuinely measurable claim for the report: *"X% of reorder decisions were confidently automated, Y% were correctly deferred to human review."*

---

## 4. Blockchain Layer — Two Build Options

**Option A — "Real" blockchain (stronger claim, more setup)**
- Ganache (local Ethereum testnet) + Solidity smart contract + `web3.py`
- Contract has two functions: `logDecision(...)` (always called) and `autoExecutePO(...)` (called only when gate passes)
- Every ledger entry is a transaction — genuinely immutable and independently verifiable, which is a strong demo moment

**Option B — Lightweight custom ledger (fallback if Ganache/Solidity setup eats too much time)**
- A simple Python hash-chained ledger (each entry stores hash of previous entry + its own data hash) backed by SQLite
- Still demonstrates tamper-evidence (`ledger_integrity_check.py` can show that altering a past entry breaks the chain)
- Weaker academically but fully workable under time pressure

**Recommendation:** build Option A first since it's the stronger novelty claim; keep Option B's hash-chain logic as a documented fallback if blockchain tooling becomes a bottleneck.

---

## 5. Execution Plan (adjust week count to your actual Review-2 date)

**Week 1 — Foundations & Confidence Module**
- Finalize/clean dataset, confirm XGBoost/Prophet/LSTM training pipeline still runs (carry over from Review-1)
- Build confidence estimation module for all three models
- Define and document the confidence threshold logic

**Week 2 — Blockchain Layer**
- Set up Ganache locally, write and deploy `ReplenishmentLedger.sol`
- Implement `web3.py` integration: `logDecision()` and `autoExecutePO()`
- Unit-test contract calls independently of the ML pipeline (mock inputs)

**Week 3 — Decision Gate + Integration**
- Build `decision_gate/gate.py` to route forecast+confidence output to either auto-execution or manual-queue logging
- Wire confidence module → SS/ROP calculator → decision gate → blockchain calls end-to-end
- Start `evaluation/gate_accuracy.py` to log how often auto-execution vs. manual review is triggered

**Week 4 — Backend + Dashboard**
- FastAPI endpoints: forecast retrieval, ledger/audit trail viewer, pending-approval queue
- Dashboard views: forecast + SHAP explanation panel, confidence score + gate status, audit ledger viewer, manual approval action
- Run a tamper-evidence demo (`ledger_integrity_check.py`) to show in the presentation

**Week 5 — Evaluation, Report, Rehearsal**
- Compute RMSE/MAPE/R² per model (as before) + new gate-accuracy metrics
- Write `novelty_and_contribution.md` framing the confidence-gated ML↔blockchain link as the core contribution
- Update title, abstract, and architecture diagram in the report
- Rehearse a live demo: show a high-confidence forecast auto-executing a PO on-chain, and a low-confidence one being deferred and logged

---

## 6. What to Emphasize in the Review-2 Report/Presentation
- **Novelty statement:** ML doesn't just feed blockchain — it *gates* whether blockchain is trusted to act autonomously, directly closing your own identified gap (disconnect between forecasts and inventory actions).
- **Explainability tie-in:** SHAP/decomposition explanations are hashed and logged alongside every decision, so the audit trail explains *why* as well as *what*.
- **Quantifiable contribution:** report the % of decisions auto-executed vs. deferred, and demonstrate ledger tamper-evidence live.
