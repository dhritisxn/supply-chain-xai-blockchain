"""
main.py

FastAPI application for the Blockchain-Verified, Confidence-Gated
Explainable AI Supply Chain Replenishment system.

Endpoints:
  GET  /                          - System overview and status
  GET  /health                    - Health check
  POST /forecast/run              - Run full pipeline for one product
  POST /forecast/scan             - Run all product/model combos
  GET  /forecast/products         - List products and regions
  GET  /forecast/models           - List model types
  GET  /forecast/metrics          - Model performance metrics
  GET  /ledger/summary            - On-chain record counts
  GET  /ledger/decisions          - All on-chain decisions
  GET  /ledger/decisions/{id}     - Specific decision
  GET  /ledger/purchase-orders    - All on-chain POs
  GET  /ledger/purchase-orders/{id} - Specific PO
  POST /ledger/verify             - Run integrity check
  GET  /approvals/pending         - Pending human approvals
  POST /approvals/{id}/approve    - Approve and execute PO
  POST /approvals/{id}/reject     - Reject decision

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from backend.routers import forecast, ledger, approvals

app = FastAPI(
    title="Supply Chain XAI Blockchain API",
    description=(
        "Blockchain-Verified, Confidence-Gated Explainable AI for "
        "Autonomous Supply Chain Replenishment"
    ),
    version="2.0.0",
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router)
app.include_router(ledger.router)
app.include_router(approvals.router)


@app.get("/")
def root():
    """System overview and pipeline status."""
    # Check blockchain connectivity
    chain_status = "disconnected"
    n_decisions = 0
    n_pos = 0
    try:
        from blockchain.scripts.interact import _get_web3, _load_contract
        w3 = _get_web3()
        contract = _load_contract(w3)
        n_decisions = contract.functions.getDecisionCount().call()
        n_pos = contract.functions.getPOCount().call()
        chain_status = "connected"
    except Exception:
        pass

    return {
        "system": "Blockchain-Verified, Confidence-Gated XAI Supply Chain",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "stages": [
                "Tri-Model Forecasting (XGBoost, Prophet, LSTM)",
                "Confidence Estimation (quantile/MC-dropout/prophet interval)",
                "SS/ROP Inventory Calculator",
                "Decision Gate (confidence-gated 3-way routing)",
                "Blockchain Audit Ledger (immutable on-chain records)",
            ],
            "blockchain": {
                "status": chain_status,
                "decisions_on_chain": n_decisions,
                "purchase_orders_on_chain": n_pos,
            },
        },
        "endpoints": {
            "forecast": "/forecast",
            "ledger": "/ledger",
            "approvals": "/approvals",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    chain_ok = False
    try:
        from blockchain.scripts.interact import _get_web3
        w3 = _get_web3()
        chain_ok = w3.is_connected()
    except Exception:
        pass

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blockchain": "connected" if chain_ok else "disconnected",
    }
