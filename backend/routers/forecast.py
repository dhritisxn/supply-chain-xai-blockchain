"""
forecast.py

FastAPI router for forecast retrieval and pipeline execution.
Provides endpoints to:
  - Run a single product through the full pipeline
  - Get model metrics and confidence scores
  - Run a full scan across all product/region/model combos
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/products")
def list_products():
    """List all available products and regions."""
    from data_loader import load_raw_sales
    raw = load_raw_sales()
    products = sorted(raw["product_id"].unique().tolist())
    regions = sorted(raw["region"].unique().tolist())
    return {"products": products, "regions": regions}


@router.get("/models")
def list_models():
    """List available model types."""
    return {
        "models": ["xgboost", "prophet", "lstm"],
        "description": {
            "xgboost": "Gradient boosted trees with quantile regression for confidence",
            "prophet": "Facebook Prophet with native uncertainty intervals",
            "lstm": "LSTM with MC-Dropout for predictive variance",
        },
    }


@router.get("/metrics")
def get_model_metrics():
    """Retrieve saved model performance metrics (RMSE, MAPE, R2)."""
    metrics_path = PROJECT_ROOT / "models" / "saved_models" / "prophet" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    import json
    with open(metrics_path) as f:
        return json.load(f)


@router.post("/run")
def run_pipeline(request: dict):
    """
    Run the full ML -> Confidence -> SS/ROP -> Decision Gate -> Blockchain
    pipeline for a single product/region/model combination.
    """
    from pipeline import run_pipeline as _run_pipeline

    try:
        result = _run_pipeline(
            product_id=request["product_id"],
            region=request["region"],
            model_type=request.get("model_type", "xgboost"),
            inventory_scenario=request.get("inventory_scenario", "below_rop"),
            current_inventory=request.get("current_inventory"),
            log_to_chain=request.get("log_to_chain", True),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dr = result["decision_record"].to_dict()
    response = {
        "decision_record": dr,
        "chain_result": result["chain_result"],
        "po_result": result["po_result"],
    }
    return response


@router.post("/scan")
def run_scan(request: dict):
    """Run all product/region/model combinations and return summary."""
    from pipeline import run_full_scan

    try:
        results = run_full_scan(
            inventory_scenario=request.get("scenario", "below_rop"),
            log_to_chain=request.get("log_to_chain", True),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    scan_results = []
    for r in results:
        dr = r["decision_record"]
        scan_results.append({
            "product_id": dr.product_id,
            "region": dr.region,
            "model_type": dr.model_type,
            "confidence": round(dr.confidence, 4),
            "decision": dr.decision.value,
            "chain_decision_id": r["chain_result"]["decision_id"] if r["chain_result"] else None,
            "chain_tx_hash": r["chain_result"]["tx_hash"] if r["chain_result"] else None,
        })

    from collections import Counter
    decisions = Counter(r["decision"] for r in scan_results)

    return {
        "total": len(scan_results),
        "no_action": decisions.get("NO_ACTION", 0),
        "autonomous": decisions.get("AUTONOMOUS_EXECUTION", 0),
        "human_approval": decisions.get("HUMAN_APPROVAL", 0),
        "results": scan_results,
    }
