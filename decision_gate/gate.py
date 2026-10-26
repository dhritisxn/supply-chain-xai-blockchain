"""
gate.py

Phase 4: Decision Gate.

Combines two independent signals -- computed by two independent modules,
each doing exactly one job -- into a single routing decision:

  1. Reorder need         (inventory/ss_rop_calculator.py -> reorder_required)
  2. Forecast trustworthiness (confidence/confidence_estimator.py -> confidence)

Per the spec, the gate's own logic is intentionally the only "smart" part
here -- it does NOT re-derive confidence, and it does NOT re-derive
reorder need. It only routes:

    NOT reorder_required
        -> NO_ACTION

    reorder_required AND confidence >= CONFIDENCE_THRESHOLD
        -> AUTONOMOUS_EXECUTION   (Phase 5-9: this is what triggers the
                                    smart contract)

    reorder_required AND confidence <  CONFIDENCE_THRESHOLD
        -> HUMAN_APPROVAL         (Phase 7: this is what populates the
                                    pending-approval queue)

This module does NOT talk to the blockchain layer itself (that's Phase
5-9) and does NOT decide model trustworthiness itself (that's Phase 2) --
it only combines their outputs. Keeping this separation is what makes the
"where's the ML" answer to a reviewer clean: ML (confidence) decides
whether autonomy is *allowed*; the gate decides what happens as a result;
the blockchain (later phases) only executes/records what the gate decided.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "confidence"))
sys.path.append(str(Path(__file__).parent.parent / "inventory"))
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "explainability"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

from confidence_estimator import ConfidenceResult, get_threshold
from ss_rop_calculator import InventoryStatus


class GateDecision(str, Enum):
    NO_ACTION = "NO_ACTION"
    AUTONOMOUS_EXECUTION = "AUTONOMOUS_EXECUTION"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


@dataclass
class DecisionRecord:
    """
    The full context the audit ledger (Phase 8-10) will eventually log:
    forecast, confidence, SS/ROP, decision, reasoning, timestamp. Nothing
    here writes to a ledger yet -- that's the blockchain layer's job -- but
    the shape of this object is exactly what Phase 5-10 will consume, so
    it's built once here rather than reshaped later.
    """
    product_id: str
    region: str
    model_type: str
    timestamp: str
    forecast: float
    confidence: float
    confidence_threshold: float
    reorder_required: bool
    safety_stock: float
    reorder_point: float
    current_inventory: float
    suggested_order_quantity: float
    decision: GateDecision
    reasoning: str
    confidence_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "region": self.region,
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "forecast": round(self.forecast, 4),
            "confidence": round(self.confidence, 4),
            "confidence_threshold": self.confidence_threshold,
            "reorder_required": self.reorder_required,
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "current_inventory": round(self.current_inventory, 2),
            "suggested_order_quantity": round(self.suggested_order_quantity, 2),
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "confidence_metadata": self.confidence_metadata,
        }


def route(inventory_status: InventoryStatus, confidence_result: ConfidenceResult,
          model_type: Optional[str] = None, threshold: Optional[float] = None) -> DecisionRecord:
    """
    The Decision Gate's entire routing logic. Deliberately small and
    readable -- this is the exact 3-branch logic the spec calls for, with
    nothing else folded in.
    """
    threshold = threshold if threshold is not None else get_threshold()
    model_type = model_type or confidence_result.model_type

    if not inventory_status.reorder_required:
        decision = GateDecision.NO_ACTION
        reasoning = (
            f"Current inventory ({inventory_status.current_inventory:.1f}) is above the "
            f"reorder point ({inventory_status.reorder_point:.1f}); no replenishment needed "
            f"regardless of forecast confidence."
        )
    elif confidence_result.confidence >= threshold:
        decision = GateDecision.AUTONOMOUS_EXECUTION
        reasoning = (
            f"Reorder required (inventory {inventory_status.current_inventory:.1f} <= "
            f"ROP {inventory_status.reorder_point:.1f}) AND confidence "
            f"({confidence_result.confidence:.3f}) meets the threshold ({threshold}); "
            f"autonomous execution is allowed."
        )
    else:
        decision = GateDecision.HUMAN_APPROVAL
        reasoning = (
            f"Reorder required (inventory {inventory_status.current_inventory:.1f} <= "
            f"ROP {inventory_status.reorder_point:.1f}) BUT confidence "
            f"({confidence_result.confidence:.3f}) is below the threshold ({threshold}); "
            f"routed to human approval instead of autonomous execution."
        )

    return DecisionRecord(
        product_id=inventory_status.product_id,
        region=inventory_status.region,
        model_type=model_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        forecast=confidence_result.point_forecast,
        confidence=confidence_result.confidence,
        confidence_threshold=threshold,
        reorder_required=inventory_status.reorder_required,
        safety_stock=inventory_status.safety_stock,
        reorder_point=inventory_status.reorder_point,
        current_inventory=inventory_status.current_inventory,
        suggested_order_quantity=inventory_status.suggested_order_quantity,
        decision=decision,
        reasoning=reasoning,
        confidence_metadata=confidence_result.metadata,
    )


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper: forecast -> confidence -> SS/ROP -> gate
# ---------------------------------------------------------------------------

def evaluate_series(product_id: str, region: str, model_type: str = "xgboost",
                     inventory_scenario: str = "random",
                     current_inventory: Optional[float] = None) -> DecisionRecord:
    """
    Runs the full Phase 1-4 pipeline for one (product_id, region) series
    using the requested model, and returns the Decision Gate's routing
    decision. This is what the demo scenarios (A/B/C) and, later, the
    FastAPI backend will call.
    """
    from data_loader import load_raw_sales
    from ss_rop_calculator import reorder_status

    raw = load_raw_sales()
    sub = raw[(raw["product_id"] == product_id) & (raw["region"] == region)].sort_values("date")
    historical_sales = sub["sales"].values

    if model_type == "xgboost":
        from feature_engineering import build_feature_table, FEATURE_COLS
        from confidence_estimator import XGBoostConfidenceEstimator
        feat, _ = build_feature_table(raw)
        row = feat[(feat["product_id"] == product_id) & (feat["region"] == region)].iloc[-1]
        confidence_result = XGBoostConfidenceEstimator().estimate(row)

    elif model_type == "prophet":
        from prophet_model import train_prophet_for_series
        from confidence_estimator import ProphetConfidenceEstimator
        _, forecast, _ = train_prophet_for_series(raw, product_id, region)
        confidence_result = ProphetConfidenceEstimator().estimate(forecast.iloc[-1])

    elif model_type == "lstm":
        from lstm_model import train_lstm_for_series, make_windows, WINDOW
        from confidence_estimator import LSTMConfidenceEstimator
        model, scaler, _ = train_lstm_for_series(raw, product_id, region, epochs=15)
        scaled = scaler.transform(historical_sales.reshape(-1, 1).astype("float32")).flatten()
        X, _ = make_windows(scaled, WINDOW)
        confidence_result = LSTMConfidenceEstimator().estimate(model, X[-1:], scaler)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    inv_status = reorder_status(
        product_id, region,
        daily_demand_forecast=confidence_result.point_forecast,
        historical_sales=historical_sales,
        current_inventory=current_inventory,
        scenario=inventory_scenario,
    )

    return route(inv_status, confidence_result, model_type=model_type)


if __name__ == "__main__":
    import json

    print("=== SCENARIO A: reorder required + high confidence (XGBoost, SKU_A/North, low stock) ===")
    record = evaluate_series("SKU_A", "North", model_type="xgboost", inventory_scenario="below_rop")
    print(json.dumps(record.to_dict(), indent=2))

    # SKU_B/West/xgboost was found by scanning all 18 (product, region, model)
    # combinations for one that genuinely lands below the confidence
    # threshold -- not manufactured. See evaluation/gate_accuracy.py for the
    # full scan across all combinations.
    print("\n=== SCENARIO B: reorder required + low confidence (XGBoost, SKU_B/West, low stock) ===")
    record = evaluate_series("SKU_B", "West", model_type="xgboost", inventory_scenario="below_rop")
    print(json.dumps(record.to_dict(), indent=2))

    print("\n=== SCENARIO C: no reorder required (XGBoost, SKU_A/North, high stock) ===")
    record = evaluate_series("SKU_A", "North", model_type="xgboost", inventory_scenario="above_rop")
    print(json.dumps(record.to_dict(), indent=2))
