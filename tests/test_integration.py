"""Integration tests: the three README demo scenarios end-to-end through
gate.evaluate_series() with the saved XGBoost models (no chain), plus the
Prophet/LSTM paths (which re-fit a model per call, so only structural
properties are asserted for them)."""

from __future__ import annotations

import pytest

from confidence_estimator import ConfidenceEstimationModule, XGBoostConfidenceEstimator
from gate import GateDecision, evaluate_series


def test_scenario_a_high_confidence_autonomous():
    rec = evaluate_series("SKU_A", "North", model_type="xgboost", inventory_scenario="below_rop")
    assert rec.decision == GateDecision.AUTONOMOUS_EXECUTION
    assert rec.confidence >= 0.85 and rec.reorder_required


def test_scenario_b_low_confidence_human_approval():
    rec = evaluate_series("SKU_B", "West", model_type="xgboost", inventory_scenario="below_rop")
    assert rec.decision == GateDecision.HUMAN_APPROVAL
    assert rec.confidence < 0.85


def test_scenario_c_no_reorder():
    rec = evaluate_series("SKU_A", "North", model_type="xgboost", inventory_scenario="above_rop")
    assert rec.decision == GateDecision.NO_ACTION and rec.suggested_order_quantity == 0.0


def test_explicit_inventory_and_inventory_status_dict():
    from ss_rop_calculator import reorder_status
    import numpy as np
    st = reorder_status("SKU_A", "North", 100.0, np.tile([0.0, 20.0], 45), current_inventory=500.0)
    d = st.to_dict()
    assert d["reorder_required"] is True and d["safety_stock"] == 43.52
    rec = evaluate_series("SKU_A", "North", model_type="xgboost", current_inventory=1e9)
    assert rec.decision == GateDecision.NO_ACTION


def test_unknown_model_type_rejected():
    with pytest.raises(ValueError):
        evaluate_series("SKU_A", "North", model_type="arima")


@pytest.mark.parametrize("model_type", ["prophet", "lstm"])
def test_prophet_and_lstm_paths_route(model_type):
    rec = evaluate_series("SKU_C", "West", model_type=model_type, inventory_scenario="below_rop")
    assert rec.model_type == model_type and 0.0 <= rec.confidence <= 1.0
    assert rec.decision in (GateDecision.AUTONOMOUS_EXECUTION, GateDecision.HUMAN_APPROVAL)


def test_facade_lazily_builds_estimators():
    m = ConfidenceEstimationModule()
    assert isinstance(m.xgboost, XGBoostConfidenceEstimator) and m.xgboost is m.xgboost
    assert m.prophet.alpha == 1.0 and m.lstm.n_passes == 30
