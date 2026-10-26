"""Unit tests for decision_gate/gate.py routing (all three branches)."""

from __future__ import annotations

import pytest

from confidence_estimator import ConfidenceResult
from gate import GateDecision, route
from ss_rop_calculator import InventoryStatus


def _inv(reorder: bool, current=300.0, rop=600.0):
    return InventoryStatus("SKU_A", "North", 80.0, 10.0, 7, 43.5, rop, current, reorder,
                           max(0.0, rop + 43.5 - current) if reorder else 0.0)


def _conf(c: float):
    return ConfidenceResult("xgboost", 80.0, 70.0, 90.0, 20.0, 0.25, c, "quantile_regression_q10_q90",
                            {"quantiles": [0.1, 0.9]})


@pytest.mark.parametrize("conf", [0.0, 0.5, 0.99, 1.0])
def test_no_reorder_is_no_action_regardless_of_confidence(conf):
    rec = route(_inv(False, current=900), _conf(conf), threshold=0.85)
    assert rec.decision == GateDecision.NO_ACTION
    assert "no replenishment needed" in rec.reasoning


@pytest.mark.parametrize("conf", [0.85, 0.86, 0.9137, 1.0])
def test_reorder_and_high_confidence_is_autonomous(conf):
    rec = route(_inv(True), _conf(conf), threshold=0.85)
    assert rec.decision == GateDecision.AUTONOMOUS_EXECUTION
    assert "autonomous execution is allowed" in rec.reasoning


@pytest.mark.parametrize("conf", [0.0, 0.5, 0.8309, 0.8499999])
def test_reorder_and_low_confidence_is_human_approval(conf):
    rec = route(_inv(True), _conf(conf), threshold=0.85)
    assert rec.decision == GateDecision.HUMAN_APPROVAL
    assert "routed to human approval" in rec.reasoning


def test_threshold_boundary_is_inclusive():
    assert route(_inv(True), _conf(0.85), threshold=0.85).decision == GateDecision.AUTONOMOUS_EXECUTION


def test_default_threshold_is_read_from_config():
    rec = route(_inv(True), _conf(0.84))
    assert rec.confidence_threshold == 0.85 and rec.decision == GateDecision.HUMAN_APPROVAL


def test_custom_threshold_changes_routing():
    assert route(_inv(True), _conf(0.84), threshold=0.80).decision == GateDecision.AUTONOMOUS_EXECUTION
    assert route(_inv(True), _conf(0.90), threshold=0.95).decision == GateDecision.HUMAN_APPROVAL


def test_record_carries_full_context():
    rec = route(_inv(True, current=300, rop=600), _conf(0.9), threshold=0.85)
    d = rec.to_dict()
    assert d["product_id"] == "SKU_A" and d["region"] == "North" and d["model_type"] == "xgboost"
    assert d["forecast"] == 80.0 and d["confidence"] == 0.9 and d["reorder_required"] is True
    assert d["reorder_point"] == 600.0 and d["current_inventory"] == 300.0
    assert d["suggested_order_quantity"] == pytest.approx(343.5)
    assert d["decision"] == "AUTONOMOUS_EXECUTION"
    assert d["confidence_metadata"] == {"quantiles": [0.1, 0.9]}


def test_model_type_override():
    assert route(_inv(True), _conf(0.9), model_type="prophet", threshold=0.85).model_type == "prophet"
