"""Unit tests for inventory/ss_rop_calculator.py against hand-computed values.

Formulas (inventory_config.yaml: Z = 1.645, L = 7 days, window = 90 days):
    SS  = Z * sigma_d * sqrt(L)
    ROP = D_daily * L + SS
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ss_rop_calculator as inv


def test_config_values():
    cfg = inv.get_inventory_config()
    assert (cfg["Z_SCORE"], cfg["LEAD_TIME_DAYS"], cfg["DEMAND_STD_WINDOW_DAYS"]) == (1.645, 7, 90)


def test_safety_stock_hand_computed():
    # 1.645 * 10 * sqrt(7) = 1.645 * 10 * 2.6457513110645906 = 43.52260906701252
    assert inv.compute_safety_stock(10.0) == pytest.approx(43.52260906701252, rel=1e-12)
    # explicit overrides: Z=2.0, sigma=5, L=4 -> 2*5*2 = 20
    assert inv.compute_safety_stock(5.0, lead_time_days=4, z_score=2.0) == pytest.approx(20.0)


def test_reorder_point_hand_computed():
    ss = inv.compute_safety_stock(10.0)
    # 100/day * 7 days + 43.5226... = 743.5226...
    assert inv.compute_reorder_point(100.0, ss) == pytest.approx(743.5226090670125)
    assert inv.compute_reorder_point(50.0, 20.0, lead_time_days=4) == pytest.approx(220.0)


def test_demand_std_is_population_std_of_trailing_window():
    # np.std (ddof=0) of [1,2,3,4] = sqrt(1.25)
    assert inv.compute_demand_std(np.array([1, 2, 3, 4])) == pytest.approx(math.sqrt(1.25))
    # only the trailing `window` values are used
    series = np.array([1000.0] * 10 + [1, 2, 3, 4])
    assert inv.compute_demand_std(series, window=4) == pytest.approx(math.sqrt(1.25))


def test_default_window_is_90_days():
    s = np.concatenate([np.full(100, 1e6), np.tile([0.0, 2.0], 45)])  # last 90 alternate 0/2 -> std 1
    assert inv.compute_demand_std(s) == pytest.approx(1.0)


def test_reorder_status_below_rop_hand_computed():
    hist = np.tile([0.0, 20.0], 45)  # sigma_d = 10
    st = inv.reorder_status("SKU_A", "North", 100.0, hist, current_inventory=500.0)
    assert st.demand_std == pytest.approx(10.0)
    assert st.safety_stock == pytest.approx(43.52260906701252)
    assert st.reorder_point == pytest.approx(743.5226090670125)
    assert st.reorder_required is True
    # order-up-to ROP + SS: 743.5226 + 43.5226 - 500 = 287.0452...
    assert st.suggested_order_quantity == pytest.approx(287.04521813402503)


def test_reorder_status_above_rop_orders_nothing():
    hist = np.tile([0.0, 20.0], 45)
    st = inv.reorder_status("SKU_A", "North", 100.0, hist, current_inventory=800.0)
    assert st.reorder_required is False and st.suggested_order_quantity == 0.0


def test_inventory_exactly_at_rop_triggers_reorder():
    hist = np.tile([0.0, 20.0], 45)
    rop = 743.5226090670125
    st = inv.reorder_status("SKU_A", "North", 100.0, hist, current_inventory=rop)
    assert st.reorder_required is True
    assert st.suggested_order_quantity == pytest.approx(43.52260906701252)


@pytest.mark.parametrize("scenario,lo,hi", [("below_rop", 0.3 * 700, 0.9 * 700),
                                            ("random", 0.4 * 700, 1.6 * 700)])
def test_simulated_inventory_ranges(scenario, lo, hi):
    rng = np.random.default_rng(0)
    vals = [inv.simulate_current_inventory(700.0, 50.0, scenario, rng) for _ in range(200)]
    assert min(vals) >= lo and max(vals) <= hi


def test_simulated_above_rop_is_always_above():
    rng = np.random.default_rng(0)
    vals = [inv.simulate_current_inventory(700.0, 50.0, "above_rop", rng) for _ in range(200)]
    assert min(vals) >= 700 + 50 + 0.5 * 50 and max(vals) <= 700 + 50 + 2.0 * 50


def test_below_rop_scenario_always_requires_reorder():
    hist = np.tile([0.0, 20.0], 45)
    for _ in range(50):
        assert inv.reorder_status("SKU_A", "North", 100.0, hist, scenario="below_rop").reorder_required
