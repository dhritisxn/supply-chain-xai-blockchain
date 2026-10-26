"""
ss_rop_calculator.py

Implements the inventory control formulas from the Review-1 report:

    Safety Stock (SS)   = Z * sigma_d * sqrt(L)
    Reorder Point (ROP) = (D_daily * L) + SS

Where Z = service level factor, sigma_d = demand uncertainty, L = lead
time (days), D_daily = forecasted daily demand.

This module does NOT decide autonomous-vs-human action -- that's the
Decision Gate (decision_gate/gate.py), which imports `reorder_status()`
from here and combines it with the Confidence Estimation Module's score.

No real inventory-on-hand data was provided, so `simulate_current_inventory()`
generates a plausible current stock level for demo purposes. Swap it for a
real inventory feed by replacing that one function -- everything downstream
(reorder_status, the Decision Gate) only needs a `current_inventory` number,
not where it came from.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent / "data"))

CONFIG_PATH = Path(__file__).parent / "inventory_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_inventory_config() -> dict:
    """Single source of truth for SS/ROP parameters -- re-reads the small
    YAML file each call so an edit takes effect without restarting
    long-lived processes (backend/dashboard), same convention as
    confidence_estimator.get_threshold()."""
    return _load_config()


@dataclass
class InventoryStatus:
    product_id: str
    region: str
    daily_demand_forecast: float
    demand_std: float
    lead_time_days: float
    safety_stock: float
    reorder_point: float
    current_inventory: float
    reorder_required: bool
    suggested_order_quantity: float

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "region": self.region,
            "daily_demand_forecast": round(self.daily_demand_forecast, 2),
            "demand_std": round(self.demand_std, 2),
            "lead_time_days": self.lead_time_days,
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "current_inventory": round(self.current_inventory, 2),
            "reorder_required": self.reorder_required,
            "suggested_order_quantity": round(self.suggested_order_quantity, 2),
        }


def compute_demand_std(historical_sales: np.ndarray, window: Optional[int] = None) -> float:
    """Trailing standard deviation of actual daily sales -- the sigma_d
    input to the Safety Stock formula."""
    cfg = get_inventory_config()
    window = window if window is not None else cfg["DEMAND_STD_WINDOW_DAYS"]
    tail = historical_sales[-window:] if len(historical_sales) > window else historical_sales
    return float(np.std(tail))


def compute_safety_stock(demand_std: float, lead_time_days: Optional[float] = None,
                          z_score: Optional[float] = None) -> float:
    cfg = get_inventory_config()
    lead_time_days = lead_time_days if lead_time_days is not None else cfg["LEAD_TIME_DAYS"]
    z_score = z_score if z_score is not None else cfg["Z_SCORE"]
    return float(z_score * demand_std * np.sqrt(lead_time_days))


def compute_reorder_point(daily_demand_forecast: float, safety_stock: float,
                           lead_time_days: Optional[float] = None) -> float:
    cfg = get_inventory_config()
    lead_time_days = lead_time_days if lead_time_days is not None else cfg["LEAD_TIME_DAYS"]
    return float(daily_demand_forecast * lead_time_days + safety_stock)


def simulate_current_inventory(reorder_point: float, safety_stock: float,
                                scenario: str = "random", rng: Optional[np.random.Generator] = None) -> float:
    """
    No real inventory feed was provided -- this stands in for one, purely
    for demo/testing. `scenario` lets the Decision Gate's demo scripts
    deliberately hit each of the three required Review-2 scenarios:

      "below_rop"    -> current stock already under the reorder point
                        (reorder required)
      "above_rop"    -> comfortably above the reorder point
                        (no reorder required)
      "random"       -> uniformly random around the reorder point (default)

    Replace this function with a real inventory-on-hand feed in
    production; nothing downstream needs to change.
    """
    rng = rng or np.random.default_rng()
    if scenario == "below_rop":
        return float(rng.uniform(0.3, 0.9) * reorder_point)
    if scenario == "above_rop":
        return float(reorder_point + safety_stock + rng.uniform(0.5, 2.0) * safety_stock)
    return float(rng.uniform(0.4, 1.6) * reorder_point)


def reorder_status(product_id: str, region: str, daily_demand_forecast: float,
                    historical_sales: np.ndarray, current_inventory: Optional[float] = None,
                    scenario: str = "random", lead_time_days: Optional[float] = None,
                    z_score: Optional[float] = None) -> InventoryStatus:
    """
    Full SS/ROP evaluation for one product/region. If `current_inventory`
    isn't supplied, one is simulated (see simulate_current_inventory).
    """
    demand_std = compute_demand_std(historical_sales)
    safety_stock = compute_safety_stock(demand_std, lead_time_days, z_score)
    reorder_point = compute_reorder_point(daily_demand_forecast, safety_stock, lead_time_days)

    if current_inventory is None:
        current_inventory = simulate_current_inventory(reorder_point, safety_stock, scenario)

    reorder_required = current_inventory <= reorder_point
    # simple order-up-to heuristic: bring stock back up to ROP + one
    # safety-stock buffer above it, not a full EOQ optimization (out of
    # scope -- no holding/ordering cost data available)
    target_level = reorder_point + safety_stock
    order_qty = max(0.0, target_level - current_inventory) if reorder_required else 0.0

    lead_time_days = lead_time_days if lead_time_days is not None else get_inventory_config()["LEAD_TIME_DAYS"]

    return InventoryStatus(
        product_id=product_id,
        region=region,
        daily_demand_forecast=daily_demand_forecast,
        demand_std=demand_std,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        current_inventory=current_inventory,
        reorder_required=reorder_required,
        suggested_order_quantity=order_qty,
    )


if __name__ == "__main__":
    from data_loader import load_raw_sales

    raw = load_raw_sales()
    sub = raw[(raw["product_id"] == "SKU_A") & (raw["region"] == "North")].sort_values("date")
    historical_sales = sub["sales"].values
    recent_avg_demand = float(historical_sales[-14:].mean())  # stand-in for a real forecast in this standalone test

    for scenario in ["below_rop", "above_rop", "random"]:
        status = reorder_status(
            "SKU_A", "North", recent_avg_demand, historical_sales, scenario=scenario
        )
        print(f"[{scenario}]", status.to_dict())
