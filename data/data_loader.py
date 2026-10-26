"""
data_loader.py

Generates (or loads) the sales dataset used by the tri-model forecasting
engine. If a real POS/sales CSV is available, point `load_raw_sales()` at it
instead of the synthetic generator -- the downstream feature engineering
expects the same schema either way:

    date, product_id, region, sales, is_holiday, is_promo, price

Synthetic generation is included so the pipeline is runnable end-to-end
without a real dataset (useful for the Review-2 demo). It builds in:
  - long-term trend
  - weekly seasonality (weekend uplift)
  - yearly seasonality (festive-season uplift)
  - holiday spikes
  - promo spikes
  - regional base-demand differences
  - Gaussian noise
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

PRODUCTS = ["SKU_A", "SKU_B", "SKU_C"]
REGIONS = ["North", "South", "West"]

# A small fixed holiday calendar (India-centric, deliberately simple for a demo)
HOLIDAYS = [
    "2023-01-26", "2023-03-08", "2023-08-15", "2023-10-24", "2023-11-12",
    "2023-12-25", "2024-01-26", "2024-03-25", "2024-08-15", "2024-10-12",
    "2024-11-01", "2024-12-25", "2025-01-26", "2025-03-14", "2025-08-15",
]
HOLIDAYS = pd.to_datetime(HOLIDAYS)


def _region_multiplier(region: str) -> float:
    return {"North": 1.15, "South": 1.0, "West": 0.9}[region]


def _product_base_demand(product: str) -> float:
    return {"SKU_A": 120, "SKU_B": 70, "SKU_C": 45}[product]


def generate_synthetic_sales(
    start: str = "2023-01-01",
    end: str = "2025-08-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    rows = []
    for product in PRODUCTS:
        base = _product_base_demand(product)
        for region in REGIONS:
            reg_mult = _region_multiplier(region)

            # long-term trend (slow linear growth + slight curvature)
            trend = np.linspace(0, base * 0.35, n) + 0.00004 * np.arange(n) ** 1.3

            # weekly seasonality: weekend uplift
            dow = dates.dayofweek.values
            weekly = np.where(dow >= 5, base * 0.18, 0.0)

            # yearly seasonality: festive season (Oct-Dec) uplift via sine wave
            doy = dates.dayofyear.values
            yearly = base * 0.25 * np.sin(2 * np.pi * (doy - 250) / 365)
            yearly = np.clip(yearly, 0, None)

            # holiday spikes
            is_holiday = dates.isin(HOLIDAYS).astype(int)
            holiday_effect = is_holiday * base * 0.6

            # promo: randomly scheduled campaigns, ~8% of days, boosts demand
            is_promo = (rng.random(n) < 0.08).astype(int)
            promo_effect = is_promo * base * rng.uniform(0.3, 0.7, n)

            # noise -- heteroscedastic: promo/holiday days are genuinely
            # harder to predict (more behavioral variability in real retail
            # data) than an ordinary day, so their noise variance is higher.
            # This is what gives the Confidence Estimation Module (Phase 2)
            # a real pattern to detect, rather than uniform noise everywhere.
            base_noise_std = base * 0.06
            noise_std = base_noise_std * (1.0 + 1.8 * is_promo + 1.2 * is_holiday)
            noise = rng.normal(0, noise_std, n)

            sales = (base + trend + weekly + yearly + holiday_effect + promo_effect + noise) * reg_mult
            sales = np.clip(sales, 0, None).round().astype(int)

            price = np.round(rng.normal(250 if product == "SKU_A" else 150, 5, n), 2)

            rows.append(pd.DataFrame({
                "date": dates,
                "product_id": product,
                "region": region,
                "sales": sales,
                "is_holiday": is_holiday,
                "is_promo": is_promo,
                "price": price,
            }))

    df = pd.concat(rows, ignore_index=True).sort_values(["product_id", "region", "date"])
    return df.reset_index(drop=True)


def load_raw_sales(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load raw sales data. If `path` is given, reads a real CSV with the
    expected schema. Otherwise generates (and caches) the synthetic dataset.
    """
    if path is not None:
        return pd.read_csv(path, parse_dates=["date"])

    cache_path = RAW_DIR / "synthetic_sales.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    df = generate_synthetic_sales()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    df = load_raw_sales()
    print(df.shape)
    print(df.head())
    print(df.groupby(["product_id", "region"])["sales"].mean())
