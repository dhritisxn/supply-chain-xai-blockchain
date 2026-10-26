"""
s02_statistical_significance.py  (Task 1)

95% confidence intervals for RMSE / MAPE (and pooled R^2 for XGBoost) via a
moving-block bootstrap of the saved models' held-out residuals, plus paired
bootstrap CIs for between-model differences on the dates all three test sets
share, plus a small XGBoost seed-variation retrain.

Outputs
  publication_package/statistical_significance.csv
  publication_package/statistical_significance_block_sensitivity.csv
  publication_package/statistical_seed_variance_xgboost.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import PKG, SEED, SERIES, cached, load_raw

N_BOOT = 2000
BLOCK = 7  # days; residuals of daily retail forecasts are autocorrelated (weekly cycle)


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap (Kunsch 1989): concatenate randomly-started
    contiguous blocks of length `block`, truncated to n."""
    if block <= 1:
        return rng.integers(0, n, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def to_matrix(df: pd.DataFrame, pcol: str, dates=None):
    """(n_dates, 9) matrices of actual and predicted values, columns in SERIES order."""
    df = df.copy()
    df["key"] = df.product_id + "__" + df.region
    if dates is not None:
        df = df[df.date.isin(dates)]
    a = df.pivot(index="date", columns="key", values="actual")
    p = df.pivot(index="date", columns="key", values=pcol)
    keys = [f"{pr}__{rg}" for pr, rg in SERIES]
    a, p = a[keys], p[keys]
    assert not a.isna().any().any(), "every series must have every date"
    return a.values, p.values, keys


def stats(a: np.ndarray, p: np.ndarray) -> dict:
    """Metrics for one (possibly resampled) panel."""
    se = (a - p) ** 2
    ape = np.abs(a - p) / np.abs(a)
    per_rmse = np.sqrt(se.mean(axis=0))
    per_mape = ape.mean(axis=0)
    out = {
        ("ALL_pooled", "rmse"): float(np.sqrt(se.mean())),
        ("ALL_pooled", "mape"): float(ape.mean()),
        ("ALL_pooled", "r2"): float(1 - se.sum() / ((a - a.mean()) ** 2).sum()),
        ("ALL_mean_of_series", "rmse"): float(per_rmse.mean()),
        ("ALL_mean_of_series", "mape"): float(per_mape.mean()),
    }
    return out, per_rmse, per_mape


def bootstrap_model(a, p, keys, block, rng):
    point, pr, pm = stats(a, p)
    boots = {k: [] for k in point}
    per_r, per_m = [], []
    n = a.shape[0]
    for _ in range(N_BOOT):
        idx = block_indices(n, block, rng)
        s, r, m = stats(a[idx], p[idx])
        for k in s:
            boots[k].append(s[k])
        per_r.append(r)
        per_m.append(m)
    per_r, per_m = np.array(per_r), np.array(per_m)
    rows = []
    for (series, metric), val in point.items():
        b = np.array(boots[(series, metric)])
        rows.append((series, metric, val, *np.percentile(b, [2.5, 97.5])))
    for j, key in enumerate(keys):
        rows.append((key, "rmse", float(pr[j]), *np.percentile(per_r[:, j], [2.5, 97.5])))
        rows.append((key, "mape", float(pm[j]), *np.percentile(per_m[:, j], [2.5, 97.5])))
    return rows


def main():
    raw = load_raw()
    preds = {
        "xgboost": (cached("test_predictions_xgboost", None), "point"),
        "prophet": (cached("test_predictions_prophet", None), "point"),
        # deterministic forward pass == how lstm/metrics.json was computed
        "lstm": (cached("test_predictions_lstm_mc", None), "point_deterministic"),
    }

    rows = []
    sens = []
    for model, (df, pcol) in preds.items():
        a, p, keys = to_matrix(df, pcol)
        rng = np.random.default_rng(SEED)
        for series, metric, pe, lo, hi in bootstrap_model(a, p, keys, BLOCK, rng):
            if model != "xgboost" and metric == "r2":
                continue
            rows.append(dict(model=model, series=series, metric=metric, point_estimate=pe,
                             ci_low=lo, ci_high=hi, n_bootstrap=N_BOOT))
        for block in [1, 7, 14, 28]:
            rng = np.random.default_rng(SEED)
            for series, metric, pe, lo, hi in bootstrap_model(a, p, keys, block, rng):
                if series.startswith("ALL") and metric in ("rmse", "mape"):
                    sens.append(dict(model=model, series=series, metric=metric, block_length=block,
                                     point_estimate=pe, ci_low=lo, ci_high=hi, n_bootstrap=N_BOOT))

    # --- Paired comparison on the dates common to all three test sets -------
    common = set(preds["xgboost"][0].date)
    for df, _ in preds.values():
        common &= set(df.date)
    common = sorted(common)
    mats = {m: to_matrix(df, pcol, common) for m, (df, pcol) in preds.items()}
    n = len(common)
    rng = np.random.default_rng(SEED)
    boot_idx = [block_indices(n, BLOCK, rng) for _ in range(N_BOOT)]

    def panel_stats(m, idx=None):
        a, p, _ = mats[m]
        if idx is not None:
            a, p = a[idx], p[idx]
        return stats(a, p)[0]

    for m in preds:
        pt = panel_stats(m)
        bs = [panel_stats(m, idx) for idx in boot_idx]
        for k in [("ALL_pooled", "rmse"), ("ALL_pooled", "mape"),
                  ("ALL_mean_of_series", "rmse"), ("ALL_mean_of_series", "mape")]:
            b = np.array([s[k] for s in bs])
            rows.append(dict(model=m, series=k[0] + "_common_dates", metric=k[1], point_estimate=pt[k],
                             ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5), n_bootstrap=N_BOOT))

    for m1, m2 in [("xgboost", "prophet"), ("xgboost", "lstm"), ("prophet", "lstm")]:
        pt1, pt2 = panel_stats(m1), panel_stats(m2)
        b1 = [panel_stats(m1, idx) for idx in boot_idx]
        b2 = [panel_stats(m2, idx) for idx in boot_idx]
        for k in [("ALL_pooled", "rmse"), ("ALL_pooled", "mape"),
                  ("ALL_mean_of_series", "rmse"), ("ALL_mean_of_series", "mape")]:
            d = np.array([x[k] - y[k] for x, y in zip(b1, b2)])
            rows.append(dict(model=f"{m1}_minus_{m2}", series=k[0] + "_common_dates", metric=k[1] + "_diff",
                             point_estimate=pt1[k] - pt2[k], ci_low=np.percentile(d, 2.5),
                             ci_high=np.percentile(d, 97.5), n_bootstrap=N_BOOT))

    out = pd.DataFrame(rows)
    out.to_csv(PKG / "statistical_significance.csv", index=False)
    pd.DataFrame(sens).to_csv(PKG / "statistical_significance_block_sensitivity.csv", index=False)
    print(f"common dates: {len(common)} ({common[0].date()} .. {common[-1].date()})")
    print(out[out.series.str.startswith("ALL")].to_string(index=False))

    # --- XGBoost seed variation (cheap to retrain; point model only) -----------
    import xgboost as xgb
    from feature_engineering import build_feature_table, FEATURE_COLS, TARGET_COL
    from xgboost_model import time_based_split, POINT_PARAMS
    feat, _ = build_feature_table(raw)
    tr, te = time_based_split(feat)
    seed_rows = []
    for s in [0, 1, 2, 3, 4, 42]:
        mdl = xgb.XGBRegressor(**{**POINT_PARAMS, "random_state": s})
        mdl.fit(tr[FEATURE_COLS], tr[TARGET_COL])
        p = mdl.predict(te[FEATURE_COLS])
        y = te[TARGET_COL].values
        seed_rows.append(dict(model="xgboost", seed=s, series="ALL_pooled",
                              rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                              mape=float(np.mean(np.abs(y - p) / np.abs(y))),
                              r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())))
    sd = pd.DataFrame(seed_rows)
    sd.to_csv(PKG / "statistical_seed_variance_xgboost.csv", index=False)
    print(sd.to_string(index=False))
    print("seeds 0-4 mean/std rmse:", sd[sd.seed != 42].rmse.mean(), sd[sd.seed != 42].rmse.std(ddof=1))


if __name__ == "__main__":
    main()
