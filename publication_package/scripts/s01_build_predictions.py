"""
s01_build_predictions.py

Step 1 of the reproducibility package. Runs the SAVED XGBoost / Prophet /
LSTM artefacts over their original held-out test sets and caches row-level
predictions + intervals in publication_package/intermediate/. Every later
script reads these caches, so all analyses use the exact same forecasts.

Also writes publication_package/metric_reproduction_check.csv, which
compares the point metrics recomputed here against (a) the metrics.json /
benchmark_summary.json files saved next to the models and (b) the numbers
quoted in docs/report_data.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (PKG, SAVED, ROOT, SERIES, SEED, load_raw, context_features,
                    xgb_test_predictions, prophet_test_predictions,
                    lstm_test_predictions_mc, cached)


def main(refresh: bool = True):
    raw = load_raw()
    ctx = context_features(raw)

    def with_ctx(df):
        return df.merge(ctx, on=["product_id", "region", "date"], how="left")

    xgb = cached("test_predictions_xgboost", lambda: with_ctx(xgb_test_predictions(raw)), refresh)
    pro = cached("test_predictions_prophet", lambda: with_ctx(prophet_test_predictions(raw, SEED)), refresh)
    lstm = cached("test_predictions_lstm_mc", lambda: with_ctx(lstm_test_predictions_mc(raw, SEED)), refresh)

    report = json.load(open(ROOT / "docs" / "report_data.json"))["model_metrics"]
    bench = json.load(open(SAVED / "benchmark_summary.json"))
    saved = {m: json.load(open(SAVED / m / "metrics.json")) for m in ["prophet", "lstm"]}

    def rm(a, p):
        a, p = np.asarray(a), np.asarray(p)
        return float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p) / np.abs(a)))

    rows = []
    r, m = rm(xgb.actual, xgb.point)
    rows.append(dict(model="xgboost", series="ALL_pooled", metric="rmse", recomputed=r,
                     saved_artifact=bench["xgboost"]["rmse"], report_data_json=report["xgboost"]["aggregate"]["rmse"]))
    rows.append(dict(model="xgboost", series="ALL_pooled", metric="mape", recomputed=m,
                     saved_artifact=bench["xgboost"]["mape"], report_data_json=report["xgboost"]["aggregate"]["mape"]))
    for name, df, pcol in [("prophet", pro, "point"), ("lstm", lstm, "point_deterministic")]:
        per = []
        for product, region in SERIES:
            key = f"{product}__{region}"
            s = df[(df.product_id == product) & (df.region == region)]
            r, m = rm(s.actual, s[pcol])
            per.append((r, m))
            for metric, val in [("rmse", r), ("mape", m)]:
                rows.append(dict(model=name, series=key, metric=metric, recomputed=val,
                                 saved_artifact=saved[name][key][metric],
                                 report_data_json=report[name]["per_series"][key][metric]))
        for i, metric in enumerate(["rmse", "mape"]):
            rows.append(dict(model=name, series="ALL_mean_of_series", metric=metric,
                             recomputed=float(np.mean([p[i] for p in per])),
                             saved_artifact=float(np.mean([v[metric] for v in saved[name].values()])),
                             report_data_json=report[name]["aggregate"][metric]))
    out = pd.DataFrame(rows)
    out["recomputed_minus_saved"] = out.recomputed - out.saved_artifact
    out["recomputed_minus_report"] = out.recomputed - out.report_data_json
    out.to_csv(PKG / "metric_reproduction_check.csv", index=False)

    spans = pd.DataFrame([
        dict(model=n, n_rows=len(d), first_date=d.date.min().date(), last_date=d.date.max().date())
        for n, d in [("xgboost", xgb), ("prophet", pro), ("lstm", lstm)]
    ])
    spans.to_csv(PKG / "test_set_spans.csv", index=False)
    print(spans.to_string(index=False))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
