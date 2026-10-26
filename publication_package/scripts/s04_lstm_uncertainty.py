"""
s04_lstm_uncertainty.py  (Task 4)

Compares the production LSTM uncertainty signal (MC-Dropout, 30 passes, on
the saved models) against a deep ensemble of 5 independently initialised
LSTMs trained with the UNCHANGED production training routine
(models/lstm_model.train_lstm_for_series, epochs=25), all scored on the
identical held-out windows.

Variants scored
  mc_dropout_saved_model      : current approach (what the pipeline uses)
  mc_dropout_member_seed{k}   : same MC-Dropout recipe on each new member
                                (separates "method" from "which weights")
  deep_ensemble_5             : mean / std across the 5 members'
                                deterministic forecasts
  deep_ensemble_5_plus_mc     : members x MC-Dropout, total predictive
                                variance = between-member + mean within-member

Outputs
  publication_package/lstm_uncertainty_comparison.csv
  publication_package/lstm_seed_variance.csv
  publication_package/intermediate/test_predictions_lstm_ensemble.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import (PKG, INTER, SEED, SERIES, cached, calibration_metrics, config,
                    load_raw, lstm_windows, set_seeds)

MEMBER_SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 25  # production default in lstm_model.train_all_lstm_models
MODEL_DIR = INTER / "lstm_ensemble"


def train_members(raw):
    import tensorflow as tf
    from lstm_model import train_lstm_for_series, predict_with_uncertainty
    tf.config.experimental.enable_op_determinism()
    cfg = config()["lstm"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for product, region in SERIES:
        for s in MEMBER_SEEDS:
            t0 = time.time()
            set_seeds(s)
            model, scaler, metrics = train_lstm_for_series(raw, product, region, epochs=EPOCHS)
            mean, scale = float(scaler.mean_[0]), float(scaler.scale_[0])
            X, y, split, dates = lstm_windows(raw, product, region, mean, scale)
            X_test, y_test = X[split:], y[split:]
            det, _ = model.predict(X_test, verbose=0)
            set_seeds(SEED)
            mc_mean, mc_std, _ = predict_with_uncertainty(model, X_test, n_passes=cfg["mc_dropout_passes"])
            frames.append(pd.DataFrame({
                "product_id": product, "region": region, "seed": s, "date": dates[split:],
                "actual": y_test * scale + mean,
                "det": det.flatten() * scale + mean,
                "mc_mean": mc_mean * scale + mean,
                "mc_std": mc_std * scale,
            }))
            model.save(MODEL_DIR / f"{product}__{region}__seed{s}.keras")
            print(f"  {product}/{region} seed={s}: RMSE={metrics['rmse']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True)


def main():
    raw = load_raw()
    cfg = config()["lstm"]
    alpha, z = cfg["alpha"], cfg["z_multiplier"]

    members = cached("lstm_ensemble_members", lambda: train_members(raw))
    base = cached("test_predictions_lstm_mc", None)

    rows = []
    # 1) production MC-Dropout on saved models
    rows.append(calibration_metrics(base.point.values, base.lower.values, base.upper.values,
                                    base.actual.values, alpha, "mc_dropout_saved_model (current)"))
    # 2) MC-Dropout on each new member
    for s in MEMBER_SEEDS:
        m = members[members.seed == s]
        rows.append(calibration_metrics(m.mc_mean.values, (m.mc_mean - z * m.mc_std).values,
                                        (m.mc_mean + z * m.mc_std).values, m.actual.values,
                                        alpha, f"mc_dropout_member_seed{s}"))
    # 3) deep ensemble
    g = members.groupby(["product_id", "region", "date"], sort=False)
    ens = g.agg(actual=("actual", "first"), point=("det", "mean"), std=("det", lambda v: np.std(v, ddof=1)),
                mc_point=("mc_mean", "mean"), mc_between_var=("mc_mean", lambda v: np.var(v, ddof=1)),
                mc_within_var=("mc_std", lambda v: np.mean(np.square(v)))).reset_index()
    ens["lower"], ens["upper"] = ens.point - z * ens["std"], ens.point + z * ens["std"]
    rows.append(calibration_metrics(ens.point.values, ens.lower.values, ens.upper.values,
                                    ens.actual.values, alpha, "deep_ensemble_5"))
    tot = np.sqrt(ens.mc_between_var + ens.mc_within_var)
    rows.append(calibration_metrics(ens.mc_point.values, (ens.mc_point - z * tot).values,
                                    (ens.mc_point + z * tot).values, ens.actual.values,
                                    alpha, "deep_ensemble_5_plus_mc_dropout"))

    out = pd.DataFrame(rows)
    out.insert(1, "documented_value_in_repo",
               ["-0.081 (threshold_config.yaml; calibrate.py retrains with epochs=15, n_passes=15)"] + [""] * (len(out) - 1))
    mc_members = out[out.method.str.startswith("mc_dropout_member")]
    summary = {"method": "mc_dropout_members_mean_over_5_seeds",
               **{c: mc_members[c].mean() for c in out.columns if c not in ("method", "documented_value_in_repo")
                  and pd.api.types.is_numeric_dtype(out[c])}}
    summary["documented_value_in_repo"] = f"sd of pearson over seeds = {mc_members.pearson_conf_vs_ape.std(ddof=1):.4f}"
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    out.to_csv(PKG / "lstm_uncertainty_comparison.csv", index=False)

    ens.to_csv(INTER / "test_predictions_lstm_ensemble.csv", index=False)

    # seed variance of point accuracy (task 1 supplement)
    sv = []
    for s in MEMBER_SEEDS:
        m = members[members.seed == s]
        per = m.groupby(["product_id", "region"]).apply(
            lambda d: pd.Series({"rmse": np.sqrt(np.mean((d.actual - d.det) ** 2)),
                                 "mape": np.mean(np.abs(d.actual - d.det) / d.actual)}), include_groups=False)
        sv.append(dict(model="lstm", seed=s, rmse_mean_of_series=per.rmse.mean(), mape_mean_of_series=per.mape.mean(),
                       rmse_pooled=float(np.sqrt(np.mean((m.actual - m.det) ** 2)))))
    e_per = ens.groupby(["product_id", "region"]).apply(
        lambda d: pd.Series({"rmse": np.sqrt(np.mean((d.actual - d.point) ** 2)),
                             "mape": np.mean(np.abs(d.actual - d.point) / d.actual)}), include_groups=False)
    sv.append(dict(model="lstm_deep_ensemble_5", seed="mean_of_5", rmse_mean_of_series=e_per.rmse.mean(),
                   mape_mean_of_series=e_per.mape.mean(),
                   rmse_pooled=float(np.sqrt(np.mean((ens.actual - ens.point) ** 2)))))
    b_per = base.groupby(["product_id", "region"]).apply(
        lambda d: pd.Series({"rmse": np.sqrt(np.mean((d.actual - d.point_deterministic) ** 2)),
                             "mape": np.mean(np.abs(d.actual - d.point_deterministic) / d.actual)}), include_groups=False)
    sv.append(dict(model="lstm_saved_model", seed="original", rmse_mean_of_series=b_per.rmse.mean(),
                   mape_mean_of_series=b_per.mape.mean(),
                   rmse_pooled=float(np.sqrt(np.mean((base.actual - base.point_deterministic) ** 2)))))
    sv = pd.DataFrame(sv)
    sv.to_csv(PKG / "lstm_seed_variance.csv", index=False)

    pd.set_option("display.width", 250)
    print(out.T.to_string())
    print(sv.to_string(index=False))


if __name__ == "__main__":
    main()
