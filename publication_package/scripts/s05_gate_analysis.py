"""
s05_gate_analysis.py  (Task 2)

Decision-gate threshold sweep and baseline comparison on the held-out test
sets (one gate decision per series-day). Every test forecast is treated as a
reorder-required decision -- the `below_rop` scenario that produced the
repo's 77.8 % / 22.2 % claim -- so the gate reduces to
`confidence >= threshold -> AUTONOMOUS_EXECUTION, else HUMAN_APPROVAL`
(decision_gate/gate.route is called on every row to guarantee that).

Ground truth: a decision "should have been deferred" iff the realised
absolute percentage error of the forecast that drove it exceeds a tolerance
(primary 10 %, sensitivity 5 % / 15 %).

Outputs
  publication_package/gate_threshold_sweep.csv
  publication_package/gate_baseline_comparison.csv
  publication_package/gate_roc_curve.png / .svg
  publication_package/gate_original_scan_realized_error.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (PKG, INTER, SEED, APE_TOLERANCE, auroc, cached, config,
                    confidence_vec, ape, load_raw)

THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)] + [0.99]
TOLERANCES = [0.05, APE_TOLERANCE, 0.15]
REPO_AUTO_RATE = 7 / 9  # 77.8 % from docs/gate_accuracy_scan.csv
N_RANDOM = 10000
N_BOOT = 2000
BLOCK = 7


def load_variants() -> dict[str, pd.DataFrame]:
    cfg = config()
    v = {}
    x = cached("test_predictions_xgboost", None)
    x["confidence"] = confidence_vec(x.point, x.lower, x.upper, cfg["xgboost"]["alpha"])
    v["xgboost (q10/q90)"] = x
    p = cached("test_predictions_prophet", None)
    p["confidence"] = confidence_vec(p.point, p.lower, p.upper, cfg["prophet"]["alpha"])
    v["prophet (native interval)"] = p
    pc = pd.read_csv(INTER / "test_predictions_prophet_conformal.csv", parse_dates=["date"])
    pc["confidence"] = confidence_vec(pc.point, pc.lower_normalized_conformal_flags,
                                      pc.upper_normalized_conformal_flags, cfg["prophet"]["alpha"])
    v["prophet (normalized conformal)"] = pc
    lm = cached("test_predictions_lstm_mc", None)
    lm["confidence"] = confidence_vec(lm.point, lm.lower, lm.upper, cfg["lstm"]["alpha"])
    v["lstm (MC-dropout)"] = lm
    le = pd.read_csv(INTER / "test_predictions_lstm_ensemble.csv", parse_dates=["date"])
    le["confidence"] = confidence_vec(le.point, le.lower, le.upper, cfg["lstm"]["alpha"])
    v["lstm (deep ensemble)"] = le
    ctx = cached("test_predictions_xgboost", None)[["product_id", "region", "date", "is_promo", "is_holiday", "cv7"]]
    for k, d in v.items():
        d["confidence"] = np.round(d["confidence"], 10)
        d["ape"] = ape(d.actual, d.point)
        if "cv7" not in d or d["cv7"].isna().any():
            from common import context_features
            c = context_features(load_raw())
            d.drop(columns=[c_ for c_ in ["is_promo", "is_holiday", "cv7", "day_of_week"] if c_ in d], inplace=True)
            v[k] = d.merge(c, on=["product_id", "region", "date"], how="left")
    return v


def gate_routes(conf: np.ndarray, threshold: float) -> np.ndarray:
    """True = AUTONOMOUS_EXECUTION. Uses the real gate.route on each row."""
    from gate import route, GateDecision
    from confidence_estimator import ConfidenceResult
    from ss_rop_calculator import InventoryStatus
    inv = InventoryStatus("x", "x", 1, 1, 7, 1, 10, 5, True, 6)  # reorder required
    out = np.empty(len(conf), bool)
    for i, c in enumerate(conf):
        cr = ConfidenceResult("m", 1, 1, 1, 0, 0, float(c), "x")
        out[i] = route(inv, cr, threshold=threshold).decision == GateDecision.AUTONOMOUS_EXECUTION
    return out


def outcome(auto: np.ndarray, bad: np.ndarray, e: np.ndarray) -> dict:
    defer = ~auto
    tp, fp = int((defer & bad).sum()), int((defer & ~bad).sum())
    fn, tn = int((auto & bad).sum()), int((auto & ~bad).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (tp and not np.isnan(prec + rec)) else 0.0
    return dict(n=len(auto), n_autonomous=int(auto.sum()), n_human_approval=int(defer.sum()),
                autonomous_rate=float(auto.mean()), n_bad=int(bad.sum()),
                bad_deferred_TP=tp, good_deferred_FP=fp, bad_executed_FN=fn, good_executed_TN=tn,
                precision_defer=prec, recall_defer=rec, f1_defer=f1,
                bad_execution_rate=fn / max(int(auto.sum()), 1) if auto.any() else np.nan,
                mean_ape_autonomous=float(e[auto].mean()) if auto.any() else np.nan,
                p90_ape_autonomous=float(np.percentile(e[auto], 90)) if auto.any() else np.nan,
                mean_ape_deferred=float(e[defer].mean()) if defer.any() else np.nan)


def block_boot_idx(df: pd.DataFrame, rng) -> list[np.ndarray]:
    """Moving-block bootstrap over dates (all series of a date move together)."""
    dates = np.sort(df.date.unique())
    pos = {d: i for i, d in enumerate(dates)}
    row_date = df.date.map(pos).values
    by_date = [np.where(row_date == i)[0] for i in range(len(dates))]
    n = len(dates)
    idxs = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))
        sel = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
        idxs.append(np.concatenate([by_date[i] for i in sel]))
    return idxs


def main():
    variants = load_variants()
    rng = np.random.default_rng(SEED)

    # ---------------- (a) threshold sweep ----------------
    sweep = []
    for name, d in variants.items():
        conf, e = d.confidence.values, d.ape.values
        for tol in TOLERANCES:
            bad = e > tol
            au = auroc(-conf, bad)
            for t in THRESHOLDS:
                auto = gate_routes(conf, t)
                sweep.append(dict(confidence_source=name, ape_tolerance=tol, threshold=t,
                                  auroc_low_conf_vs_bad=au, **outcome(auto, bad, e)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(PKG / "gate_threshold_sweep.csv", index=False)

    # ---------------- (b) baselines ----------------
    rows = []
    for name, d in variants.items():
        conf, e = d.confidence.values, d.ape.values
        bad = e > APE_TOLERANCE
        n = len(d)
        boot = block_boot_idx(d, rng)

        def add(label, auto, note="", ci=None):
            r = dict(confidence_source=name, gate=label, **outcome(auto, bad, e), note=note)
            if ci is not None:
                r.update(ci)
            rows.append(r)

        def diff_ci(auto):
            """Bad-execution-rate of `auto` minus that of a random gate at the
            SAME rate (whose expectation is the base rate of bad forecasts in
            the resample), block-bootstrapped over dates."""
            ds = []
            for idx in boot:
                a, b = auto[idx], bad[idx]
                if a.any():
                    ds.append(b[a].mean() - b.mean())
            ds = np.array(ds)
            pe = bad[auto].mean() - bad.mean() if auto.any() else np.nan
            return dict(bad_exec_rate_minus_random_same_rate=pe,
                        diff_ci_low=np.percentile(ds, 2.5), diff_ci_high=np.percentile(ds, 97.5),
                        frac_boot_diff_ge_0=float(np.mean(ds >= 0)))

        auto85 = gate_routes(conf, 0.85)
        add("confidence_gate @0.85 (deployed)", auto85, ci=diff_ci(auto85))
        # confidence gate whose threshold is set to reproduce the repo's 77.8 % rate
        k = int(round(REPO_AUTO_RATE * n))
        t_match = np.sort(conf)[::-1][k - 1]
        auto_m = gate_routes(conf, t_match)
        add(f"confidence_gate @rate 77.8% (threshold={t_match:.4f})", auto_m, ci=diff_ci(auto_m))
        add("always_autonomous (= autonomous iff reorder_required here)", np.ones(n, bool),
            note="every row is reorder-required in this population, so the two naive rules coincide")
        for rate, label in [(REPO_AUTO_RATE, "random_gate @77.8%"),
                            (auto85.mean(), f"random_gate @matched rate {auto85.mean():.3f}")]:
            sims = []
            for _ in range(N_RANDOM):
                a = rng.random(n) < rate
                sims.append((bad[a].mean(), e[a].mean()))
            sims = np.array(sims)
            r = outcome(rng.random(n) < rate, bad, e)  # one representative draw for counts
            r.update(bad_execution_rate=float(sims[:, 0].mean()), mean_ape_autonomous=float(sims[:, 1].mean()),
                     autonomous_rate=rate)
            rows.append(dict(confidence_source=name, gate=label, **r,
                             note=f"bad_execution_rate / mean_ape_autonomous = mean of {N_RANDOM} draws; "
                                  f"95% range of bad_execution_rate [{np.percentile(sims[:,0],2.5):.4f}, {np.percentile(sims[:,0],97.5):.4f}]; "
                                  "counts columns are one representative draw"))
        # non-ML heuristics
        promo = ((d.is_promo == 1) | (d.is_holiday == 1)).values
        add("heuristic: defer iff promo or holiday", ~promo, ci=diff_ci(~promo),
            note="uses the exact flags that drive the synthetic data generator's noise")
        cv = d.cv7.values
        kdef = int((~auto85).sum())
        auto_cv = np.ones(n, bool)
        if kdef:
            auto_cv[np.argsort(-cv)[:kdef]] = False
        add("heuristic: defer the most volatile (trailing 7-day CV), same deferral count as gate@0.85",
            auto_cv, ci=diff_ci(auto_cv) if kdef else None)
    comp = pd.DataFrame(rows)
    comp.to_csv(PKG / "gate_baseline_comparison.csv", index=False)

    # ---------------- original 9-decision scan, realised error ----------------
    scan = pd.read_csv(PKG.parent / "docs" / "gate_accuracy_scan.csv")
    x = variants["xgboost (q10/q90)"]
    last = x[x.date == x.date.max()]
    m = scan.merge(last, left_on=["Product", "Region"], right_on=["product_id", "region"])
    m = m[["Product", "Region", "Confidence", "confidence", "Decision", "Forecast", "point", "actual", "ape"]]
    m.columns = ["product", "region", "confidence_in_repo_csv", "confidence_recomputed", "decision",
                 "forecast_in_repo_csv", "forecast_recomputed", "actual_sales_2025_08_01", "realized_ape"]
    m.to_csv(PKG / "gate_original_scan_realized_error.csv", index=False)

    plot(variants)
    pd.set_option("display.width", 250)
    print(comp[["confidence_source", "gate", "autonomous_rate", "bad_execution_rate", "mean_ape_autonomous",
                "precision_defer", "recall_defer", "bad_exec_rate_minus_random_same_rate", "diff_ci_low", "diff_ci_high"]].to_string())
    print(sweep[(sweep.ape_tolerance == APE_TOLERANCE)][["confidence_source", "threshold", "autonomous_rate",
          "precision_defer", "recall_defer", "f1_defer", "bad_execution_rate", "auroc_low_conf_vs_bad"]].to_string())
    print(m.to_string())


def plot(variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference palette (dataviz skill, light mode), fixed slot order
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    for (name, d), c in zip(variants.items(), colors):
        conf, bad = d.confidence.values, (d.ape.values > APE_TOLERANCE)
        ts = np.unique(np.concatenate([[-np.inf], conf, [np.inf]]))
        tpr, fpr, prec, rec = [], [], [], []
        for t in ts:
            defer = conf < t
            tp, fp = (defer & bad).sum(), (defer & ~bad).sum()
            tpr.append(tp / bad.sum()); fpr.append(fp / (~bad).sum())
            if defer.any():
                prec.append(tp / defer.sum()); rec.append(tp / bad.sum())
        au = auroc(-conf, bad)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUROC={au:.3f}")
        axes[1].plot(rec, prec, color=c, lw=2, label=name)
        d85 = conf < 0.85
        tp = (d85 & bad).sum()
        axes[0].plot((d85 & ~bad).sum() / (~bad).sum(), tp / bad.sum(), "o", ms=8, color=c,
                     mec="white", mew=2)
        if d85.any():
            axes[1].plot(tp / bad.sum(), tp / d85.sum(), "o", ms=8, color=c, mec="white", mew=2)
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=muted)
    axes[0].text(0.86, 0.62, "random gate", color=muted, ha="center")
    base = np.mean(variants["xgboost (q10/q90)"].ape > APE_TOLERANCE)
    axes[1].axhline(base, ls="--", lw=1, color=colors[0])
    axes[1].text(0.62, base + 0.015, f"random gate, XGBoost (base rate {base:.3f})", color=muted, ha="right")
    axes[0].set(xlabel="False-deferral rate (good forecasts sent to human)",
                ylabel="Recall: bad forecasts deferred", title="ROC: low confidence as a detector of APE > 10%")
    axes[1].set(xlabel="Recall: bad forecasts deferred", ylabel="Precision of deferrals",
                title="Precision-recall of deferral", ylim=(0, 1))
    for ax in axes:
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.title.set_color(ink)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.045), labelcolor=ink)
    fig.text(0.01, 0.01, "Dots mark the deployed threshold (0.85). Held-out test sets, one decision per series-day; "
             "bad = realised APE > 10%.", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(PKG / "gate_roc_curve.png", dpi=200, facecolor="white")
    # fixed id salt + no date so the SVG is byte-identical across runs
    with plt.rc_context({"svg.hashsalt": "publication_package"}):
        fig.savefig(PKG / "gate_roc_curve.svg", facecolor="white", metadata={"Date": None})


if __name__ == "__main__":
    main()
