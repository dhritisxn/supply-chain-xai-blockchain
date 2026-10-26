"""
shap_explainer.py

Wraps SHAP's TreeExplainer around the trained XGBoost point-forecast model
to produce per-forecast feature attributions -- this is the explainability
layer referenced throughout the Review-1 report as "High: Local & Global
feature impact".

Also provides `hash_explanation()`, used later by the blockchain audit
module: the full SHAP explanation is never written on-chain (too large,
and not necessary for verification) -- only a deterministic hash of it is,
so the explanation can be independently recomputed and checked against the
on-chain record.
"""

from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).parent.parent / "data"))
from feature_engineering import FEATURE_COLS


class ShapExplainer:
    def __init__(self, xgb_model):
        self.model = xgb_model
        self.explainer = shap.TreeExplainer(xgb_model)

    def explain_row(self, feature_row: pd.Series) -> dict:
        """
        Returns a structured local explanation for a single forecast:
        base value, per-feature SHAP contribution, and the feature values
        themselves, sorted by |contribution| descending.
        """
        X = pd.DataFrame([feature_row[FEATURE_COLS].astype(float).to_dict()], columns=FEATURE_COLS)
        shap_values = self.explainer(X)

        contributions = {
            feat: float(val)
            for feat, val in zip(FEATURE_COLS, shap_values.values[0])
        }
        sorted_contribs = dict(
            sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        return {
            "base_value": float(shap_values.base_values[0]),
            "predicted_value": float(shap_values.base_values[0] + shap_values.values[0].sum()),
            "feature_values": {feat: float(X[feat].iloc[0]) for feat in FEATURE_COLS},
            "contributions": sorted_contribs,
            "top_drivers": list(sorted_contribs.items())[:5],
        }

    def explain_batch(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Global/batch SHAP values, e.g. for a summary/importance plot."""
        return self.explainer(feature_df[FEATURE_COLS]).values


def hash_explanation(explanation: dict) -> str:
    """
    Deterministic SHA-256 hash of an explanation dict, used for the
    on-chain audit record (see blockchain/ module). Keys are sorted and
    floats are rounded before hashing so the hash is stable across
    re-serialization / minor floating point noise.
    """
    def _round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: _round_floats(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_round_floats(v) for v in obj]
        return obj

    canonical = json.dumps(_round_floats(explanation), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent / "models"))
    from xgboost_model import load_xgboost_models
    from data_loader import load_raw_sales
    from feature_engineering import build_feature_table

    point_model, _, _, _ = load_xgboost_models()
    raw = load_raw_sales()
    feat, _ = build_feature_table(raw)
    row = feat.iloc[-1]

    explainer = ShapExplainer(point_model)
    explanation = explainer.explain_row(row)
    print("Predicted:", explanation["predicted_value"], " Actual:", row["sales"])
    print("Top drivers:", explanation["top_drivers"])
    print("Explanation hash:", hash_explanation(explanation))
