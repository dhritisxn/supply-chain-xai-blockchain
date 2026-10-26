"""
lstm_attention.py

Standardizes the LSTM's attention weights (produced by the AdditiveAttention
layer inside models/lstm_model.py) into the same explanation shape used by
the SHAP and Prophet-decomposition modules.

Per the Review-1 XAI strategy table, this is explicitly the "Moderate /
acknowledged XAI limitation" tier: attention weights show *where* the model
looked in the input window, not a rigorously causal feature attribution
the way SHAP values are. This module is deliberately honest about that
distinction rather than overselling it.
"""

from __future__ import annotations
import numpy as np


def explain_prediction(attention_weights: np.ndarray, window_dates: list, predicted_value: float) -> dict:
    """
    attention_weights: 1D array of length WINDOW (softmax weights over the
        input timesteps for a single prediction).
    window_dates: the calendar dates corresponding to each timestep, most
        recent last -- same order as attention_weights.
    """
    weights = attention_weights.flatten()
    pairs = list(zip([str(d) for d in window_dates], weights.tolist()))
    pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)

    return {
        "predicted_value": float(predicted_value),
        "attention_by_day": dict(pairs),
        "top_attended_days": pairs_sorted[:5],
        "explainability_tier": "moderate (attention approximation, not causal attribution)",
    }
