"""
lstm_model.py

Trains a sequence-to-one LSTM (with a lightweight additive attention layer
on top) to forecast next-day demand from a rolling window of past days.
One model is trained per (product_id, region) series -- consistent with
the Prophet setup.

The attention layer serves double duty:
  1. It's part of the forecasting architecture.
  2. Its output weights are the "attention approximation" explainability
     signal referenced in the Review-1 XAI strategy table (moderate
     transparency, since it's an approximation rather than a ground-truth
     causal attribution the way SHAP is for XGBoost).

MC-Dropout inference (multiple stochastic forward passes with dropout left
"on") is wired in here via `predict_with_uncertainty()` because it needs
access to the trained Keras model object -- the Confidence Estimation
Module (Phase 2) will call this function rather than reimplementing it.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent / "data"))
from data_loader import load_raw_sales

SAVED_DIR = Path(__file__).parent / "saved_models" / "lstm"
WINDOW = 21  # days of history used to predict the next day


class AdditiveAttention(layers.Layer):
    """Simple additive (Bahdanau-style) attention over the LSTM's timesteps."""

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, encoder_outputs):
        # encoder_outputs: (batch, timesteps, hidden)
        score = self.V(tf.nn.tanh(self.W(encoder_outputs)))          # (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)                        # (batch, timesteps, 1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)    # (batch, hidden)
        return context, tf.squeeze(weights, axis=-1)                  # weights: (batch, timesteps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_model(window: int = WINDOW, n_features: int = 1, dropout: float = 0.2):
    """
    Returns (training_model, inference_model). Both share the same layers/
    weights -- training_model has a single 'forecast' output so Keras'
    standard single-loss compile/fit works cleanly; inference_model exposes
    the extra attention-weights output (used only for explainability, never
    for backprop) so predict_with_uncertainty() can read it after training.
    """
    inputs = layers.Input(shape=(window, n_features))
    x = layers.LSTM(48, return_sequences=True, dropout=dropout, recurrent_dropout=0.0)(inputs)
    x = layers.Dropout(dropout)(x, training=True)  # keep active at inference for MC-Dropout
    context, attn_weights = AdditiveAttention(32)(x)
    x = layers.Dense(16, activation="relu")(context)
    x = layers.Dropout(dropout)(x, training=True)
    output = layers.Dense(1, name="forecast")(x)

    training_model = keras.Model(inputs=inputs, outputs=output)
    training_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    inference_model = keras.Model(inputs=inputs, outputs=[output, attn_weights])
    return training_model, inference_model


def make_windows(series: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X)[..., None], np.array(y)


def train_lstm_for_series(raw: pd.DataFrame, product_id: str, region: str,
                           test_frac: float = 0.15, epochs: int = 25, verbose: int = 0):
    sub = raw[(raw["product_id"] == product_id) & (raw["region"] == region)].sort_values("date")
    values = sub["sales"].values.astype("float32")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = make_windows(scaled, WINDOW)
    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    training_model, inference_model = build_model()
    training_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        verbose=verbose,
    )

    preds_scaled, attn = inference_model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
        "mape": float(mean_absolute_percentage_error(actual, preds)),
    }
    return inference_model, scaler, metrics


def predict_with_uncertainty(model: keras.Model, X: np.ndarray, n_passes: int = 30):
    """
    MC-Dropout: run several stochastic forward passes and use the spread
    across passes as an uncertainty signal. Returns per-sample
    (mean_prediction, std_prediction, attention_weights_from_first_pass).

    IMPORTANT: this calls the model directly as `model(X, training=True)`
    rather than `model.predict(X)`. Keras 3's `.predict()` forces global
    inference mode and ignores the per-layer `training=True` baked into
    the Dropout calls at construction time (verified empirically -- predict()
    returns bit-identical output across repeated calls, while direct
    __call__ with training=True does not). Calling the model directly is
    what actually makes the dropout masks resample each pass.
    """
    X_tensor = tf.constant(X, dtype=tf.float32)
    preds = []
    attn_first = None
    for i in range(n_passes):
        out, attn = model(X_tensor, training=True)
        preds.append(out.numpy().flatten())
        if i == 0:
            attn_first = attn.numpy()
    preds = np.array(preds)  # (n_passes, n_samples)
    return preds.mean(axis=0), preds.std(axis=0), attn_first


def train_all_lstm_models(save: bool = True, epochs: int = 25):
    raw = load_raw_sales()
    products = sorted(raw["product_id"].unique())
    regions = sorted(raw["region"].unique())

    all_metrics = {}
    if save:
        SAVED_DIR.mkdir(parents=True, exist_ok=True)

    for product in products:
        for region in regions:
            key = f"{product}__{region}"
            model, scaler, metrics = train_lstm_for_series(raw, product, region, epochs=epochs)
            all_metrics[key] = metrics
            print(f"  {key}: RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.4f}")

            if save:
                model.save(SAVED_DIR / f"{key}.keras")
                np.save(SAVED_DIR / f"{key}_scaler_mean.npy", scaler.mean_)
                np.save(SAVED_DIR / f"{key}_scaler_scale.npy", scaler.scale_)

    if save:
        with open(SAVED_DIR / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

    avg_rmse = float(np.mean([m["rmse"] for m in all_metrics.values()]))
    avg_mape = float(np.mean([m["mape"] for m in all_metrics.values()]))
    return all_metrics, {"avg_rmse": avg_rmse, "avg_mape": avg_mape}


if __name__ == "__main__":
    metrics, summary = train_all_lstm_models(epochs=25)
    print("LSTM summary:", summary)
