"""Shared evaluation metrics for model comparison."""
from __future__ import annotations

import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE between actual and predicted arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    # MAPE — guard against division by zero
    nonzero = actual != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
    else:
        mape = float("nan")

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}
