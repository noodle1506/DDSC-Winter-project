"""ARIMA modeling: fit, forecast, and evaluate."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pmdarima as pm

from src.models.evaluate import compute_metrics


def fit_arima(train: pd.Series) -> pm.ARIMA:
    """Fit an auto-ARIMA model on a training series."""
    model = pm.auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_order=10,
    )
    return model


def run_arima_pipeline(
    df: pd.DataFrame,
    symbol: str,
    test_days: int = 60,
    out_dir: Path | None = None,
) -> dict:
    """Full ARIMA pipeline: split, fit, forecast, evaluate, and optionally save outputs."""
    close = df["close"]

    if len(close) < test_days + 60:
        raise ValueError(f"{symbol}: not enough data ({len(close)} rows) for test_days={test_days}")

    train = close.iloc[:-test_days]
    test = close.iloc[-test_days:]

    # Fit
    model = fit_arima(train)

    # Forecast
    forecast = model.predict(n_periods=test_days)
    forecast = np.asarray(forecast, dtype=float)

    # Metrics
    metrics = compute_metrics(test.values, forecast)
    metrics["symbol"] = symbol
    metrics["order"] = str(model.order)

    # Save outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Results CSV
        results = pd.DataFrame({
            "date": test.index,
            "actual": test.values,
            "forecast": forecast,
        })
        results.to_csv(out_dir / f"{symbol}_arima_results.csv", index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        # Show last 120 days of training + test period
        tail_train = train.iloc[-120:]
        ax.plot(tail_train.index, tail_train.values, label="Train (last 120 days)", color="blue")
        ax.plot(test.index, test.values, label="Actual", color="green")
        ax.plot(test.index, forecast, label="ARIMA Forecast", color="red", linestyle="--")
        ax.set_title(f"{symbol} — ARIMA{model.order} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol}_arima_forecast.png", dpi=150)
        plt.close(fig)

    return metrics
