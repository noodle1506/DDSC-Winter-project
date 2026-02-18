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
        max_d=1,
    )
    return model


def run_arima_pipeline(
    df: pd.DataFrame,
    symbol: str,
    test_days: int = 60,
    out_dir: Path | None = None,
) -> dict:
    """Full ARIMA pipeline using returns-based forecasting.

    Strategy:
      1. Compute daily returns from close prices.
      2. Split returns into train / test (time-based).
      3. Fit auto-ARIMA on training returns (already ~stationary, expect d=0).
      4. Forecast `test_days` returns.
      5. Reconstruct price forecasts from the last known training price.
      6. Evaluate reconstructed prices against actual test prices.
    """
    close = df["close"]

    if len(close) < test_days + 60:
        raise ValueError(f"{symbol}: not enough data ({len(close)} rows) for test_days={test_days}")

    # Compute daily returns (pct_change), drop the leading NaN
    returns = close.pct_change().dropna()

    # Split: the last `test_days` returns correspond to the last `test_days` prices
    # returns has one fewer row than close, so align carefully
    train_returns = returns.iloc[:-test_days]
    test_returns = returns.iloc[-test_days:]

    # Actual test prices for evaluation
    test_prices = close.iloc[-test_days:]
    # Last known price before the test period (anchor for reconstruction)
    last_train_price = close.iloc[-(test_days + 1)]

    # Fit ARIMA on returns
    model = fit_arima(train_returns)

    # Forecast returns
    forecast_returns = model.predict(n_periods=test_days)
    forecast_returns = np.asarray(forecast_returns, dtype=float)

    # Reconstruct prices: price[t] = price[t-1] * (1 + r[t])
    forecast_prices = np.empty(test_days, dtype=float)
    prev_price = last_train_price
    for i, r in enumerate(forecast_returns):
        prev_price = prev_price * (1 + r)
        forecast_prices[i] = prev_price

    forecast_prices = np.maximum(forecast_prices, 0)  # safety clamp

    # Metrics on prices (comparable with LSTM)
    metrics = compute_metrics(test_prices.values, forecast_prices)
    metrics["symbol"] = symbol
    metrics["order"] = str(model.order)

    # Save outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Results CSV
        results = pd.DataFrame({
            "date": test_prices.index,
            "actual": test_prices.values,
            "forecast": forecast_prices,
        })
        results.to_csv(out_dir / f"{symbol}_arima_results.csv", index=False)

        # Plot
        train_prices = close.iloc[:-test_days]
        fig, ax = plt.subplots(figsize=(12, 6))
        tail_train = train_prices.iloc[-120:]
        ax.plot(tail_train.index, tail_train.values, label="Train (last 120 days)", color="blue")
        ax.plot(test_prices.index, test_prices.values, label="Actual", color="green")
        ax.plot(test_prices.index, forecast_prices, label="ARIMA Forecast", color="red", linestyle="--")
        ax.set_title(f"{symbol} — ARIMA{model.order} Forecast (returns-based)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol}_arima_forecast.png", dpi=150)
        plt.close(fig)

    return metrics
