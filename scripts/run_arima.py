"""Fit ARIMA models and generate forecasts for all tickers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.tickers import TICKERS
from src.models.arima import run_arima_pipeline


def main():
    arima_dir = OUTPUT_DIR / "arima"
    total = len(TICKERS)
    all_metrics = []

    print(f"Running ARIMA on {total} tickers...\n")

    for i, symbol in enumerate(TICKERS, 1):
        path = PROCESSED_DIR / f"{symbol}_daily_clean.csv"
        if not path.exists():
            print(f"  [{i}/{total}] {symbol}: SKIPPED — cleaned file not found")
            continue

        try:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            metrics = run_arima_pipeline(df, symbol, test_days=60, out_dir=arima_dir)
            all_metrics.append(metrics)
            print(f"  [{i}/{total}] {symbol}: ARIMA{metrics['order']} — "
                  f"RMSE={metrics['rmse']}, MAE={metrics['mae']}, MAPE={metrics['mape']}%")
        except Exception as e:
            print(f"  [{i}/{total}] {symbol}: FAILED — {e}")

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(arima_dir / "arima_summary.csv", index=False)
        print(f"\n--- ARIMA Summary ---")
        print(summary.to_string(index=False))
        print(f"\nSaved to: {arima_dir / 'arima_summary.csv'}")


if __name__ == "__main__":
    main()
