"""Train LSTM models and generate forecasts for all tickers."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.tickers import TICKERS
from src.models.lstm import run_lstm_pipeline


def main():
    lstm_dir = OUTPUT_DIR / "lstm"
    checkpoint_dir = lstm_dir / "checkpoints"
    total = len(TICKERS)
    all_metrics = []

    print(f"Running LSTM on {total} tickers...")
    print("(This may take several minutes per ticker on CPU)\n")

    for i, symbol in enumerate(TICKERS, 1):
        path = PROCESSED_DIR / f"{symbol}_daily_clean.csv"
        if not path.exists():
            print(f"  [{i}/{total}] {symbol}: SKIPPED — cleaned file not found")
            continue

        try:
            start = time.time()
            print(f"  [{i}/{total}] {symbol}: Training...")
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            metrics = run_lstm_pipeline(
                df, symbol,
                out_dir=lstm_dir,
                checkpoint_dir=checkpoint_dir,
                verbose=True,
            )
            elapsed = time.time() - start
            all_metrics.append(metrics)
            print(f"  [{i}/{total}] {symbol}: Done in {elapsed:.0f}s — "
                  f"Test: RMSE={metrics['rmse']}, MAE={metrics['mae']}, MAPE={metrics['mape']}% | "
                  f"Val: RMSE={metrics['val_rmse']}, MAPE={metrics['val_mape']}% | "
                  f"Best epoch: {metrics['best_epoch']}\n")
        except Exception as e:
            print(f"  [{i}/{total}] {symbol}: FAILED — {e}\n")

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(lstm_dir / "lstm_summary.csv", index=False)
        print(f"--- LSTM Summary ---")
        print(summary.to_string(index=False))
        print(f"\nSaved to: {lstm_dir / 'lstm_summary.csv'}")


if __name__ == "__main__":
    main()
