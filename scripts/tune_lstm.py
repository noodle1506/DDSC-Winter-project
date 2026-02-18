"""Hyperparameter tuning for LSTM via grid search on a single ticker.

Evaluates on the VALIDATION set (test set is held out).
"""
from __future__ import annotations

import sys
import time
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.models.lstm import prepare_lstm_data, build_lstm_model, train_lstm
from src.models.evaluate import compute_metrics

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# --- Tuning configuration ---
TICKER = "AAPL"
TEST_DAYS = 60
VAL_DAYS = 60
TUNING_EPOCHS = 30

PARAM_GRID = {
    "look_back": [30, 60],
    "learning_rate": [0.001, 0.0001],
    "batch_size": [32, 64],
    "hidden_size": [50, 64],
}


def main():
    tune_dir = OUTPUT_DIR / "lstm"
    tune_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    path = PROCESSED_DIR / f"{TICKER}_daily_clean.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/clean_all.py first.")
        return

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    close = df["close"]
    print(f"Tuning LSTM hyperparameters on {TICKER} ({len(close)} rows)")
    print(f"Validation set: {VAL_DAYS} days | Test set: {TEST_DAYS} days (held out)\n")

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"Grid search: {total} combinations x {TUNING_EPOCHS} max epochs\n")

    results = []
    overall_start = time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        lb = params["look_back"]
        lr = params["learning_rate"]
        bs = params["batch_size"]
        hs = params["hidden_size"]

        print(f"  [{i}/{total}] look_back={lb}, lr={lr}, batch_size={bs}, hidden_size={hs}")

        try:
            start = time.time()

            # Prepare data with this look_back
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_lstm_data(
                close, look_back=lb, test_days=TEST_DAYS, val_days=VAL_DAYS
            )

            # Build and train
            model = build_lstm_model(hidden_size=hs)
            history = train_lstm(
                model, X_train, y_train, X_val, y_val,
                epochs=TUNING_EPOCHS, batch_size=bs, learning_rate=lr,
                verbose=False,
            )

            # Evaluate on validation set
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).to(device)
                val_pred_scaled = model(X_val_t).cpu().numpy()

            val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
            val_actual = close.iloc[-(TEST_DAYS + VAL_DAYS):-TEST_DAYS].values
            metrics = compute_metrics(val_actual, val_pred)

            elapsed = time.time() - start
            row = {**params, **metrics, "best_epoch": history["best_epoch"], "time_s": round(elapsed, 1)}
            results.append(row)
            print(f"           val_RMSE={metrics['rmse']}, val_MAPE={metrics['mape']}%, "
                  f"best_epoch={history['best_epoch']}, {elapsed:.0f}s")

        except Exception as e:
            print(f"           FAILED: {e}")

    # Sort by val MAPE and display
    results_df = pd.DataFrame(results).sort_values("mape")
    results_df.to_csv(tune_dir / "tuning_results.csv", index=False)

    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"TUNING COMPLETE — {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")

    print("All results (ranked by validation MAPE):\n")
    print(results_df.to_string(index=False))

    print(f"\n--- Best Configuration ---")
    best = results_df.iloc[0]
    print(f"  look_back:    {int(best['look_back'])}")
    print(f"  learning_rate: {best['learning_rate']}")
    print(f"  batch_size:   {int(best['batch_size'])}")
    print(f"  hidden_size:  {int(best['hidden_size'])}")
    print(f"  val RMSE:     {best['rmse']}")
    print(f"  val MAE:      {best['mae']}")
    print(f"  val MAPE:     {best['mape']}%")
    print(f"\nSaved to: {tune_dir / 'tuning_results.csv'}")


if __name__ == "__main__":
    main()
