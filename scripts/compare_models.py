"""Compare ARIMA vs LSTM results side-by-side."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR


def main():
    arima_dir = OUTPUT_DIR / "arima"
    lstm_dir = OUTPUT_DIR / "lstm"
    comp_dir = OUTPUT_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    arima_path = arima_dir / "arima_summary.csv"
    lstm_path = lstm_dir / "lstm_summary.csv"

    if not arima_path.exists():
        print("ERROR: ARIMA summary not found. Run `python scripts/run_arima.py` first.")
        return
    if not lstm_path.exists():
        print("ERROR: LSTM summary not found. Run `python scripts/run_lstm.py` first.")
        return

    arima = pd.read_csv(arima_path)
    lstm = pd.read_csv(lstm_path)

    # Merge on symbol
    merged = arima.merge(lstm, on="symbol", suffixes=("_arima", "_lstm"))

    # Determine winners
    for metric in ["rmse", "mae", "mape"]:
        merged[f"{metric}_winner"] = np.where(
            merged[f"{metric}_arima"] <= merged[f"{metric}_lstm"], "ARIMA", "LSTM"
        )

    # Save full comparison
    merged.to_csv(comp_dir / "model_comparison.csv", index=False)

    # Print comparison table
    display = merged[["symbol",
                       "rmse_arima", "rmse_lstm", "rmse_winner",
                       "mae_arima", "mae_lstm", "mae_winner",
                       "mape_arima", "mape_lstm", "mape_winner"]].copy()

    print("=" * 80)
    print("MODEL COMPARISON: ARIMA vs LSTM")
    print("=" * 80)
    print(display.to_string(index=False))
    print()

    # Summary counts
    for metric in ["rmse", "mae", "mape"]:
        arima_wins = (merged[f"{metric}_winner"] == "ARIMA").sum()
        lstm_wins = (merged[f"{metric}_winner"] == "LSTM").sum()
        print(f"  {metric.upper()}: ARIMA won {arima_wins}/{len(merged)}, "
              f"LSTM won {lstm_wins}/{len(merged)}")

    print()

    # --- Bar charts ---
    symbols = merged["symbol"].tolist()
    x = np.arange(len(symbols))
    width = 0.35

    # RMSE comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, merged["rmse_arima"], width, label="ARIMA", color="#4C72B0")
    ax.bar(x + width / 2, merged["rmse_lstm"], width, label="LSTM", color="#DD8452")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title("RMSE Comparison: ARIMA vs LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(comp_dir / "rmse_comparison.png", dpi=150)
    plt.close(fig)

    # MAPE comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, merged["mape_arima"], width, label="ARIMA", color="#4C72B0")
    ax.bar(x + width / 2, merged["mape_lstm"], width, label="LSTM", color="#DD8452")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE Comparison: ARIMA vs LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(comp_dir / "mape_comparison.png", dpi=150)
    plt.close(fig)

    # --- Overlay plots for 3 most interesting tickers (highest MAPE difference) ---
    merged["mape_diff"] = abs(merged["mape_arima"] - merged["mape_lstm"])
    top3 = merged.nlargest(3, "mape_diff")["symbol"].tolist()

    for symbol in top3:
        arima_results_path = arima_dir / f"{symbol}_arima_results.csv"
        lstm_results_path = lstm_dir / f"{symbol}_lstm_results.csv"

        if not arima_results_path.exists() or not lstm_results_path.exists():
            continue

        arima_res = pd.read_csv(arima_results_path, parse_dates=["date"])
        lstm_res = pd.read_csv(lstm_results_path, parse_dates=["date"])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(arima_res["date"], arima_res["actual"], label="Actual", color="green", linewidth=2)
        ax.plot(arima_res["date"], arima_res["forecast"], label="ARIMA", color="#4C72B0", linestyle="--")
        ax.plot(lstm_res["date"], lstm_res["forecast"], label="LSTM", color="#DD8452", linestyle="--")
        ax.set_title(f"{symbol} — Actual vs ARIMA vs LSTM")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(comp_dir / f"{symbol}_overlay.png", dpi=150)
        plt.close(fig)
        print(f"  Saved overlay plot: {symbol}_overlay.png")

    print(f"\nAll comparison outputs saved to: {comp_dir}")


if __name__ == "__main__":
    main()
