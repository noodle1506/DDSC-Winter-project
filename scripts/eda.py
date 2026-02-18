"""Run EDA: generate summary statistics and plots for all tickers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.tickers import TICKERS
from src.eda.analysis import (
    summary_statistics,
    plot_closing_prices,
    plot_individual_ticker,
    plot_correlation_heatmap,
)


def load_all_cleaned() -> dict[str, pd.DataFrame]:
    """Load all cleaned CSVs into a dict keyed by symbol."""
    dfs = {}
    for symbol in TICKERS:
        path = PROCESSED_DIR / f"{symbol}_daily_clean.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        dfs[symbol] = df
    return dfs


def main():
    eda_dir = OUTPUT_DIR / "eda"

    print("Loading cleaned data...")
    dfs = load_all_cleaned()
    print(f"  Loaded {len(dfs)} tickers\n")

    # Summary statistics
    print("Computing summary statistics...")
    stats = summary_statistics(dfs)
    eda_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(eda_dir / "summary_statistics.csv", index=False)
    print(stats.to_string(index=False))
    print()

    # Normalized closing prices
    print("Plotting normalized closing prices...")
    plot_closing_prices(dfs, eda_dir)
    print("  Saved: closing_prices_all.png")

    # Individual ticker overviews
    print("Plotting individual ticker overviews...")
    for symbol, df in dfs.items():
        plot_individual_ticker(df, symbol, eda_dir)
        print(f"  Saved: {symbol}_overview.png")

    # Correlation heatmap
    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(dfs, eda_dir)
    print("  Saved: correlation_heatmap.png")

    print(f"\nDone. All EDA outputs saved to: {eda_dir}")


if __name__ == "__main__":
    main()
