"""Clean raw Stooq data for all tickers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR
from src.tickers import TICKERS
from src.clean.prices_daily import clean_daily, save_processed


def main():
    raw_dir = RAW_DIR / "stooq"
    total = len(TICKERS)
    summaries = []

    print(f"Cleaning data for {total} tickers...\n")

    for i, symbol in enumerate(TICKERS, 1):
        raw_path = raw_dir / f"{symbol}_daily.csv"
        if not raw_path.exists():
            print(f"  [{i}/{total}] {symbol}: SKIPPED — raw file not found")
            continue

        df_raw = pd.read_csv(raw_path)
        df_clean = clean_daily(df_raw, symbol)
        path = save_processed(df_clean, symbol, PROCESSED_DIR)

        nan_count = df_clean["close"].isna().sum()
        date_range = f"{df_clean.index.min().date()} to {df_clean.index.max().date()}"
        summaries.append({
            "symbol": symbol,
            "rows": len(df_clean),
            "date_range": date_range,
            "nan_close": nan_count,
        })
        print(f"  [{i}/{total}] {symbol}: {len(df_clean)} rows ({date_range}) -> {path.name}")

    print(f"\nDone. Cleaned data saved to: {PROCESSED_DIR}")

    if summaries:
        print("\n--- Summary ---")
        summary_df = pd.DataFrame(summaries)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
