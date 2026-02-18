"""Fetch daily OHLCV data for all tickers from Stooq."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from repo root: python scripts/fetch_all.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR
from src.tickers import TICKERS
from src.fetch.stooq import fetch_stooq_daily, save_raw_csv


def main():
    out_dir = RAW_DIR / "stooq"
    total = len(TICKERS)

    print(f"Fetching daily data for {total} tickers from Stooq...\n")

    for i, symbol in enumerate(TICKERS, 1):
        try:
            df = fetch_stooq_daily(symbol)
            path = save_raw_csv(df, symbol, out_dir)
            date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
            print(f"  [{i}/{total}] {symbol}: {len(df)} rows ({date_range}) -> {path.name}")
        except Exception as e:
            print(f"  [{i}/{total}] {symbol}: FAILED - {e}")

        if i < total:
            time.sleep(2)

    print(f"\nDone. Raw data saved to: {out_dir}")


if __name__ == "__main__":
    main()
