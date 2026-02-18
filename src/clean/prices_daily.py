"""Clean and preprocess daily OHLCV stock data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def clean_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean a raw daily OHLCV DataFrame.

    Steps:
      1. Standardize column names to lowercase.
      2. Parse date, set as index, sort ascending.
      3. Remove duplicate dates.
      4. Forward-fill then back-fill small gaps.
      5. Drop rows where close is still NaN.
      6. Validate no negative prices.
      7. Add a daily returns column.
    """
    df = df.copy()

    # Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse and set date index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Fill small gaps
    df = df.ffill().bfill()

    # Drop if close is still missing
    df = df.dropna(subset=["close"])

    # Validate
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and (df[col] < 0).any():
            bad = (df[col] < 0).sum()
            print(f"  WARNING [{symbol}]: {bad} negative values in '{col}' — dropping them")
            df = df[df[col] >= 0]

    # Daily returns
    df["returns"] = df["close"].pct_change()

    return df


def save_processed(df: pd.DataFrame, symbol: str, out_dir: Path) -> Path:
    """Save cleaned DataFrame to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_daily_clean.csv"
    df.to_csv(out_path)
    return out_path
