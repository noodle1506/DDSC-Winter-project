from __future__ import annotations

from pathlib import Path
import pandas as pd


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """Download daily OHLCV data for a US stock from Stooq."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    df = pd.read_csv(url)
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"No data returned for {symbol}. Check if the ticker is valid.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def save_raw_csv(df: pd.DataFrame, symbol: str, out_dir: Path) -> Path:
    """Save a raw DataFrame to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_daily.csv"
    df.to_csv(out_path, index=False)
    return out_path
