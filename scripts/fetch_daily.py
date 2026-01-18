# scripts/fetch_daily.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import ALPHAVANTAGE_API_KEY
from src.fetch.alphavantage import fetch_time_series_daily, save_raw_json


def main() -> None:
    symbol = "AAPL"
    outputsize = "compact"  # start compact, switch to full later

    resp = fetch_time_series_daily(
        symbol=symbol,
        apikey=ALPHAVANTAGE_API_KEY,
        outputsize=outputsize,
        prefer_adjusted=True,
        pause_seconds=12.0,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/raw/alphavantage") / resp.symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{resp.function}_{resp.outputsize}_{ts}.json"
    save_raw_json(resp.raw, str(out_path))

    print(f"Saved raw JSON to: {out_path}")
    print(f"Full path: {out_path.resolve()}")
    print(f"Function used: {resp.function}")


if __name__ == "__main__":
    main()
