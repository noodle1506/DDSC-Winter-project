# src/fetch/alphavantage.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class FetchResponse:
    """Response from Alpha Vantage API fetch."""

    raw: dict[str, Any]
    symbol: str
    function: str
    outputsize: str


def fetch_time_series_daily(
    symbol: str,
    apikey: str,
    outputsize: str = "compact",
    prefer_adjusted: bool = True,
    pause_seconds: float = 12.0,
) -> FetchResponse:
    """
    Fetch daily time series data from Alpha Vantage API.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        apikey: Alpha Vantage API key
        outputsize: 'compact' (last 100 data points) or 'full' (20+ years)
        prefer_adjusted: If True, use TIME_SERIES_DAILY_ADJUSTED; else use TIME_SERIES_DAILY
        pause_seconds: Seconds to pause before making request (rate limiting)

    Returns:
        FetchResponse with raw data, symbol, function, and outputsize
    """
    if pause_seconds > 0:
        time.sleep(pause_seconds)

    function = "TIME_SERIES_DAILY_ADJUSTED" if prefer_adjusted else "TIME_SERIES_DAILY"
    base_url = "https://www.alphavantage.co/query"

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": apikey,
        "outputsize": outputsize,
        "datatype": "json",
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    raw = response.json()

    return FetchResponse(
        raw=raw,
        symbol=symbol,
        function=function,
        outputsize=outputsize,
    )


def save_raw_json(raw: dict[str, Any], filepath: str) -> None:
    """
    Save raw JSON data to a file.

    Args:
        raw: JSON data as dictionary
        filepath: Path to save the JSON file
    """
    with open(filepath, "w") as f:
        json.dump(raw, f, indent=2)

