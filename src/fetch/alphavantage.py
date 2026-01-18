# src/fetch/alphavantage.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class FetchResponse:
    raw: dict[str, Any]
    symbol: str
    function: str
    outputsize: str


def _is_soft_error(raw: dict[str, Any]) -> bool:
    return any(k in raw for k in ("Error Message", "Note", "Information"))


def fetch_time_series_daily(
    symbol: str,
    apikey: str,
    outputsize: str = "compact",
    prefer_adjusted: bool = True,
    pause_seconds: float = 12.0,
    timeout_seconds: float = 30.0,
) -> FetchResponse:
    if pause_seconds > 0:
        time.sleep(pause_seconds)

    base_url = "https://www.alphavantage.co/query"

    def _call(function: str) -> dict[str, Any]:
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": apikey,
            "outputsize": outputsize,
            "datatype": "json",
        }
        resp = requests.get(base_url, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # 1) Try adjusted if preferred
    if prefer_adjusted:
        raw = _call("TIME_SERIES_DAILY_ADJUSTED")
        # if it looks blocked/rate-limited/premium/etc, fallback
        if _is_soft_error(raw) or "Time Series (Daily)" not in raw:
            raw = _call("TIME_SERIES_DAILY")
            function_used = "TIME_SERIES_DAILY"
        else:
            function_used = "TIME_SERIES_DAILY_ADJUSTED"
    else:
        raw = _call("TIME_SERIES_DAILY")
        function_used = "TIME_SERIES_DAILY"

    return FetchResponse(raw=raw, symbol=symbol, function=function_used, outputsize=outputsize)


def save_raw_json(raw: dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
