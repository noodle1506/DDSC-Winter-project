from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root explicitly (repo root)
# src/config.py -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Explicitly load .env from repo root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

if not ALPHAVANTAGE_API_KEY:
    raise RuntimeError(
        f"Missing ALPHAVANTAGE_API_KEY. Expected it in {ENV_PATH}. "
        "Create a .env file with ALPHAVANTAGE_API_KEY=..."
    )
