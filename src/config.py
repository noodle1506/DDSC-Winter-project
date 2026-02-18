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

# Optional — only needed if using the AlphaVantage fetcher
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or None

# Standard directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
