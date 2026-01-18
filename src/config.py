# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

if not ALPHAVANTAGE_API_KEY:
    raise RuntimeError(
        "Missing ALPHAVANTAGE_API_KEY. Create a .env file with ALPHAVANTAGE_API_KEY=..."
    )
