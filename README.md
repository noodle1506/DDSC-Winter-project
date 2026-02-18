# Stock Market Prediction: ARIMA vs LSTM

A data science project that compares traditional time series forecasting (ARIMA) with deep learning (LSTM) for predicting stock prices across 10 major US equities.

## Tickers

AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA, WMT, XOM

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# For PyTorch CPU (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run each phase individually or run the full pipeline at once.

### Individual phases

```bash
python scripts/fetch_all.py       # 1. Download data from Stooq
python scripts/clean_all.py       # 2. Clean and preprocess
python scripts/eda.py             # 3. Generate EDA plots and stats
python scripts/run_arima.py       # 4. Fit ARIMA models
python scripts/run_lstm.py        # 5. Train LSTM models (~15-30 min)
python scripts/compare_models.py  # 6. Compare ARIMA vs LSTM
```

### Full pipeline (all phases)

```bash
python scripts/run_all.py
```

## Project Structure

```
├── scripts/
│   ├── fetch_all.py          # Data collection from Stooq
│   ├── clean_all.py          # Data cleaning
│   ├── eda.py                # Exploratory data analysis
│   ├── run_arima.py          # ARIMA modeling
│   ├── run_lstm.py           # LSTM modeling
│   ├── compare_models.py     # Model comparison
│   └── run_all.py            # Master pipeline script
├── src/
│   ├── config.py             # Project paths and configuration
│   ├── tickers.py            # Central ticker list
│   ├── fetch/stooq.py        # Stooq API client
│   ├── clean/prices_daily.py # Cleaning logic
│   ├── eda/analysis.py       # EDA functions
│   └── models/
│       ├── evaluate.py       # RMSE, MAE, MAPE metrics
│       ├── arima.py          # ARIMA wrapper (pmdarima)
│       └── lstm.py           # LSTM model (PyTorch)
├── data/
│   ├── raw/stooq/            # Raw downloaded CSVs
│   └── processed/            # Cleaned CSVs
└── outputs/
    ├── eda/                  # EDA plots and summary stats
    ├── arima/                # ARIMA forecasts and plots
    ├── lstm/                 # LSTM forecasts, plots, and saved models
    └── comparison/           # Side-by-side comparison charts
```

## Outputs

After running the pipeline, check the `outputs/` folder:

- `outputs/eda/` — Normalized price charts, per-ticker overviews, correlation heatmap, summary statistics CSV
- `outputs/arima/` — Per-ticker ARIMA forecast plots and CSVs, summary metrics
- `outputs/lstm/` — Per-ticker LSTM forecast plots and CSVs, saved model weights, summary metrics
- `outputs/comparison/` — RMSE and MAPE bar charts, overlay plots, full comparison CSV

## Methodology

- **Data**: Daily OHLCV from Stooq (free API, full history available)
- **Test set**: Last 60 trading days (~3 months), time-based split
- **ARIMA**: Auto-selected (p,d,q) via `pmdarima.auto_arima`
- **LSTM**: 2-layer LSTM (50 units each), dropout 0.2, 60-day look-back window, MinMaxScaler fit on training data only, early stopping with patience=5
- **Metrics**: RMSE, MAE, MAPE
