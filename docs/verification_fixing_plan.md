# Verification & Fixing Plan

**Date**: 2026-02-17
**Project**: Stock Market Prediction — ARIMA vs LSTM
**Scope**: Full pipeline verification from data ingestion through model comparison

---

## Executive Summary

A comprehensive verification was performed across the entire pipeline: data fetching, cleaning, EDA, ARIMA modeling, LSTM modeling, and model comparison. The pipeline is structurally sound and functional. However, **12 issues** were identified — 3 critical, 4 moderate, and 5 minor. The most significant problems are ARIMA forecasts diverging to negative values on 7/10 tickers, LSTM summary CSV missing key metrics, and a data consistency mismatch for XOM.

---

## Verification Results

### What Passed

| Component | Status | Details |
|-----------|--------|---------|
| Raw data (10 tickers) | PASS | All 10 CSVs present with correct columns (Date, Open, High, Low, Close, Volume) |
| Cleaned data (10 tickers) | PASS | All 10 CSVs present, 0 NaN close prices, 0 negative close prices |
| Data date alignment | PASS | ARIMA and LSTM test on identical 60-day windows for all tickers |
| EDA outputs | PASS | summary_statistics.csv, 10 overview PNGs, closing_prices_all.png, correlation_heatmap.png all exist |
| ARIMA results CSVs | PASS | All 10 tickers have 60-row results with no NaN in actual or forecast |
| LSTM results CSVs | PASS | All 10 tickers have 60-row results with no NaN in actual or forecast |
| LSTM model .pt files | PASS | All 10 model weight files present (~134 KB each) |
| compute_metrics() | PASS | RMSE, MAE, MAPE calculations verified correct, handles zero-division |
| Comparison CSV | PASS | All 10 tickers present, winners computed correctly |
| Dependency versions | PASS | All packages installed (pandas 3.0, torch 2.10, pmdarima 2.1.1, etc.) |
| ARIMA MAPE consistency | PASS | All 10 tickers' summary MAPE matches recalculated MAPE from results CSV |
| LSTM MAPE consistency (9/10) | PASS | 9 tickers match perfectly between summary and results CSV |

---

## Issues Found

### CRITICAL — Must Fix

#### C1. ARIMA (5,2,0) Forecasts Diverge and Go Negative

**Severity**: CRITICAL
**Affected tickers**: AAPL, GOOGL, JPM, MSFT, NVDA, WMT (6/10 tickers with order `(5,2,0)`)
**Root cause**: `auto_arima` selects `d=2` (double differencing) for most tickers, causing forecasts to follow a polynomial trend that diverges rapidly over a 60-day horizon.

**Evidence**:
| Ticker | Day 1 Error | Day 60 Error | Day 60 Forecast | Day 60 Actual | MAPE |
|--------|------------|-------------|----------------|---------------|------|
| NVDA | $8.51 | $186.97 | **-$1.92** | $185.05 | 51.6% |
| JPM | $6.22 | $203.65 | $103.48 | $307.13 | 36.2% |
| WMT | $0.70 | $51.52 | $77.33 | $128.85 | 22.9% |
| MSFT | $5.25 | $130.78 | $265.93 | $396.71 | 18.7% |

All (5,2,0) forecasts are **monotonically decreasing**, with NVDA actually going to a negative price.

**Fix plan**:
1. In `src/models/arima.py`, constrain `auto_arima` to `max_d=1` to prevent over-differencing:
   ```python
   model = pm.auto_arima(
       train,
       seasonal=False,
       stepwise=True,
       suppress_warnings=True,
       error_action="ignore",
       max_order=10,
       max_d=1,          # ADD: prevent d=2 over-differencing
   )
   ```
2. Add a post-forecast clamp to enforce non-negative predictions:
   ```python
   forecast = np.maximum(forecast, 0)  # Stock prices cannot be negative
   ```
3. Re-run `scripts/run_arima.py` to regenerate all ARIMA outputs.
4. Re-run `scripts/compare_models.py` to regenerate comparison.

---

#### C2. LSTM Summary CSV Missing Validation Metrics and Best Epoch

**Severity**: CRITICAL
**File**: `outputs/lstm/lstm_summary.csv`
**Current columns**: `rmse, mae, mape, symbol`
**Expected columns**: `symbol, rmse, mae, mape, val_rmse, val_mae, val_mape, best_epoch`

**Root cause**: The `run_lstm_pipeline()` function returns all 8 fields, but the summary CSV only contains 4. This happened because the LSTM was originally run with an older version of the code that didn't include validation metrics, and the summary was never regenerated after the code was updated.

**Impact**:
- Cannot assess overfitting (no val vs test comparison available in summary)
- best_epoch information is lost — cannot verify early stopping effectiveness
- The LSTM report references validation metrics that aren't actually saved

**Fix plan**:
1. Re-run `scripts/run_lstm.py` to regenerate `lstm_summary.csv` with all columns.
2. This will also regenerate the missing loss curve plots and checkpoint files (see issue M1).

---

#### C3. XOM LSTM Data Inconsistency

**Severity**: CRITICAL
**File**: `outputs/lstm/lstm_summary.csv` and `outputs/lstm/XOM_lstm_results.csv`

**Evidence**:
- Summary says: RMSE=3.8886, MAE=2.6415, MAPE=1.9846%
- Recalculated from results CSV: MAPE=6.2859%
- Discrepancy: **4.30 percentage points**

**Root cause**: XOM was re-run separately (it's the only ticker with a loss curve and checkpoint file). The summary CSV was updated with the new metrics, but the results CSV was also regenerated with different forecasts. The metrics in the summary appear to be from a different run than the saved results CSV, or the model produced better metrics during training but the saved model produced different test predictions.

**Fix plan**:
1. Re-run the full LSTM pipeline (`scripts/run_lstm.py`) to regenerate all outputs consistently.
2. Verify XOM summary MAPE matches recalculated MAPE from its results CSV.

---

### MODERATE — Should Fix

#### M1. Missing LSTM Loss Curve PNGs (9/10 tickers)

**Severity**: MODERATE
**Missing files**: `{TICKER}_loss_curve.png` for AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA, WMT
**Present**: Only `XOM_loss_curve.png` exists

**Root cause**: Same as C2 — the initial LSTM run used code that didn't generate loss curves. Only XOM's re-run generated the plot.

**Impact**: Cannot visually diagnose overfitting/underfitting for 9 tickers. The LSTM report (docs/lstm_report.md Section 5) claims these plots exist for every ticker.

**Fix plan**: Resolved by re-running `scripts/run_lstm.py` (same as C2).

---

#### M2. Missing LSTM Checkpoint Files (9/10 tickers)

**Severity**: MODERATE
**Missing files**: `outputs/lstm/checkpoints/{TICKER}_best.pt` for all tickers except XOM

**Impact**: Cannot reload the best early-stopped model weights for 9 tickers. Only the final model .pt file exists.

**Fix plan**: Resolved by re-running `scripts/run_lstm.py` (same as C2).

---

#### M3. LSTM Predictions Are Near-Constant for 3 Tickers (High MAPE)

**Severity**: MODERATE
**Affected tickers**: AAPL, GOOGL, WMT

**Evidence**:
| Ticker | Actual Range | Forecast Range | Forecast Std | MAPE |
|--------|-------------|---------------|-------------|------|
| AAPL | [246.70, 286.19] | [206.71, 215.78] | **2.77** | 20.9% |
| GOOGL | [289.45, 343.69] | [232.06, 249.84] | **4.45** | 23.5% |
| WMT | [100.61, 133.89] | [88.34, 96.16] | **2.00** | 20.7% |

The LSTM is predicting near-constant values significantly below the actual prices for these tickers. This suggests the model is under-predicting — likely because the test period prices are well above the training data range, and the MinMaxScaler's [0,1] mapping becomes inaccurate.

**Root cause**: This is Limitation #7 in the LSTM report — when test prices exceed the training range, MinMaxScaler produces out-of-range scaled values, and the model's inverse-transform produces systematically low predictions. These tickers likely had strong upward trends in the most recent 60 days.

**Fix plan** (recommended, not strictly required):
1. Consider using a rolling or expanding window scaler instead of fitting MinMaxScaler only on training data
2. OR use log-returns instead of raw prices for the LSTM input (scale-invariant)
3. OR add a note in the comparison analysis documenting this known limitation
4. At minimum, document this behavior in the final report

---

#### M4. `fetch_daily.py` Missing sys.path Setup

**Severity**: MODERATE
**File**: `scripts/fetch_daily.py`

**Current code** (line 7-8):
```python
from src.config import ALPHAVANTAGE_API_KEY
from src.fetch.alphavantage import fetch_time_series_daily, save_raw_json
```

Unlike all other scripts in `scripts/`, this file does NOT have the `sys.path.insert(0, ...)` setup. This means running `python scripts/fetch_daily.py` from the repo root will fail with `ModuleNotFoundError: No module named 'src'` unless PYTHONPATH is manually set.

**Fix plan**:
Add the standard path setup at the top of the file:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

---

### MINOR — Nice to Fix

#### m1. ARIMA statsmodels Warnings

**Severity**: MINOR
**Warning**: `ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at 'start'.`
**Warning**: `FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.`

**Root cause**: The training series passed to `auto_arima` has a DatetimeIndex, but pmdarima/statsmodels doesn't detect it as a "supported" frequency index (because stock market dates have irregular gaps for weekends/holidays).

**Fix plan**: Either:
1. Set `series.index.freq = 'B'` (business day) before passing to auto_arima, OR
2. Reset the index to a RangeIndex: `train = train.reset_index(drop=True)` before fitting. This is cosmetic — results are correct despite the warning.

---

#### m2. First Row `returns` is Always NaN

**Severity**: MINOR
**File**: All processed CSV files

The first row of every cleaned CSV has `returns=NaN` because `pct_change()` cannot compute a return for the very first data point. This is mathematically correct and harmless — all downstream code handles it properly (e.g., `dropna()` in EDA). No fix needed, but worth noting.

---

#### m3. WMT `close_min` Shown as 0.0 in EDA Summary

**Severity**: MINOR
**File**: `outputs/eda/summary_statistics.csv`

WMT's `close_min` is displayed as `0.0` due to rounding (`round(..., 2)`) on a value of $0.00245. The actual minimum is a valid historical split-adjusted price from 1973.

**Fix plan**: Optionally increase decimal precision in `summary_statistics()` for `close_min`:
```python
"close_min": round(close.min(), 4),  # was round(..., 2)
```

---

#### m4. No Random Seed Set for LSTM Reproducibility

**Severity**: MINOR
**File**: `src/models/lstm.py`

As noted in the LSTM report (Limitation #6), training results vary between runs due to random weight initialization and shuffled batches. No seed is set.

**Fix plan** (optional):
Add at the start of `run_lstm_pipeline()`:
```python
torch.manual_seed(42)
np.random.seed(42)
```

---

#### m5. Comparison Overlay Plots Only Cover 3 Tickers

**Severity**: MINOR
**File**: `scripts/compare_models.py` (line 107-108)

The script only generates overlay plots for the 3 tickers with the highest MAPE difference. For a complete analysis, overlays for all 10 tickers would be more useful.

**Fix plan** (optional): Change `top3 = merged.nlargest(3, "mape_diff")` to loop over all tickers, or make it configurable.

---

## Recommended Fix Execution Order

| Step | Action | Fixes | Estimated Time |
|------|--------|-------|----------------|
| 1 | Add `max_d=1` and non-negative clamp to `src/models/arima.py` | C1 | 2 min |
| 2 | Add `sys.path.insert` to `scripts/fetch_daily.py` | M4 | 1 min |
| 3 | (Optional) Add `torch.manual_seed(42)` to `src/models/lstm.py` | m4 | 1 min |
| 4 | Re-run `scripts/run_arima.py` | C1 | ~5 min |
| 5 | Re-run `scripts/run_lstm.py` | C2, C3, M1, M2 | ~15-30 min (CPU) |
| 6 | Re-run `scripts/compare_models.py` | Updates comparison with new results | ~1 min |
| 7 | Verify XOM MAPE consistency after re-run | C3 | 1 min |
| 8 | (Optional) Fix WMT close_min rounding and re-run EDA | m3 | 2 min |

**Total estimated time**: ~25-40 minutes (dominated by LSTM training)

---

## Summary Table

| ID | Severity | Component | Issue | Fix Complexity |
|----|----------|-----------|-------|----------------|
| C1 | CRITICAL | ARIMA | Forecasts diverge negative (d=2) | Easy (1-line config change + re-run) |
| C2 | CRITICAL | LSTM | Summary CSV missing val metrics | Easy (re-run pipeline) |
| C3 | CRITICAL | LSTM/XOM | Summary vs results CSV mismatch | Easy (re-run pipeline) |
| M1 | MODERATE | LSTM | 9/10 loss curve PNGs missing | Fixed by C2 re-run |
| M2 | MODERATE | LSTM | 9/10 checkpoint files missing | Fixed by C2 re-run |
| M3 | MODERATE | LSTM | Near-constant predictions (3 tickers) | Medium (scaler/architecture change) |
| M4 | MODERATE | fetch_daily | Missing sys.path setup | Easy (1-line add) |
| m1 | MINOR | ARIMA | statsmodels index warnings | Easy (cosmetic) |
| m2 | MINOR | Data | First row returns=NaN | No fix needed |
| m3 | MINOR | EDA | WMT close_min rounds to 0.0 | Easy (precision tweak) |
| m4 | MINOR | LSTM | No random seed | Easy (1-line add) |
| m5 | MINOR | Comparison | Only 3 overlay plots | Easy (loop change) |
