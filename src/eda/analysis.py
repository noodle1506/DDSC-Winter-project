"""EDA functions: summary statistics and visualizations."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def summary_statistics(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute summary statistics for each ticker's close price and returns."""
    rows = []
    for symbol, df in dfs.items():
        close = df["close"]
        ret = df["returns"].dropna()
        rows.append({
            "symbol": symbol,
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "trading_days": len(df),
            "close_mean": round(close.mean(), 2),
            "close_std": round(close.std(), 2),
            "close_min": round(close.min(), 2),
            "close_max": round(close.max(), 2),
            "return_mean": round(ret.mean() * 100, 4),
            "return_std": round(ret.std() * 100, 4),
            "return_skew": round(ret.skew(), 4),
            "return_kurtosis": round(ret.kurtosis(), 4),
        })
    return pd.DataFrame(rows)


def plot_closing_prices(dfs: dict[str, pd.DataFrame], out_dir: Path,
                        last_n_years: int = 5) -> None:
    """Plot all tickers' closing prices normalized to 100, last N years only."""
    fig, ax = plt.subplots(figsize=(14, 7))
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=last_n_years)

    for symbol, df in dfs.items():
        subset = df[df.index >= cutoff]["close"]
        if len(subset) < 2:
            continue
        normalized = subset / subset.iloc[0] * 100
        ax.plot(normalized.index, normalized.values, label=symbol, linewidth=1.2)

    ax.set_title(f"Normalized Closing Prices (Last {last_n_years} Years, Base=100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "closing_prices_all.png", dpi=150)
    plt.close(fig)


def plot_individual_ticker(df: pd.DataFrame, symbol: str, out_dir: Path,
                           last_n_years: int = 5) -> None:
    """Generate a 2x2 overview plot for a single ticker."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=last_n_years)
    recent = df[df.index >= cutoff].copy()
    if len(recent) < 10:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{symbol} — Overview (Last {last_n_years} Years)", fontsize=14)

    # Top-left: Close price
    axes[0, 0].plot(recent.index, recent["close"], linewidth=1)
    axes[0, 0].set_title("Close Price")
    axes[0, 0].set_ylabel("USD")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Returns distribution
    ret = recent["returns"].dropna()
    axes[0, 1].hist(ret, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[0, 1].set_title("Daily Returns Distribution")
    axes[0, 1].set_xlabel("Return")
    axes[0, 1].axvline(0, color="red", linestyle="--", alpha=0.5)

    # Bottom-left: 30-day rolling volatility
    rolling_vol = ret.rolling(30).std() * np.sqrt(252) * 100
    axes[1, 0].plot(rolling_vol.index, rolling_vol.values, linewidth=1)
    axes[1, 0].set_title("30-Day Rolling Volatility (Annualized %)")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Volume
    if "volume" in recent.columns:
        axes[1, 1].bar(recent.index, recent["volume"], width=1.5, alpha=0.6)
        axes[1, 1].set_title("Daily Volume")
        axes[1, 1].set_ylabel("Shares")
    else:
        axes[1, 1].text(0.5, 0.5, "Volume data not available",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{symbol}_overview.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(dfs: dict[str, pd.DataFrame], out_dir: Path,
                             last_n_years: int = 5) -> None:
    """Plot a correlation heatmap of daily returns across tickers."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=last_n_years)

    returns = {}
    for symbol, df in dfs.items():
        subset = df[df.index >= cutoff]["returns"].dropna()
        returns[symbol] = subset

    returns_df = pd.DataFrame(returns).dropna()
    if returns_df.empty:
        return

    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Daily Returns Correlation Heatmap (Last 5 Years)")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
