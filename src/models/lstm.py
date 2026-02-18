"""LSTM modeling with PyTorch: data prep, model, training, and evaluation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.models.evaluate import compute_metrics


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_lstm_data(
    series: pd.Series,
    look_back: int = 60,
    test_days: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Create scaled sliding-window sequences for LSTM training.

    The scaler is fit ONLY on training data to avoid data leakage.
    Returns X_train, y_train, X_test, y_test, scaler.
    """
    values = series.values.reshape(-1, 1)
    split = len(values) - test_days

    # Fit scaler on train portion only
    scaler = MinMaxScaler()
    scaler.fit(values[:split])
    scaled = scaler.transform(values)

    # Build sequences
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Split — account for the look_back offset
    train_size = split - look_back
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM: (samples, sequence_length, features=1)
    X_train = X_train.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)

    return X_train, y_train, X_test, y_test, scaler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Two-layer LSTM with dropout for stock price prediction."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take last time step
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


def build_lstm_model(look_back: int = 60) -> LSTMModel:
    """Build and return an LSTM model."""
    return LSTMModel(input_size=1, hidden_size=50, dropout=0.2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(
    model: LSTMModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    verbose: bool = True,
) -> list[float]:
    """Train the LSTM model with early stopping.

    Returns list of epoch losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create data loaders
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)

    # Use last 10% as validation
    val_size = max(1, int(len(X_t) * 0.1))
    X_val, y_val = X_t[-val_size:].to(device), y_t[-val_size:].to(device)
    X_tr, y_tr = X_t[:-val_size], y_t[:-val_size]

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_tr)
        losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return losses


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_lstm_pipeline(
    df: pd.DataFrame,
    symbol: str,
    look_back: int = 60,
    test_days: int = 60,
    epochs: int = 50,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Full LSTM pipeline: prepare data, train, predict, evaluate, save outputs."""
    close = df["close"]

    if len(close) < look_back + test_days + 60:
        raise ValueError(f"{symbol}: not enough data ({len(close)} rows)")

    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(
        close, look_back=look_back, test_days=test_days
    )

    # Build and train
    model = build_lstm_model(look_back)
    train_lstm(model, X_train, y_train, epochs=epochs, verbose=verbose)

    # Predict
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        predictions_scaled = model(X_test_t).cpu().numpy()

    # Inverse transform back to original scale
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    actual = close.iloc[-test_days:].values

    # Metrics
    metrics = compute_metrics(actual, predictions)
    metrics["symbol"] = symbol

    # Save outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Results CSV
        test_index = df.index[-test_days:]
        results = pd.DataFrame({
            "date": test_index,
            "actual": actual,
            "forecast": predictions,
        })
        results.to_csv(out_dir / f"{symbol}_lstm_results.csv", index=False)

        # Save model
        torch.save(model.state_dict(), out_dir / f"{symbol}_lstm_model.pt")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        tail_train = close.iloc[-(120 + test_days):-test_days]
        ax.plot(tail_train.index, tail_train.values, label="Train (last 120 days)", color="blue")
        ax.plot(test_index, actual, label="Actual", color="green")
        ax.plot(test_index, predictions, label="LSTM Forecast", color="red", linestyle="--")
        ax.set_title(f"{symbol} — LSTM Forecast (look_back={look_back})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol}_lstm_forecast.png", dpi=150)
        plt.close(fig)

    return metrics
