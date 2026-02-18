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
    val_days: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Create scaled sliding-window sequences with a 3-way split.

    Split strategy (time-based, no leakage):
      - Test:       last `test_days` rows
      - Validation:  `val_days` rows before test
      - Train:       everything before validation

    The scaler is fit ONLY on training data.
    Returns X_train, y_train, X_val, y_val, X_test, y_test, scaler.
    """
    values = series.values.reshape(-1, 1)
    n = len(values)
    test_start = n - test_days
    val_start = test_start - val_days
    train_end = val_start

    # Fit scaler on train portion only
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values)

    # Build sequences
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Split — account for the look_back offset
    train_size = train_end - look_back
    val_size = val_days

    X_train = X[:train_size].reshape(-1, look_back, 1)
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size].reshape(-1, look_back, 1)
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :].reshape(-1, look_back, 1)
    y_test = y[train_size + val_size :]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


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


def build_lstm_model(hidden_size: int = 50, dropout: float = 0.2) -> LSTMModel:
    """Build and return an LSTM model."""
    return LSTMModel(input_size=1, hidden_size=hidden_size, dropout=dropout)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(
    model: LSTMModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 5,
    checkpoint_dir: Path | None = None,
    symbol: str = "",
    verbose: bool = True,
) -> dict:
    """Train the LSTM model with early stopping and optional checkpointing.

    Returns a dict with 'train_losses', 'val_losses', and 'best_epoch'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train)
    X_v = torch.from_numpy(X_val).to(device)
    y_v = torch.from_numpy(y_val).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Train ---
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
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

        # --- Early stopping + checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save checkpoint to disk
            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"{symbol}_best.pt"
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "val_loss": best_val_loss,
                }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_lstm_pipeline(
    df: pd.DataFrame,
    symbol: str,
    look_back: int = 60,
    test_days: int = 60,
    val_days: int = 60,
    hidden_size: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    out_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Full LSTM pipeline: prepare data, train, predict, evaluate, save outputs."""
    close = df["close"]

    min_required = look_back + val_days + test_days + 60
    if len(close) < min_required:
        raise ValueError(f"{symbol}: not enough data ({len(close)} rows, need {min_required})")

    # Prepare data (3-way split)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_lstm_data(
        close, look_back=look_back, test_days=test_days, val_days=val_days
    )

    # Build and train
    model = build_lstm_model(hidden_size=hidden_size)
    history = train_lstm(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir, symbol=symbol, verbose=verbose,
    )

    # Predict on test set
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        pred_scaled = model(X_test_t).cpu().numpy()

    # Inverse transform back to original scale
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = close.iloc[-test_days:].values

    # Also get validation predictions for reporting
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(device)
        val_pred_scaled = model(X_val_t).cpu().numpy()
    val_predictions = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual = close.iloc[-(test_days + val_days):-test_days].values

    # Metrics
    test_metrics = compute_metrics(actual, predictions)
    val_metrics = compute_metrics(val_actual, val_predictions)

    metrics = {
        "symbol": symbol,
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "mape": test_metrics["mape"],
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_mape": val_metrics["mape"],
        "best_epoch": history["best_epoch"],
    }

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

        # Save final model
        torch.save(model.state_dict(), out_dir / f"{symbol}_lstm_model.pt")

        # Forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))
        tail_train = close.iloc[-(120 + test_days + val_days):-(test_days + val_days)]
        val_index = df.index[-(test_days + val_days):-test_days]
        ax.plot(tail_train.index, tail_train.values, label="Train (last 120 days)", color="blue")
        ax.plot(val_index, val_actual, label="Validation", color="orange")
        ax.plot(test_index, actual, label="Actual (test)", color="green")
        ax.plot(test_index, predictions, label="LSTM Forecast", color="red", linestyle="--")
        ax.set_title(f"{symbol} — LSTM Forecast (look_back={look_back})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol}_lstm_forecast.png", dpi=150)
        plt.close(fig)

        # Loss curve plot
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs_range = range(1, len(history["train_losses"]) + 1)
        ax.plot(epochs_range, history["train_losses"], label="Train Loss")
        ax.plot(epochs_range, history["val_losses"], label="Validation Loss")
        ax.axvline(history["best_epoch"], color="red", linestyle=":", alpha=0.7,
                    label=f"Best Epoch ({history['best_epoch']})")
        ax.set_title(f"{symbol} — Training Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol}_loss_curve.png", dpi=150)
        plt.close(fig)

    return metrics
