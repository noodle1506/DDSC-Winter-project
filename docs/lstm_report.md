# LSTM Model Report

## 1. Architecture Design

### Model Structure

```
Input (batch, 60, 1)
  -> LSTM Layer 1 (50 hidden units, returns full sequence)
  -> Dropout (0.2)
  -> LSTM Layer 2 (50 hidden units, returns last time step only)
  -> Dropout (0.2)
  -> Dense (25 units, ReLU activation)
  -> Dense (1 unit, linear output)
Output: predicted next-day close price (scaled)
```

### Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input window (look_back) | 60 days | ~3 months of trading data; captures medium-term trends without adding noise from distant history |
| Number of LSTM layers | 2 | Adds depth for learning hierarchical temporal patterns (short-term vs medium-term); more layers risk overfitting on noisy financial data |
| Hidden units | 50 | Balanced capacity — enough to learn price dynamics without overfitting; tuning explored 50 vs 64 |
| Dropout | 0.2 | Standard regularization for sequential models; prevents co-adaptation of hidden units across time steps |
| Dense layer (25 units) | Bottleneck before output | Reduces dimensionality from 50 hidden units, adds a non-linear transformation (ReLU) before the final prediction |
| Output | 1 unit, linear | Regression task — predicting a continuous price value |

### Why LSTM Over Other Architectures

- **vs Simple RNN**: LSTMs solve the vanishing gradient problem, enabling learning from longer sequences (60 days).
- **vs GRU**: LSTMs and GRUs perform comparably on most tasks; LSTM was chosen as the more established baseline in financial literature.
- **vs Transformer**: Transformers require more data and compute; LSTM is simpler to implement, interpret, and tune for a 10-ticker project.

---

## 2. Data Preprocessing Pipeline

### Scaling

- **Method**: MinMaxScaler (scales values to [0, 1])
- **Critical detail**: The scaler is fit ONLY on the training set, then applied to validation and test. This prevents data leakage — the model never "sees" future price ranges during training.
- **Why MinMax over StandardScaler**: LSTM sigmoid/tanh activations work best with bounded inputs in [0, 1].

### Sequence Creation (Sliding Window)

For each time step *t*, the input is a window of the previous `look_back` scaled close prices:

```
X[t] = [price[t-60], price[t-59], ..., price[t-1]]
y[t] = price[t]
```

This creates overlapping sequences. With 10,000+ daily prices per ticker, we get thousands of training samples.

### Train / Validation / Test Split

All splits are **time-based** (no shuffling across time) to prevent look-ahead bias:

```
|---- Train ----|--- Val (60 days) ---|--- Test (60 days) ---|
                                                        today
```

- **Train**: All data before the validation window
- **Validation** (60 days): Used for early stopping and hyperparameter selection
- **Test** (60 days): Held out entirely — only used for final evaluation

The validation set is never used for gradient updates. It solely determines when to stop training and which hyperparameters perform best.

---

## 3. Baseline Training Process

### Optimizer and Loss

- **Optimizer**: Adam (lr=0.001)
  - Adaptive learning rate per parameter
  - Standard choice for deep learning; converges faster than SGD on most tasks
- **Loss function**: Mean Squared Error (MSE)
  - Directly optimizes for prediction accuracy in price units
  - Penalizes large errors more heavily (squared term)

### Training Loop

1. Shuffle training sequences into mini-batches (batch_size=32)
2. Forward pass: predict next-day price from each 60-day window
3. Compute MSE loss between prediction and actual
4. Backpropagate gradients through time (BPTT)
5. Update weights via Adam
6. After each epoch, evaluate on validation set (no gradient computation)
7. Track train loss and validation loss for monitoring

### Batch Size

- **32**: Small enough for noisy gradients (acts as regularization), large enough for stable training.

---

## 4. Hyperparameter Tuning

### Search Space

| Parameter | Values Tested | Default |
|-----------|--------------|---------|
| look_back | 30, 60 | 60 |
| learning_rate | 0.001, 0.0001 | 0.001 |
| batch_size | 32, 64 | 32 |
| hidden_size | 50, 64 | 50 |

**Total**: 16 combinations, each trained for up to 30 epochs with early stopping.

### Method

- **Grid search** on a single representative ticker (AAPL)
- Evaluated on the **validation set only** — the test set was not used for any tuning decisions
- Ranked by validation MAPE (most interpretable metric for comparing across price scales)

### Results (AAPL)

Full results saved to `outputs/lstm/tuning_results.csv`. Top 5 configurations by validation MAPE:

| look_back | lr | batch_size | hidden_size | val RMSE | val MAPE | Best Epoch |
|-----------|------|-----------|-------------|----------|----------|------------|
| 30 | 0.001 | 32 | 64 | 5.62 | 1.74% | 6 |
| 60 | 0.001 | 32 | 64 | 6.02 | 1.82% | 7 |
| 60 | 0.001 | 32 | 50 | 5.93 | 1.85% | 5 |
| 60 | 0.001 | 64 | 64 | 6.00 | 1.87% | 9 |
| 60 | 0.0001 | 32 | 64 | 6.94 | 2.11% | 15 |

**Best configuration**: look_back=30, lr=0.001, batch_size=32, hidden_size=64 (val MAPE=1.74%)

### Key Observations

- **Learning rate is the dominant factor**: lr=0.001 consistently outperforms lr=0.0001 (which undertrains within 30 epochs). The bottom 5 configs are all lr=0.0001.
- **hidden_size=64 slightly outperforms 50**: The top 4 configs all use 64 hidden units.
- **batch_size=32 outperforms 64**: Smaller batches provide noisier but more frequent updates, acting as implicit regularization.
- **look_back=30 vs 60 is close**: The best config uses 30, but 3 of the top 5 use 60. The difference is marginal (~0.1% MAPE), suggesting the model is fairly robust to window size in this range.

---

## 5. Early Stopping and Model Checkpointing

### Early Stopping

- **Patience**: 5 epochs
- **Monitor**: Validation loss (MSE)
- **Behavior**: If validation loss does not improve for 5 consecutive epochs, training stops and the model reverts to the weights from the best epoch
- **Purpose**: Prevents overfitting — financial data is noisy, and training too long causes the model to memorize training noise rather than learn generalizable patterns

### Model Checkpointing

- **Checkpoint saved**: Every time validation loss improves, the full model state is saved to disk at `outputs/lstm/checkpoints/{TICKER}_best.pt`
- **Checkpoint contents**: epoch number, model weights, validation loss
- **Final model**: Also saved separately as `outputs/lstm/{TICKER}_lstm_model.pt`
- **Purpose**: Enables recovery if training is interrupted; allows loading the best model for inference without retraining

### Loss Curve Monitoring

For each ticker, a loss curve plot (`{TICKER}_loss_curve.png`) is saved showing:
- Training loss per epoch
- Validation loss per epoch
- A vertical line marking the best epoch

This enables visual diagnosis of:
- **Overfitting**: validation loss rising while training loss falls
- **Underfitting**: both losses remain high
- **Good fit**: both losses converge and stabilize

---

## 6. Assumptions and Limitations

### Assumptions

1. **Past prices contain predictive signal**: The model assumes that patterns in the last 60 days of close prices carry information about the next day's price. This is a weak form of the "markets are somewhat predictable" hypothesis.

2. **Stationarity not required**: Unlike ARIMA, LSTM does not assume the time series is stationary. The MinMaxScaler normalizes the range, but the model can learn from non-stationary trends.

3. **Univariate input**: Only the close price is used. We assume close price alone captures sufficient dynamics for next-day prediction. Volume, open, high, low, and external factors (news, earnings) are excluded.

4. **No look-ahead bias**: All splits are time-based. The scaler is fit on training data only. No future information leaks into training.

### Limitations

1. **Financial markets are inherently noisy**: Stock prices are influenced by news, earnings, macroeconomic events, and sentiment — none of which are captured by historical price alone. LSTM (and any model) will struggle during unexpected events (e.g., COVID crash, earnings surprises).

2. **Univariate limitation**: Using only close price ignores valuable signals. Adding OHLCV, technical indicators (RSI, MACD), or sentiment data could improve performance but adds complexity.

3. **Fixed look-back window**: The 60-day window may not be optimal for all tickers. Volatile stocks (TSLA, NVDA) may benefit from shorter windows; stable stocks (WMT, XOM) from longer ones.

4. **Point predictions only**: The model outputs a single price, not a confidence interval. In practice, uncertainty quantification (e.g., Monte Carlo dropout) would be needed for risk-aware decisions.

5. **No transaction costs or practical trading**: This is a prediction accuracy study, not a trading strategy. Real-world deployment would need to account for bid-ask spreads, slippage, and transaction costs.

6. **Training is not deterministic**: Due to random weight initialization and shuffled batches, results vary slightly between runs. Setting a random seed would improve reproducibility.

7. **MinMaxScaler sensitivity**: If test-period prices move far outside the training range (e.g., a stock doubles), the scaler's [0,1] mapping becomes inaccurate, degrading predictions.
