"""
Stock price data loader for QLSTM training.
Downloads S&P 500 daily closing prices via yfinance and creates sliding-window sequences.
"""

import os
import numpy as np
import torch
import yfinance as yf


def download_sp500(start="2020-01-01", end="2024-01-01", cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sp500_{start}_{end}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    ticker = yf.Ticker("^GSPC")
    df = ticker.history(start=start, end=end, auto_adjust=True)
    prices = df["Close"].to_numpy().astype(np.float32)
    np.save(cache_path, prices)
    return prices


def make_sequences(series, seq_len=10):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def load_sp500_dataset(seq_len=10, train_ratio=0.8, normalize=True):
    """
    Returns:
        X_train, y_train, X_test, y_test, scaler_info
    scaler_info: (mean, std) used for denormalisation later
    """
    prices = download_sp500()

    if normalize:
        mean = float(prices.mean())
        std = float(prices.std())
        prices_n = (prices - mean) / std
    else:
        mean, std = 0.0, 1.0
        prices_n = prices

    X, y = make_sequences(prices_n, seq_len=seq_len)
    n_train = int(len(X) * train_ratio)

    X_train = torch.tensor(X[:n_train])
    y_train = torch.tensor(y[:n_train])
    X_test = torch.tensor(X[n_train:])
    y_test = torch.tensor(y[n_train:])

    return X_train, y_train, X_test, y_test, (mean, std)


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, (mu, sd) = load_sp500_dataset(seq_len=10)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Price mean={mu:.2f}, std={sd:.2f}")
    print(f"Sample train input: {X_tr[0].flatten().numpy()}")
    print(f"Sample train target: {y_tr[0].item():.4f}")
