"""
Sine Wave Training: QLSTM vs Classical LSTM
Toy benchmark to verify QLSTM implementation works correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qlstm import QLSTM
from classical_lstm import ClassicalLSTM


def generate_sine_data(n_points=500, seq_len=20, freq=0.05):
    """Generate sine wave sequences for training."""
    t = np.linspace(0, n_points * freq * 2 * np.pi, n_points)
    data = np.sin(t)

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])

    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return torch.tensor(X), torch.tensor(y)


def train_model(model, X_train, y_train, epochs=50, lr=0.01, model_name="Model"):
    """Train a model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model_name}: {n_params} parameters")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"  Training complete in {total_time:.1f}s")

    return loss_history, total_time


def main():
    output_dir = "/raid/home/vikram_govt/Dikshant/gautam/iqc/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    print("Generating sine wave data...")
    seq_len = 10
    X, y = generate_sine_data(n_points=200, seq_len=seq_len, freq=0.1)

    # Use small subset for QLSTM (quantum simulation is slow)
    n_train = 50
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train : n_train + 30]
    y_test = y[n_train : n_train + 30]

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # --- QLSTM ---
    print("\n" + "=" * 50)
    print("Training QLSTM (n_qubits=2, n_layers=1)")
    print("=" * 50)
    qlstm = QLSTM(input_size=1, hidden_size=2, output_size=1, n_qubits=2, n_layers=1)
    qlstm_loss, qlstm_time = train_model(
        qlstm, X_train, y_train, epochs=30, lr=0.01, model_name="QLSTM"
    )

    # --- Classical LSTM ---
    print("\n" + "=" * 50)
    print("Training Classical LSTM")
    print("=" * 50)
    classical = ClassicalLSTM(input_size=1, hidden_size=8, output_size=1)
    classical_loss, classical_time = train_model(
        classical, X_train, y_train, epochs=30, lr=0.01, model_name="Classical LSTM"
    )

    # --- Evaluation ---
    qlstm.eval()
    classical.eval()
    with torch.no_grad():
        qlstm_pred = qlstm(X_test).numpy().flatten()
        classical_pred = classical(X_test).numpy().flatten()
        y_true = y_test.numpy().flatten()

    qlstm_mse = np.mean((qlstm_pred - y_true) ** 2)
    classical_mse = np.mean((classical_pred - y_true) ** 2)
    qlstm_mae = np.mean(np.abs(qlstm_pred - y_true))
    classical_mae = np.mean(np.abs(classical_pred - y_true))

    qlstm_params = sum(p.numel() for p in qlstm.parameters())
    classical_params = sum(p.numel() for p in classical.parameters())

    print("\n" + "=" * 50)
    print("Results Comparison")
    print("=" * 50)
    print(f"{'Metric':<20} {'QLSTM':<15} {'Classical LSTM':<15}")
    print("-" * 50)
    print(f"{'MSE':<20} {qlstm_mse:<15.6f} {classical_mse:<15.6f}")
    print(f"{'MAE':<20} {qlstm_mae:<15.6f} {classical_mae:<15.6f}")
    print(f"{'Parameters':<20} {qlstm_params:<15} {classical_params:<15}")
    print(f"{'Training Time (s)':<20} {qlstm_time:<15.1f} {classical_time:<15.1f}")

    # --- Plot 1: Loss curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(qlstm_loss, label="QLSTM", linewidth=2)
    axes[0].plot(classical_loss, label="Classical LSTM", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("MSE Loss", fontsize=12)
    axes[0].set_title("Training Loss Curves", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Predictions ---
    axes[1].plot(y_true, "k-", label="Ground Truth", linewidth=2)
    axes[1].plot(qlstm_pred, "b--", label=f"QLSTM (MAE={qlstm_mae:.4f})", linewidth=2)
    axes[1].plot(classical_pred, "r--", label=f"Classical (MAE={classical_mae:.4f})", linewidth=2)
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].set_title("Sine Wave Prediction", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "sine_wave_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
