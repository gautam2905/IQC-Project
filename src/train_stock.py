"""
Full stock price benchmark for QLSTM on S&P 500.

Trains and compares four configurations:
  1. Classical LSTM (baseline)
  2. QLSTM — random init
  3. QLSTM — identity-block init (Grant et al.)
  4. QLSTM — two-stage LS warmup (Boabang & Gyamerah)

Produces:
  - Loss curves (plots/stock_loss_curves.png)
  - Prediction overlay (plots/stock_predictions.png)
  - JSON metrics (plots/stock_metrics.json)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from qlstm import QLSTM
from classical_lstm import ClassicalLSTM
from mitigations import apply_identity_init_to_qlstm, two_stage_ls_warmup
from data_loader import load_sp500_dataset


def train(model, X, y, epochs=20, lr=0.01, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] params={n_params}")
    t0 = time.time()
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (e + 1) % 5 == 0 or e == 0:
            print(f"  epoch {e+1:3d}/{epochs}  loss={loss.item():.6f}  "
                  f"elapsed={time.time()-t0:.1f}s")
    return losses, time.time() - t0, n_params


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy().flatten()
        true = y.numpy().flatten()
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(mse))
    return pred, true, {"mse": mse, "mae": mae, "rmse": rmse}


def main():
    out_dir = "/raid/home/vikram_govt/Dikshant/gautam/iqc/plots"
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)

    # Load S&P 500
    X_tr_full, y_tr_full, X_te_full, y_te_full, (mu, sd) = load_sp500_dataset(seq_len=10)
    # Use a subset because quantum simulation is slow
    N_TRAIN, N_TEST = 80, 40
    X_tr = X_tr_full[:N_TRAIN]; y_tr = y_tr_full[:N_TRAIN]
    X_te = X_te_full[:N_TEST];  y_te = y_te_full[:N_TEST]
    print(f"Data: train={X_tr.shape[0]}, test={X_te.shape[0]}, "
          f"price_mean={mu:.1f}, price_std={sd:.1f}")

    EPOCHS = 20

    # --- Classical LSTM baseline ---
    classical = ClassicalLSTM(input_size=1, hidden_size=8, output_size=1)
    cls_loss, cls_time, cls_params = train(classical, X_tr, y_tr,
                                           epochs=EPOCHS, lr=0.02,
                                           name="Classical LSTM")
    cls_pred, cls_true, cls_metrics = evaluate(classical, X_te, y_te)

    # --- QLSTM random init (baseline) ---
    np.random.seed(0); torch.manual_seed(0)
    qlstm_rand = QLSTM(input_size=1, hidden_size=4, output_size=1,
                       n_qubits=2, n_layers=1)
    q_rand_loss, q_rand_time, q_rand_params = train(
        qlstm_rand, X_tr, y_tr, epochs=EPOCHS, lr=0.02,
        name="QLSTM (random init)")
    q_rand_pred, _, q_rand_metrics = evaluate(qlstm_rand, X_te, y_te)

    # --- QLSTM identity-block init ---
    np.random.seed(0); torch.manual_seed(0)
    qlstm_id = QLSTM(input_size=1, hidden_size=4, output_size=1,
                     n_qubits=2, n_layers=1)
    apply_identity_init_to_qlstm(qlstm_id, eps=0.05)
    q_id_loss, q_id_time, q_id_params = train(
        qlstm_id, X_tr, y_tr, epochs=EPOCHS, lr=0.02,
        name="QLSTM (identity-block init)")
    q_id_pred, _, q_id_metrics = evaluate(qlstm_id, X_te, y_te)

    # --- QLSTM two-stage least squares ---
    np.random.seed(0); torch.manual_seed(0)
    qlstm_ls = QLSTM(input_size=1, hidden_size=4, output_size=1,
                     n_qubits=2, n_layers=1)
    print("\n[QLSTM + two-stage LS] Stage 1: convex LS warm-up ...")
    two_stage_ls_warmup(qlstm_ls, X_tr, y_tr, ridge=1e-2)
    q_ls_loss, q_ls_time, q_ls_params = train(
        qlstm_ls, X_tr, y_tr, epochs=EPOCHS, lr=0.02,
        name="QLSTM (two-stage LS)")
    q_ls_pred, _, q_ls_metrics = evaluate(qlstm_ls, X_te, y_te)

    # --- Save JSON metrics ---
    summary = {
        "classical_lstm": {"params": cls_params, "time_s": round(cls_time, 1),
                           **cls_metrics},
        "qlstm_random":   {"params": q_rand_params, "time_s": round(q_rand_time, 1),
                           **q_rand_metrics},
        "qlstm_identity": {"params": q_id_params, "time_s": round(q_id_time, 1),
                           **q_id_metrics},
        "qlstm_two_stage_ls": {"params": q_ls_params, "time_s": round(q_ls_time, 1),
                               **q_ls_metrics},
        "train_samples": N_TRAIN, "test_samples": N_TEST, "epochs": EPOCHS,
    }
    with open(os.path.join(out_dir, "stock_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Final Stock Price Benchmark (S&P 500)")
    print("=" * 60)
    print(f"{'Model':<26} {'MAE':>10} {'RMSE':>10} {'Params':>8} {'Time(s)':>10}")
    print("-" * 66)
    for key, label in [
        ("classical_lstm",    "Classical LSTM"),
        ("qlstm_random",      "QLSTM random init"),
        ("qlstm_identity",    "QLSTM identity init"),
        ("qlstm_two_stage_ls","QLSTM two-stage LS"),
    ]:
        s = summary[key]
        print(f"{label:<26} {s['mae']:>10.4f} {s['rmse']:>10.4f} "
              f"{s['params']:>8} {s['time_s']:>10.1f}")

    # --- Plot loss curves ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(cls_loss,     label="Classical LSTM",        linewidth=2)
    ax.plot(q_rand_loss,  label="QLSTM (random init)",   linewidth=2)
    ax.plot(q_id_loss,    label="QLSTM (identity init)", linewidth=2)
    ax.plot(q_ls_loss,    label="QLSTM (two-stage LS)",  linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("S&P 500 Training Loss — Mitigation Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stock_loss_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # --- Plot predictions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cls_true,    "k-",  label="Ground truth",      linewidth=2)
    ax.plot(cls_pred,    "r--", label=f"Classical (MAE={cls_metrics['mae']:.3f})")
    ax.plot(q_rand_pred, "b--", label=f"QLSTM rand (MAE={q_rand_metrics['mae']:.3f})")
    ax.plot(q_id_pred,   "g--", label=f"QLSTM identity (MAE={q_id_metrics['mae']:.3f})")
    ax.plot(q_ls_pred,   "m--", label=f"QLSTM LS (MAE={q_ls_metrics['mae']:.3f})")
    ax.set_xlabel("Time Step (test set)")
    ax.set_ylabel("Normalised Price")
    ax.set_title("S&P 500 Prediction — QLSTM vs Classical")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stock_predictions.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"\nPlots + metrics saved to {out_dir}/")


if __name__ == "__main__":
    main()
