"""
Single-job worker for the parallel mitigation grid.

Runs ONE training (classical-LSTM or QLSTM at a given (n_qubits, n_layers,
mitigation)) and writes a JSON result file. Designed to be launched in
parallel by run_grid_parallel.py.

Each worker is pinned to a single CPU thread (OMP_NUM_THREADS=1, torch
threads=1) so 15+ workers running concurrently don't stomp on each other's
BLAS pools.
"""

import os
# Pin thread counts BEFORE importing numpy/torch so BLAS picks them up.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlstm import QLSTM
from classical_lstm import ClassicalLSTM
from mitigations import (
    apply_identity_init_to_qlstm,
    two_stage_ls_warmup,
    two_stage_ls_novel_warmup,
    train_with_feature_penalty,
)
from data_loader import load_sp500_dataset


def train(model, X, y, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, time.time() - t0


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy().flatten()
        true = y.numpy().flatten()
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(mse))
    return pred.tolist(), true.tolist(), {"mse": mse, "mae": mae, "rmse": rmse}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qlstm", "classical"], required=True)
    p.add_argument("--n-qubits", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=4)
    p.add_argument("--mitigation",
                   choices=["random", "identity", "ls", "ls_novel"],
                   default="random")
    p.add_argument("--feature-lambda", type=float, default=1.0,
                   help="Feature-stability penalty weight for ls_novel.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    X_tr_full, y_tr_full, X_te_full, y_te_full, _ = load_sp500_dataset(seq_len=10)
    X_tr = X_tr_full[: args.n_train]
    y_tr = y_tr_full[: args.n_train]
    X_te = X_te_full[: args.n_test]
    y_te = y_te_full[: args.n_test]

    Phi_0 = None
    if args.model == "classical":
        model = ClassicalLSTM(input_size=1, hidden_size=args.hidden_size,
                              output_size=1)
    else:
        model = QLSTM(input_size=1, hidden_size=4, output_size=1,
                      n_qubits=args.n_qubits, n_layers=args.n_layers)
        if args.mitigation == "identity":
            apply_identity_init_to_qlstm(model, eps=0.05)
        elif args.mitigation == "ls":
            two_stage_ls_warmup(model, X_tr, y_tr, ridge=1e-2)
        elif args.mitigation == "ls_novel":
            Phi_0 = two_stage_ls_novel_warmup(model, X_tr, y_tr, ridge=1e-2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[start] {args.tag}  params={n_params}", flush=True)

    if args.mitigation == "ls_novel":
        losses, train_time = train_with_feature_penalty(
            model, X_tr, y_tr, Phi_0,
            epochs=args.epochs, lr=args.lr,
            feature_lambda=args.feature_lambda,
        )
    else:
        losses, train_time = train(model, X_tr, y_tr,
                                   epochs=args.epochs, lr=args.lr)
    pred, true, metrics = evaluate(model, X_te, y_te)

    out = {
        "tag": args.tag,
        "model": args.model,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "mitigation": args.mitigation,
        "feature_lambda": args.feature_lambda,
        "hidden_size": args.hidden_size,
        "params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "train_time_s": round(train_time, 1),
        "losses": losses,
        "predictions": pred,
        "ground_truth": true,
        **metrics,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done]  {args.tag}  MAE={metrics['mae']:.4f}  "
          f"time={train_time/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
