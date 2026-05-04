"""
Barren Plateau Mitigation Strategies for QLSTM.

Two methods implemented:

1. Identity-Block Initialization (Grant et al., 2019)
   Initialises pairs of sub-layers so that adjacent layers cancel each other,
   making the overall unitary start near the identity. This guarantees
   non-vanishing initial gradients, analogous to He/Xavier init in classical DL.

2. Two-Stage Least Squares (Boabang & Gyamerah, 2026)
   Stage 1: Treat the VQC output features as a fixed linear map and solve a
            convex regularised least-squares problem for the downstream
            classical projection weights (good warm start).
   Stage 2: Unfreeze the quantum parameters and fine-tune end-to-end with
            standard gradient descent.
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Strategy 1: Identity-block initialisation
# ============================================================================

def identity_block_init(n_layers, n_qubits, eps=0.01):
    """
    Produce a (n_layers, n_qubits, 3) tensor where consecutive layer pairs
    cancel each other. For an even number of layers, layer 2k+1 is the exact
    negative of layer 2k (up to a small perturbation eps), making their
    composite unitary ≈ I. If n_layers is odd, the last layer is initialised
    small.

    The tiny perturbation eps > 0 is added so gradients are not exactly zero
    (otherwise the optimiser gets stuck at the saddle).
    """
    weights = np.zeros((n_layers, n_qubits, 3), dtype=np.float32)
    for k in range(0, n_layers - 1, 2):
        block = np.random.uniform(-np.pi, np.pi, (n_qubits, 3)).astype(np.float32)
        weights[k] = block
        weights[k + 1] = -block  # inverse rotation
    if n_layers % 2 == 1:
        weights[-1] = np.random.normal(0, eps, (n_qubits, 3)).astype(np.float32)

    # small symmetric perturbation so gradients are non-zero
    weights = weights + np.random.normal(0, eps, weights.shape).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def apply_identity_init_to_qlstm(model, eps=0.01):
    """
    Walk through a QLSTM model and re-initialise every VQC's q_params tensor
    using identity-block initialisation.
    """
    for module in model.modules():
        if hasattr(module, "q_params") and isinstance(module.q_params, nn.Parameter):
            shape = module.q_params.shape  # (n_layers, n_qubits, 3)
            new_w = identity_block_init(shape[0], shape[1], eps=eps)
            with torch.no_grad():
                module.q_params.copy_(new_w)
    return model


# ============================================================================
# Strategy 2: Two-stage least squares
# ============================================================================

def two_stage_ls_warmup(model, X_train, y_train, ridge=1e-2):
    """
    Stage 1 of the two-stage LS mitigation.

    For the given (frozen-quantum) model, compute its pre-projection features
    phi = features before the final linear layer for every training sample.
    Solve a regularised least-squares problem analytically:

        w* = (PhiT Phi + λ I)^{-1} PhiT y

    and copy w* into the model's final linear layer. This gives the network
    an immediate good output mapping **before** touching the quantum params,
    so subsequent gradient descent starts in a smooth basin rather than a
    random flat region of the landscape.

    Assumptions:
      - model has an attribute ``fc_out`` (Linear layer producing final output)
      - model forward can be split so we can extract pre-projection features

    Returns the modified model in-place.
    """
    model.eval()

    # Hook the final linear layer to capture its input
    captured = {}

    def hook(mod, inp, out):
        captured["features"] = inp[0].detach().cpu().numpy()

    handle = model.fc_out.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(X_train)

    handle.remove()

    Phi = captured["features"]                 # (N, hidden_size)
    y = y_train.detach().cpu().numpy()         # (N, output_size)

    # Ridge regression: w = (PhiT Phi + λI)^-1 PhiT y
    d = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(d, dtype=np.float32)
    B = Phi.T @ y
    w_star = np.linalg.solve(A, B)             # (hidden_size, output_size)

    # Bias = mean residual
    bias = (y - Phi @ w_star).mean(axis=0)     # (output_size,)

    with torch.no_grad():
        model.fc_out.weight.copy_(torch.tensor(w_star.T, dtype=torch.float32))
        model.fc_out.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    model.train()
    return model


# ============================================================================
# Strategy 3 (NOVEL): Feature-stability-regularised two-stage LS
# ============================================================================
#
# Motivation. The vanilla two-stage LS warm-up (Boabang & Gyamerah, 2026) fits
# the classical readout to the *current* quantum feature matrix Phi_0 in
# closed form. As Stage 2 proceeds, gradient descent updates the quantum
# parameters and Phi drifts; the carefully-fitted readout becomes misaligned
# and the model overfits the training window. We observed this directly in
# our S&P 500 benchmark (LS warm-up MAE = 2.374 vs random-init MAE = 1.242).
#
# Novel fix. Add a penalty that keeps Phi_t close to Phi_0 during Stage 2:
#
#     L_total = MSE(f(X), y) + lambda * mean( (Phi_t - Phi_0)^2 )
#
# This is a cheap surrogate for the Fisher-information penalty proposed in
# the report's "Future Direction" section. It explicitly ties the quantum
# parameter updates to the feature distribution that the analytic readout
# was solved against, retaining the convex warm-start benefit while guarding
# against the distribution-shift failure mode.
# ============================================================================


def _capture_features(model, X):
    """One frozen forward pass; return the features that flow into fc_out."""
    captured = {}

    def hook(mod, inp, out):
        captured["features"] = inp[0].detach()

    handle = model.fc_out.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(X)
    handle.remove()
    return captured["features"]  # tensor (N, hidden)


def two_stage_ls_novel_warmup(model, X_train, y_train, ridge=1e-2):
    """
    Stage 1 of the NOVEL mitigation. Identical analytic readout solve as the
    vanilla method, but additionally caches the Stage-1 feature matrix Phi_0
    so the caller can use it as a regulariser during Stage 2.

    Returns Phi_0 (torch.Tensor, shape (N, hidden_size)). The model's
    fc_out has been updated in place with the LS solution.
    """
    model.eval()
    captured = {}

    def hook(mod, inp, out):
        captured["features"] = inp[0].detach().cpu().numpy()

    handle = model.fc_out.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(X_train)
    handle.remove()

    Phi = captured["features"]
    y = y_train.detach().cpu().numpy()

    d = Phi.shape[1]
    A = Phi.T @ Phi + ridge * np.eye(d, dtype=np.float32)
    B = Phi.T @ y
    w_star = np.linalg.solve(A, B)
    bias = (y - Phi @ w_star).mean(axis=0)

    with torch.no_grad():
        model.fc_out.weight.copy_(torch.tensor(w_star.T, dtype=torch.float32))
        model.fc_out.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    Phi_0 = torch.tensor(Phi, dtype=torch.float32)
    model.train()
    return Phi_0


def train_with_feature_penalty(model, X, y, Phi_0, epochs, lr,
                                feature_lambda=1.0):
    """
    Stage 2 of the NOVEL mitigation. Standard Adam optimisation with the
    additional feature-stability term lambda * mean((Phi_t - Phi_0)^2).

    Phi_t is captured via a forward hook on model.fc_out at every step, so
    the penalty is differentiable end-to-end through the quantum parameters.

    Returns (training_loss_list, wall_time_seconds). The reported per-epoch
    loss is the MSE on the data ONLY (not including the penalty), so the
    curve is directly comparable to the other mitigations.
    """
    import time
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    captured = {}

    def hook(mod, inp, out):
        captured["features"] = inp[0]  # keep grad — DO NOT detach

    handle = model.fc_out.register_forward_hook(hook)

    losses = []
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        data_loss = loss_fn(pred, y)
        Phi_t = captured["features"]
        feat_pen = ((Phi_t - Phi_0) ** 2).mean()
        total = data_loss + feature_lambda * feat_pen
        total.backward()
        opt.step()
        losses.append(float(data_loss.item()))

    handle.remove()
    return losses, time.time() - t0


if __name__ == "__main__":
    # Quick sanity checks
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from qlstm import QLSTM

    torch.manual_seed(0)
    np.random.seed(0)

    m = QLSTM(input_size=1, hidden_size=4, output_size=1, n_qubits=2, n_layers=2)
    print("Before identity init, sample q_params norm:",
          m.qlstm_cell.vqc_forget.q_params.norm().item())

    apply_identity_init_to_qlstm(m)
    print("After identity init, sample q_params norm:",
          m.qlstm_cell.vqc_forget.q_params.norm().item())

    # Two-stage LS smoke test
    X = torch.randn(5, 3, 1)
    y = torch.randn(5, 1)
    two_stage_ls_warmup(m, X, y, ridge=1e-2)
    print("fc_out weight after LS warmup:", m.fc_out.weight.detach().numpy().flatten())
    print("fc_out bias after LS warmup  :", m.fc_out.bias.item())
