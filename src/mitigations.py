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
