"""
Full BP analysis over the grid specified in the proposal:
  n ∈ {2, 4, 6} qubits
  L ∈ {1, 2, 4} layers
200 random initialisations per (n, L) configuration.
"""

import os
import json
import time
import numpy as np
import pennylane as qml
import torch
import matplotlib.pyplot as plt


def compute_variance(n_qubits, n_layers, n_samples=200):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    # Collect full gradient vector per sample, then compute per-parameter
    # variance and average across parameters (avoids structural zeros of
    # individual parameters, e.g. a Z-rotation on a |0> state with Z-meas).
    all_grads = []
    for _ in range(n_samples):
        inputs = torch.randn(n_qubits, dtype=torch.float64) * 0.5
        weights = torch.tensor(
            np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3)),
            dtype=torch.float64,
            requires_grad=True,
        )
        out = circuit(inputs, weights)
        out.backward()
        all_grads.append(weights.grad.detach().numpy().flatten())

    G = np.array(all_grads)                      # (n_samples, n_params)
    per_param_var = G.var(axis=0)                # variance across samples
    mean_variance = float(per_param_var.mean())  # average across parameters
    return {"n_qubits": n_qubits, "n_layers": n_layers,
            "variance": mean_variance,
            "max_variance": float(per_param_var.max()),
            "n_samples": n_samples}


def main():
    out_dir = "/raid/home/vikram_govt/Dikshant/gautam/iqc/plots"
    os.makedirs(out_dir, exist_ok=True)

    qubit_range = [2, 4, 6]
    layer_range = [1, 2, 4]
    results = []

    for L in layer_range:
        for n in qubit_range:
            t0 = time.time()
            print(f"Computing n={n}, L={L} ...", end=" ", flush=True)
            r = compute_variance(n, L, n_samples=200)
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
            print(f"Var={r['variance']:.6e}  ({r['time_s']}s)")

    with open(os.path.join(out_dir, "bp_full_grid.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: variance vs qubits for each L
    fig, ax = plt.subplots(figsize=(9, 6))
    for L in layer_range:
        sub = [r for r in results if r["n_layers"] == L]
        ns = [r["n_qubits"] for r in sub]
        vs = [r["variance"] for r in sub]
        ax.plot(ns, vs, "o-", label=f"L={L} layers", markersize=8, linewidth=2)

    # Theoretical O(1/2^n) reference
    n_ref = np.array([2, 4, 6])
    v0 = [r["variance"] for r in results if r["n_layers"] == 1 and r["n_qubits"] == 2][0]
    theo = v0 * (4.0) / (2.0 ** n_ref)
    ax.plot(n_ref, theo, "k--", alpha=0.5, label=r"$O(1/2^n)$ reference")

    ax.set_xlabel("Number of Qubits (n)", fontsize=13)
    ax.set_ylabel(r"Var$[\partial L / \partial \theta]$", fontsize=13)
    ax.set_title("Barren Plateau Analysis: Gradient Variance (full grid)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(qubit_range)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bp_full_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_dir}/bp_full_grid.png")

    # Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    matrix = np.zeros((len(layer_range), len(qubit_range)))
    for r in results:
        li = layer_range.index(r["n_layers"])
        qi = qubit_range.index(r["n_qubits"])
        matrix[li, qi] = r["variance"]
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto",
                   norm=plt.matplotlib.colors.LogNorm())
    ax.set_xticks(range(len(qubit_range)))
    ax.set_xticklabels([str(q) for q in qubit_range])
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels([str(l) for l in layer_range])
    ax.set_xlabel("Qubits (n)")
    ax.set_ylabel("Layers (L)")
    ax.set_title("Gradient Variance Heatmap")
    for i in range(len(layer_range)):
        for j in range(len(qubit_range)):
            ax.text(j, i, f"{matrix[i,j]:.2e}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="Gradient Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bp_full_grid_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSummary:")
    for L in layer_range:
        for n in qubit_range:
            r = [x for x in results if x["n_qubits"] == n and x["n_layers"] == L][0]
            print(f"  n={n}, L={L}: Var = {r['variance']:.6e}")


if __name__ == "__main__":
    main()
