"""
Barren Plateau Analysis for VQC sub-circuits in QLSTM
Based on: Larocca et al. (2024) and McClean et al. (2018)

Measures gradient variance across random initializations to detect
barren plateaus as a function of qubit count and circuit depth.
"""

import numpy as np
import pennylane as qml
import torch
import matplotlib.pyplot as plt
import os
import json


def create_vqc_circuit(n_qubits, n_layers):
    """Create a VQC circuit matching the QLSTM sub-circuit architecture."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Strongly entangling layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # Global cost: expectation of sum of PauliZ
        return qml.expval(qml.PauliZ(0))

    return circuit


def compute_gradient_variance(n_qubits, n_layers, n_samples=200):
    """
    Compute variance of gradients over random parameter initializations.

    Args:
        n_qubits: number of qubits
        n_layers: number of VQC layers
        n_samples: number of random initializations

    Returns:
        dict with gradient variance statistics
    """
    circuit = create_vqc_circuit(n_qubits, n_layers)

    gradients = []

    for i in range(n_samples):
        # Random input
        inputs = torch.randn(n_qubits, dtype=torch.float64) * 0.5

        # Random parameters (uniform in [0, 2pi])
        weights = torch.tensor(
            np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3)),
            dtype=torch.float64,
            requires_grad=True,
        )

        # Forward pass
        result = circuit(inputs, weights)

        # Backward pass to get gradients
        result.backward()

        # Collect gradient of first parameter as representative
        grad = weights.grad.detach().numpy().flatten()
        gradients.append(grad)

        if (i + 1) % 50 == 0:
            print(f"  n={n_qubits}, L={n_layers}: {i+1}/{n_samples} samples done")

    gradients = np.array(gradients)  # (n_samples, n_params)

    # Compute statistics per parameter
    mean_grad = np.mean(gradients, axis=0)
    var_grad = np.var(gradients, axis=0)

    # Overall statistics
    result = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_samples": n_samples,
        "mean_gradient_variance": float(np.mean(var_grad)),
        "max_gradient_variance": float(np.max(var_grad)),
        "min_gradient_variance": float(np.min(var_grad)),
        "mean_gradient_mean": float(np.mean(np.abs(mean_grad))),
        "all_variances": var_grad.tolist(),
    }

    return result


def run_analysis(qubit_range=None, layer_range=None, n_samples=200, output_dir="plots"):
    """Run full BP analysis across configurations."""
    if qubit_range is None:
        qubit_range = [2, 4]
    if layer_range is None:
        layer_range = [1, 2]

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for n_layers in layer_range:
        for n_qubits in qubit_range:
            print(f"\nAnalyzing n_qubits={n_qubits}, n_layers={n_layers}...")
            result = compute_gradient_variance(n_qubits, n_layers, n_samples)
            results.append(result)
            print(f"  Mean gradient variance: {result['mean_gradient_variance']:.6f}")

    # Save raw results
    results_path = os.path.join(output_dir, "bp_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot 1: Gradient variance vs number of qubits (one line per layer count)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for n_layers in layer_range:
        layer_results = [r for r in results if r["n_layers"] == n_layers]
        qubits = [r["n_qubits"] for r in layer_results]
        variances = [r["mean_gradient_variance"] for r in layer_results]
        ax.plot(qubits, variances, "o-", label=f"L={n_layers} layers", markersize=8, linewidth=2)

    ax.set_xlabel("Number of Qubits (n)", fontsize=13)
    ax.set_ylabel("Mean Gradient Variance", fontsize=13)
    ax.set_title("Barren Plateau Analysis: Gradient Variance vs Qubit Count", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qubit_range)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "gradient_variance_vs_qubits.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Plot 2: Gradient variance heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    var_matrix = np.zeros((len(layer_range), len(qubit_range)))
    for r in results:
        li = layer_range.index(r["n_layers"])
        qi = qubit_range.index(r["n_qubits"])
        var_matrix[li, qi] = r["mean_gradient_variance"]

    im = ax.imshow(var_matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(qubit_range)))
    ax.set_xticklabels([str(q) for q in qubit_range])
    ax.set_yticks(range(len(layer_range)))
    ax.set_yticklabels([str(l) for l in layer_range])
    ax.set_xlabel("Number of Qubits (n)")
    ax.set_ylabel("Number of Layers (L)")
    ax.set_title("Gradient Variance Heatmap")

    for i in range(len(layer_range)):
        for j in range(len(qubit_range)):
            ax.text(j, i, f"{var_matrix[i,j]:.4f}", ha="center", va="center", fontsize=11)

    plt.colorbar(im, ax=ax, label="Gradient Variance")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "gradient_variance_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Barren Plateau Analysis for QLSTM VQC Sub-circuits")
    print("=" * 60)

    results = run_analysis(
        qubit_range=[2, 4],
        layer_range=[1, 2],
        n_samples=200,
        output_dir="/raid/home/vikram_govt/Dikshant/gautam/iqc/plots",
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for r in results:
        print(f"  n={r['n_qubits']}, L={r['n_layers']}: "
              f"Var(grad) = {r['mean_gradient_variance']:.6f}")
