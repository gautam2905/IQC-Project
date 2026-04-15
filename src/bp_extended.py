"""
Extended Barren Plateau Analysis: n = 2 to 20 qubits
Measures gradient variance to clearly show exponential decay.
"""

import numpy as np
import pennylane as qml
import torch
import matplotlib.pyplot as plt
import os
import json
import time


def compute_gradient_variance(n_qubits, n_layers, n_samples=100):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    gradients_first_param = []

    for i in range(n_samples):
        inputs = torch.randn(n_qubits, dtype=torch.float64) * 0.5
        weights = torch.tensor(
            np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3)),
            dtype=torch.float64,
            requires_grad=True,
        )
        result = circuit(inputs, weights)
        result.backward()
        # Collect gradient of first parameter only (representative)
        grad_val = weights.grad[0, 0, 0].item()
        gradients_first_param.append(grad_val)

    gradients_first_param = np.array(gradients_first_param)
    variance = float(np.var(gradients_first_param))
    mean = float(np.mean(gradients_first_param))

    return {"n_qubits": n_qubits, "n_layers": n_layers, "n_samples": n_samples,
            "gradient_variance": variance, "gradient_mean": mean}


def main():
    output_dir = "/raid/home/vikram_govt/Dikshant/gautam/iqc/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Qubit range — adaptive samples for larger circuits
    configs = [
        # (n_qubits, n_samples)
        (2, 200), (4, 200), (6, 200), (8, 150),
        (10, 100), (12, 100), (14, 80), (16, 60), (18, 50), (20, 40),
    ]

    layer_range = [1, 2]
    all_results = []

    for n_layers in layer_range:
        print(f"\n{'='*60}")
        print(f"  Layers L = {n_layers}")
        print(f"{'='*60}")
        for n_qubits, n_samples in configs:
            print(f"\n  n={n_qubits} qubits, {n_samples} samples ...", end=" ", flush=True)
            t0 = time.time()
            result = compute_gradient_variance(n_qubits, n_layers, n_samples)
            elapsed = time.time() - t0
            result["time_seconds"] = round(elapsed, 1)
            all_results.append(result)
            print(f"Var={result['gradient_variance']:.6e}  ({elapsed:.1f}s)")

    # Save results
    results_path = os.path.join(output_dir, "bp_extended_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Plot: Gradient variance vs qubits (log scale) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for n_layers in layer_range:
        layer_results = [r for r in all_results if r["n_layers"] == n_layers]
        qubits = [r["n_qubits"] for r in layer_results]
        variances = [r["gradient_variance"] for r in layer_results]
        ax.plot(qubits, variances, "o-", label=f"L={n_layers} layers", markersize=7, linewidth=2)

    # Add theoretical O(1/2^n) reference line
    q_ref = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # Scale to match the L=1 data at n=2
    l1_at_2 = [r["gradient_variance"] for r in all_results if r["n_layers"] == 1 and r["n_qubits"] == 2][0]
    theoretical = l1_at_2 * (2**2) / (2**q_ref.astype(float))
    ax.plot(q_ref, theoretical, "k--", alpha=0.4, linewidth=1.5, label=r"$O(1/2^n)$ reference")

    ax.set_xlabel("Number of Qubits (n)", fontsize=13)
    ax.set_ylabel(r"$\mathrm{Var}[\partial L / \partial \theta]$", fontsize=13)
    ax.set_title("Barren Plateau Analysis: Gradient Variance vs Qubit Count", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bp_extended_gradient_variance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Qubits':>8} {'L=1 Var':>14} {'L=2 Var':>14}")
    print(f"{'-'*60}")
    qubit_list = sorted(set(r["n_qubits"] for r in all_results))
    for n in qubit_list:
        v1 = [r["gradient_variance"] for r in all_results if r["n_qubits"] == n and r["n_layers"] == 1]
        v2 = [r["gradient_variance"] for r in all_results if r["n_qubits"] == n and r["n_layers"] == 2]
        v1_str = f"{v1[0]:.6e}" if v1 else "N/A"
        v2_str = f"{v2[0]:.6e}" if v2 else "N/A"
        print(f"{n:>8} {v1_str:>14} {v2_str:>14}")


if __name__ == "__main__":
    main()
