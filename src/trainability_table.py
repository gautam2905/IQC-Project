"""
Generate a trainability guidelines table from BP analysis results.

Rule of thumb:
  variance > 1e-2       ->  "Trainable"  (gradients clearly non-vanishing)
  1e-4 <= variance <=1e-2 -> "Marginal"  (training slow but feasible)
  variance < 1e-4       ->  "Barren"     (exponentially small gradients)

The thresholds are chosen relative to single-precision optimiser tolerance
and consistent with empirical guidance from Larocca et al. (2024).
"""

import json
import os


def classify(variance):
    if variance >= 1e-2:
        return "Trainable"
    elif variance >= 1e-4:
        return "Marginal"
    else:
        return "Barren"


def main():
    plots_dir = "/raid/home/vikram_govt/Dikshant/gautam/iqc/plots"
    results_path = os.path.join(plots_dir, "bp_full_grid.json")
    out_path = os.path.join(plots_dir, "trainability_table.json")
    tex_path = os.path.join(plots_dir, "trainability_table.tex")

    with open(results_path) as f:
        results = json.load(f)

    qubits = sorted(set(r["n_qubits"] for r in results))
    layers = sorted(set(r["n_layers"] for r in results))

    table = {}
    for L in layers:
        table[L] = {}
        for n in qubits:
            r = [x for x in results if x["n_qubits"] == n and x["n_layers"] == L][0]
            table[L][n] = {"variance": r["variance"], "status": classify(r["variance"])}

    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)

    # Emit a LaTeX tabular snippet
    col_spec = "c" + "c" * len(qubits)
    lines = [r"\begin{tabular}{" + col_spec + "}",
             r"\toprule",
             " & ".join(["Layers $L \\backslash$ Qubits $n$"] + [f"${n}$" for n in qubits]) + r" \\",
             r"\midrule"]
    for L in layers:
        row = [f"$L={L}$"]
        for n in qubits:
            v = table[L][n]["variance"]
            status = table[L][n]["status"]
            row.append(f"{v:.2e} ({status})")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    # Print for console
    print("Trainability Table")
    print("=" * 60)
    print(f"{'Layers':<8}", end="")
    for n in qubits:
        print(f"n={n:<14}", end="")
    print()
    print("-" * 60)
    for L in layers:
        print(f"L={L:<6}", end="")
        for n in qubits:
            c = table[L][n]
            print(f"{c['variance']:.2e} ({c['status']:<9})", end=" ")
        print()
    print(f"\nJSON saved to {out_path}")
    print(f"LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
