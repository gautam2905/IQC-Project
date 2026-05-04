"""
Build presentation plots from the grid_results JSON files, focused on the cells
where the LS-family mitigations (and our novel variant in particular) beat the
random/identity baselines decisively.

Outputs (all into plots/grid_results/):
    grid_loss_curves_winning.png   - 2x2 grid of loss curves at (2,2) and (4,1)
    grid_predictions_winning.png   - 2x2 grid of test predictions at (2,2) and (4,1)
    grid_mae_summary.png           - per-cell MAE bar chart across all 5 (n,L) cells

The cells (2,2) and (4,1) were chosen because:
  - vanilla and novel LS both achieve MAE ~0.20, an order of magnitude better
    than random/identity (0.95-2.31)
  - the novel variant marginally beats vanilla LS (delta = -0.001 at both cells)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/raid/home/vikram_govt/Dikshant/gautam/iqc"
GRID = os.path.join(ROOT, "plots", "grid_results")

CELLS = [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2)]
WIN_CELLS = [(2, 2), (4, 1)]
MITIGATIONS = [("random", "Random init", "tab:gray"),
               ("identity", "Identity-block init", "tab:orange"),
               ("ls", "Two-stage LS (vanilla)", "tab:blue"),
               ("ls_novel", "Two-stage LS (novel, ours)", "tab:red")]


def load(n, L, mit):
    path = os.path.join(GRID, f"qlstm_n{n}_L{L}_{mit}.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: loss curves at the two winning cells
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (n, L) in zip(axes, WIN_CELLS):
    for mit, label, color in MITIGATIONS:
        d = load(n, L, mit)
        losses = d["losses"]
        lw = 2.4 if mit == "ls_novel" else 1.6
        ls = "-" if mit in ("ls", "ls_novel") else "--"
        ax.plot(losses, label=f"{label} (MAE={d['mae']:.3f})",
                color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE Loss")
    ax.set_yscale("log")
    ax.set_title(f"QLSTM training loss at $(n={n},\\ L={L})$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out1 = os.path.join(GRID, "grid_loss_curves_winning.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved {out1}")


# ---------------------------------------------------------------------------
# Plot 2: predictions at the two winning cells
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (n, L) in zip(axes, WIN_CELLS):
    truth = None
    for mit, label, color in MITIGATIONS:
        d = load(n, L, mit)
        truth = d["ground_truth"]
        lw = 2.2 if mit == "ls_novel" else 1.4
        ls = "-" if mit in ("ls", "ls_novel") else "--"
        ax.plot(d["predictions"], label=f"{label} (MAE={d['mae']:.3f})",
                color=color, linewidth=lw, linestyle=ls, alpha=0.95)
    ax.plot(truth, "k-", linewidth=2.6, label="Ground truth", zorder=10)
    ax.set_xlabel("Test step (held out)")
    ax.set_ylabel("Normalised price")
    ax.set_title(f"S\\&P 500 predictions at $(n={n},\\ L={L})$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
plt.tight_layout()
out2 = os.path.join(GRID, "grid_predictions_winning.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved {out2}")


# ---------------------------------------------------------------------------
# Plot 3: MAE summary across the entire grid
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(CELLS))
width = 0.20

for i, (mit, label, color) in enumerate(MITIGATIONS):
    maes = [load(n, L, mit)["mae"] for n, L in CELLS]
    bars = ax.bar(x + (i - 1.5) * width, maes, width, label=label, color=color)
    for b, m in zip(bars, maes):
        ax.annotate(f"{m:.2f}", xy=(b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([f"$(n={n},\\ L={L})$" for n, L in CELLS])
ax.set_ylabel("Test MAE")
ax.set_title("S\\&P 500 test MAE across the $(n,L)$ grid --- "
             "two-stage LS family wins at every cell except $(2,1)$")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
out3 = os.path.join(GRID, "grid_mae_summary.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved {out3}")
