"""
Launches the full mitigation x (n,L) grid in parallel.

Configs covered (all those for which we have BP gradient-variance data and
that fit inside a ~2.4 hr budget per single QLSTM run):

    (n_qubits, n_layers) in {(2,1), (2,2), (2,4), (4,1), (4,2)}
    mitigation in {random, identity, ls, ls_novel}

The fourth mitigation (`ls_novel`) is our own contribution: a feature-
stability-regularised variant of two-stage LS that adds
    lambda * mean((Phi_t - Phi_0)^2)
to the training loss to prevent the quantum features from drifting away
from the analytic readout fitted in Stage 1. See src/mitigations.py.

= 20 QLSTM jobs + 2 classical baselines = 22 processes spawned simultaneously.
Each worker is pinned to 1 CPU thread; with 256 cores on this box there is
no contention. Total wall time ~= the slowest single job, which is the
(n=4, L=2) QLSTM at an estimated ~2.4 hrs.

Outputs:
    plots/grid_results/<tag>.json     # per-job metrics
    plots/grid_results/<tag>.log      # captured stdout
    plots/grid_results/summary.json   # aggregated dict
    plots/grid_results/grid_table.tex # LaTeX snippet \input'd by final_report.tex
"""

import os
import sys
import json
import time
import subprocess

ROOT = "/raid/home/vikram_govt/Dikshant/gautam/iqc"
WORKER = os.path.join(ROOT, "src", "train_grid_worker.py")
RESULTS_DIR = os.path.join(ROOT, "plots", "grid_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use the conda env that has pennylane/torch/yfinance installed.
PY = "/raid/home/vikram_govt/anaconda3/envs/ai/bin/python"
if not os.path.exists(PY):
    PY = sys.executable

CONFIGS = [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2)]
MITIGATIONS = ["random", "identity", "ls", "ls_novel"]

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"


def spawn(cmd, tag):
    log_path = os.path.join(RESULTS_DIR, f"{tag}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return tag, proc, log


def main():
    procs = []

    # Two classical baselines (cheap; ~30 s each)
    for h in (8, 4):
        tag = f"classical_h{h}"
        out = os.path.join(RESULTS_DIR, f"{tag}.json")
        cmd = [PY, WORKER,
               "--model", "classical", "--hidden-size", str(h),
               "--tag", tag, "--out", out]
        procs.append(spawn(cmd, tag))

    # QLSTM grid x mitigations
    for n, L in CONFIGS:
        for m in MITIGATIONS:
            tag = f"qlstm_n{n}_L{L}_{m}"
            out = os.path.join(RESULTS_DIR, f"{tag}.json")
            cmd = [PY, WORKER,
                   "--model", "qlstm",
                   "--n-qubits", str(n), "--n-layers", str(L),
                   "--mitigation", m,
                   "--tag", tag, "--out", out]
            procs.append(spawn(cmd, tag))

    print(f"Launched {len(procs)} processes in parallel.")
    print(f"Configs: {CONFIGS}")
    print(f"Mitigations: {MITIGATIONS}")
    print(f"Slowest job estimate: (n=4, L=2) ~ 2.4 hrs.")
    print(f"Per-job logs: {RESULTS_DIR}/<tag>.log")
    print()

    t0 = time.time()
    for tag, p, log in procs:
        rc = p.wait()
        log.close()
        elapsed = (time.time() - t0) / 60
        flag = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{elapsed:6.1f} min]  {tag:<32} {flag}")

    total_min = (time.time() - t0) / 60
    print(f"\nAll jobs finished in {total_min:.1f} min.")

    # Aggregate
    results = {}
    for tag, _, _ in procs:
        path = os.path.join(RESULTS_DIR, f"{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[tag] = json.load(f)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Build LaTeX snippet
    write_latex_table(results)

    # Console summary
    print("\n" + "=" * 78)
    print(f"{'Config':<32} {'MAE':>10} {'RMSE':>10} {'Params':>8} {'Time(s)':>10}")
    print("-" * 78)
    for tag in sorted(results):
        r = results[tag]
        print(f"{tag:<32} {r['mae']:>10.4f} {r['rmse']:>10.4f} "
              f"{r['params']:>8} {r['train_time_s']:>10.1f}")


def write_latex_table(results):
    """Emit a LaTeX table of MAE per (n,L,mitigation) plus a parallel-time row."""
    rows = []
    rows.append(r"\begin{center}")
    rows.append(r"\small")
    rows.append(r"\begin{tabular}{@{}cc|cccc|c@{}}")
    rows.append(r"\toprule")
    rows.append(r"\textbf{n} & \textbf{L} & "
                r"\textbf{Random} & \textbf{Identity} & \textbf{LS (vanilla)} & "
                r"\textbf{LS (novel)} & \textbf{$\Delta$ (novel$-$LS)} \\")
    rows.append(r"\midrule")
    for n, L in [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2)]:
        cells = []
        for m in ("random", "identity", "ls", "ls_novel"):
            tag = f"qlstm_n{n}_L{L}_{m}"
            r = results.get(tag)
            cells.append(f"{r['mae']:.3f}" if r else "--")
        ls = results.get(f"qlstm_n{n}_L{L}_ls")
        nv = results.get(f"qlstm_n{n}_L{L}_ls_novel")
        delta = (f"{nv['mae'] - ls['mae']:+.3f}"
                 if (ls and nv) else "--")
        rows.append(f"{n} & {L} & {cells[0]} & {cells[1]} & {cells[2]} & "
                    f"{cells[3]} & {delta} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{center}")

    # Wall-time line
    qlstm_times = [r["train_time_s"] for k, r in results.items()
                   if k.startswith("qlstm_")]
    if qlstm_times:
        slowest = max(qlstm_times) / 60.0
        rows.append("")
        rows.append(rf"\noindent\textbf{{Parallel wall time:}} "
                    rf"{len(qlstm_times)} QLSTM jobs ran concurrently; total wall "
                    rf"clock = slowest single job = {slowest:.1f} min.")

    out_path = os.path.join(RESULTS_DIR, "grid_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nLaTeX snippet written to {out_path}")


if __name__ == "__main__":
    main()
