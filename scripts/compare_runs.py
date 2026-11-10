"""Compare training curves of several runs, grouped by run type (the run dir name before "-s<seed>").

    uv run --extra viz python scripts/compare_runs.py outputs/intrinsic_ppo/task-*
    uv run --extra viz python scripts/compare_runs.py outputs/intrinsic_ppo/task-* --metric near

Prints mean (min-max over seeds) of the metric at a few points in training, and saves a plot
(compare_<metric>.png) next to the first run's parent directory.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run_dir: Path, metric: str):
    with open(run_dir / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r["steps"]) for r in rows]), np.array([float(r[metric]) for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--metric", default="success")
    p.add_argument("--points", type=int, default=8, help="number of checkpoints in the table")
    args = p.parse_args()

    groups = defaultdict(list)
    for run in args.runs:
        if not (run / "metrics.csv").exists():
            print(f"skipping {run}: no metrics.csv")
            continue
        label = re.sub(r"-s\d+-\d{8}-\d{6}$", "", run.name)
        groups[label].append(load(run, args.metric))

    # Align on the shortest run's logging steps.
    steps = min((s for runs in groups.values() for s, _ in runs), key=len)
    idx = np.unique(np.linspace(0, len(steps) - 1, args.points).round().astype(int))
    width = max(len(g) for g in groups) + 9
    print(f"{args.metric} (mean over seeds, [min-max])")
    print("steps".ljust(width) + "".join(f"{steps[i] / 1e6:>19.1f}M" for i in idx))
    curves = {}
    for label, runs in sorted(groups.items()):
        vals = np.stack([v[: len(steps)] for _, v in runs])  # (seeds, T)
        curves[label] = vals
        cells = "".join(f"{vals[:, i].mean():>8.3f} [{vals[:, i].min():.2f}-{vals[:, i].max():.2f}]" for i in idx)
        print(f"{label} (n={len(runs)})".ljust(width) + cells)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, vals in curves.items():
        line = ax.plot(steps / 1e6, vals.mean(0), label=f"{label} (n={len(vals)})")[0]
        ax.fill_between(steps / 1e6, vals.min(0), vals.max(0), color=line.get_color(), alpha=0.2)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel(args.metric)
    ax.legend()
    fig.tight_layout()
    out = args.runs[0].parent / f"compare_{args.metric}.png"
    fig.savefig(out, dpi=120)
    print(f"plot: {out.resolve()}")


if __name__ == "__main__":
    main()
