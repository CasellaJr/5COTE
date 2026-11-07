#!/usr/bin/env python3
"""
Shows what each client actually sees under the CONTIGUOUS and the SHUFFLED
vertical split: one column per split, one row per client, each panel plots
the variables held by that client (same split_data_vertical / same rng seed as
vertical.py, so the shuffled assignment is the one used in the experiments).

    python plot_split_comparison.py --ds_name NATOPS --n_clients 4 --seed 0
    python plot_split_comparison.py --ds_name NATOPS --mode classmean --class_idx 2

--mode sample     : one single series (default, --sample_idx)
--mode classmean  : mean series of one class (smoother, --class_idx = index in sorted labels)
Each variable is z-normalised for display only, so that variables with very
different scales are comparable; the data given to the models is untouched.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4169E1", "#E08A00", "#2E8B57", "#A0282D", "#7B4FA3", "#2AA3A3", "#B58900", "#666666"]


def plot_split_comparison(X, Y, ds_name, n_clients=4, seed=0, mode="sample",
                          sample_idx=0, class_idx=0, out_dir="plots"):
    from utils import split_data_vertical  # same function used by the experiments

    n, d, T = X.shape
    if mode == "classmean":
        cls = np.unique(Y)[class_idx]
        series = X[Y == cls].mean(axis=0)           # (d, T)
        subtitle = f"mean series of class {cls}"
    else:
        series = X[sample_idx]                      # (d, T)
        subtitle = f"sample {sample_idx}"
    mu, sd = series.mean(axis=1, keepdims=True), series.std(axis=1, keepdims=True) + 1e-8
    series = (series - mu) / sd

    # indices held by each client under the two splits (labels irrelevant here)
    _, _, _, _, idx_cont = split_data_vertical(X, Y, X, Y, n_clients, rng=None, shuffle_dims=False)
    _, _, _, _, idx_shuf = split_data_vertical(X, Y, X, Y, n_clients,
                                               rng=np.random.RandomState(seed), shuffle_dims=True)

    # one figure: top strip = which client holds each variable (as in the TikZ cell figure),
    # below = the signals each client actually receives, for both splits
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(7.2, 1.3 * n_clients + 1.1))
    gs = fig.add_gridspec(n_clients + 1, 2, height_ratios=[0.45] + [1] * n_clients,
                          hspace=0.55, wspace=0.12)
    t = np.arange(T)
    first = {}
    for col, (name, idxs) in enumerate([("Contiguous split", idx_cont), ("Shuffled split", idx_shuf)]):
        owner = np.zeros(d, dtype=int)
        for c in range(n_clients):
            owner[np.asarray(idxs[c], dtype=int)] = c
        ast = fig.add_subplot(gs[0, col])
        for i in range(d):
            ast.add_patch(Rectangle((i, 0), 1, 1, color=COLORS[owner[i] % len(COLORS)], alpha=0.9, lw=0))
            if d <= 30:
                ast.text(i + 0.5, 0.5, str(i + 1), color="white", ha="center", va="center", fontsize=5.5)
        ast.set_xlim(0, d); ast.set_ylim(0, 1); ast.axis("off")
        ast.set_title(name, fontsize=9, weight="bold", pad=3)
        for c in range(n_clients):
            kw = {}
            if first:
                kw = dict(sharex=first["x"], sharey=first["y"])
            ax = fig.add_subplot(gs[c + 1, col], **kw)
            first.setdefault("x", ax); first.setdefault("y", ax)
            for dim in sorted(idxs[c]):
                ax.plot(t, series[dim], color=COLORS[c % len(COLORS)], lw=0.8, alpha=0.8)
            chans = ",".join(str(int(i) + 1) for i in sorted(idxs[c]))
            if len(chans) > 40:
                chans = f"{len(idxs[c])} variables"
            ax.set_title(f"Client {c + 1}: {chans}", fontsize=6.5, loc="left", pad=2)
            ax.tick_params(labelsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if c < n_clients - 1:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.tick_params(labelleft=False)
            if c == n_clients - 1:
                ax.set_xlabel("Timestamps", fontsize=7)
        if col == 0:
            fig.axes[-n_clients // 2].set_ylabel("Normalised value", fontsize=7)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"split_comparison_{ds_name}.pdf"
    fig.savefig(path)          # save BEFORE any show(): show() can close the figure
    plt.close(fig)
    print(f"Saved {path} ({subtitle}; seed {seed})")
    return path


def main():
    from aeon.datasets import load_classification
    from utils import preprocess_data, safe_convert_labels
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_name", default="NATOPS")
    ap.add_argument("--extract_path", default=None)
    ap.add_argument("--n_clients", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["sample", "classmean"], default="sample")
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--class_idx", type=int, default=0)
    ap.add_argument("--out_dir", default="plots")
    a = ap.parse_args()
    Xtr, Ytr, _ = load_classification(a.ds_name, split="Train", return_metadata=True, extract_path=a.extract_path)
    Xte, Yte, _ = load_classification(a.ds_name, split="Test", return_metadata=True, extract_path=a.extract_path)
    Ytr, Yte = safe_convert_labels(Ytr, Yte)
    Xtr, Ytr, Xte, Yte = preprocess_data(Xtr, Ytr, Xte, Yte)
    plot_split_comparison(Xtr, np.asarray(Ytr), a.ds_name, a.n_clients, a.seed, a.mode,
                          a.sample_idx, a.class_idx, a.out_dir)


if __name__ == "__main__":
    main()