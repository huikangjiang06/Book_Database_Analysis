"""
plot_stability.py — Plot mean HDBSCAN cluster stability across model sizes

For a given model family, finds all out/<family>_<size>/summary.json files,
reads the per-cluster stability scores, and produces a mean ± std bar/scatter
chart with one data point per model size.

Usage:
    python src/plot_stability.py --model_family Qwen3-Embedding
    python src/plot_stability.py --model_family Pythia --out stability_pythia.png
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _size_sort_key(size_str: str) -> float:
    """
    Parse a model-size string like '0.6B', '4B', '70M', '1.5B'
    into a float in a common unit (millions of parameters) for sorting.
    """
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", size_str.strip())
    if not m:
        return float("inf")
    val = float(m.group(1))
    suffix = m.group(2).upper()
    if suffix == "B":
        return val * 1000
    if suffix == "M":
        return val
    if suffix == "K":
        return val / 1000
    return val  # no suffix → treat as-is


def load_stability(model_family: str) -> list[dict]:
    """
    Scan out/<model_family>/*/summary.json and collect stability stats.
    Returns a list of dicts sorted by model size (ascending).
    """
    pattern = os.path.join(OUT_DIR, model_family, "*", "summary.json")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"[error] No summary.json found matching: {pattern}")
        sys.exit(1)

    records = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        stab = data.get("cluster_stability", {})
        if not stab:
            print(f"[warn] {path}: no cluster_stability — run cluster.py first, skipping.")
            continue

        scores = list(stab.values())
        records.append({
            "size":        data["model_size"],
            "n_clusters":  data["n_clusters"],
            "n_noise":     data["n_noise"],
            "scores":      scores,
            "mean":        float(np.mean(scores)),
            "std":         float(np.std(scores, ddof=0)),
            "median":      float(np.median(scores)),
        })

    if not records:
        print("[error] No usable stability data found.")
        sys.exit(1)

    records.sort(key=lambda r: _size_sort_key(r["size"]))
    return records


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_stability(records: list[dict], model_family: str, out_path: str) -> None:
    sizes  = [r["size"]   for r in records]
    means  = [r["mean"]   for r in records]
    stds   = [r["std"]    for r in records]
    n_cl   = [r["n_clusters"] for r in records]
    x      = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(max(5, len(sizes) * 1.6), 4))

    # Bars (mean)
    ax.bar(x, means, width=0.5, color="steelblue", alpha=0.7, label="Mean stability")

    # Error bars (±1 std)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                label="± 1 std")

    # Individual cluster dots with jitter
    for xi, r in enumerate(records):
        jitter = np.random.default_rng(xi).uniform(-0.18, 0.18, len(r["scores"]))
        ax.scatter(xi + jitter, r["scores"],
                   s=16, color="black", alpha=0.35, zorder=4,
                   label="Per-cluster" if xi == 0 else "_nolegend_")

    # k= annotation above each bar
    for xi, r in enumerate(records):
        top = r["mean"] + r["std"]
        ax.text(xi, top + 0.01, f"k={r['n_clusters']}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("HDBSCAN Cluster Stability")
    ax.set_title(f"{model_family} — Cluster Stability by Model Size")
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out_path}")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_table(records: list[dict], model_family: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {model_family} — Cluster Stability Summary")
    print(f"{'='*62}")
    print(f"  {'Size':>8}  {'k':>4}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in records:
        print(f"  {r['size']:>8}  {r['n_clusters']:>4}  "
              f"{r['mean']:>8.4f}  {r['std']:>8.4f}  {r['median']:>8.4f}")
    print(f"{'='*62}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot HDBSCAN cluster stability across model sizes"
    )
    parser.add_argument("--model_family", required=True,
                        help="e.g. Qwen3-Embedding, Pythia, OpenAI")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: out/<family>_stability.png)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for jitter (reproducibility)")
    args = parser.parse_args()

    records = load_stability(args.model_family)
    print_table(records, args.model_family)

    out_path = args.out or os.path.join(
        OUT_DIR, args.model_family, f"stability.png"
    )
    plot_stability(records, args.model_family, out_path)


if __name__ == "__main__":
    main()
