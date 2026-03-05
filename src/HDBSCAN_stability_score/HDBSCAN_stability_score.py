"""
HDBSCAN_stability_score.py
===========================
Reads pre-generated HDBSCAN cluster stability scores from
  out/<family>/<size>/summary.json
and plots mean ± std per model size for a given model family.

For each model size the ``cluster_stability`` dict in summary.json holds one
persistence score per cluster (keyed by cluster id).  This script aggregates
those scores across all sizes and produces:

  results.json          — collected stats per model size
  hdbscan_stability.png — bar chart (mean ± std + per-cluster jitter dots)

All outputs go under:
  out/HDBSCAN_stability_score/<family>/

Usage:
    python src/HDBSCAN_stability_score/HDBSCAN_stability_score.py \\
        --model_family Qwen3-Embedding
    python src/HDBSCAN_stability_score/HDBSCAN_stability_score.py \\
        --model_family Pythia
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
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(ROOT, "out")
BASE_OUT = os.path.join(OUT_DIR, "HDBSCAN_stability_score")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _size_sort_key(size_str: str) -> float:
    """
    Parse a model-size string like '0.6B', '4B', '70M', '1.5B',
    'text-embedding-3-small' into a comparable float (in millions of params).
    Unrecognised strings sort to the end.
    """
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", size_str.strip())
    if not m:
        return float("inf")
    val    = float(m.group(1))
    suffix = m.group(2).upper()
    if suffix == "B":
        return val * 1000
    if suffix == "M":
        return val
    if suffix == "K":
        return val / 1000
    return val


def load_stability(model_family: str) -> list[dict]:
    """
    Scan out/<model_family>/*/summary.json and collect per-cluster stability
    scores for every model size.

    Returns a list of size-dicts sorted ascending by parameter count.
    Each dict has keys:
        size, n_clusters, n_noise, scores, mean, std, median
    """
    pattern = os.path.join(OUT_DIR, model_family, "*", "summary.json")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        print(f"[error] No summary.json found matching: {pattern}")
        print(f"        Run cluster.py first for each size in {model_family}.")
        sys.exit(1)

    records = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        stab = data.get("cluster_stability", {})
        if not stab:
            print(f"[warn]  {path}: 'cluster_stability' missing or empty — "
                  f"skipping (was cluster.py run with the version that saves it?)")
            continue

        scores = list(stab.values())
        records.append({
            "size":       data.get("model_size", os.path.basename(os.path.dirname(path))),
            "n_clusters": data.get("n_clusters", len(scores)),
            "n_noise":    data.get("n_noise", 0),
            "scores":     scores,
            "mean":       float(np.mean(scores)),
            "std":        float(np.std(scores, ddof=0)),
            "median":     float(np.median(scores)),
            "min":        float(np.min(scores)),
            "max":        float(np.max(scores)),
        })

    if not records:
        print("[error] No usable stability data found after filtering.")
        sys.exit(1)

    records.sort(key=lambda r: _size_sort_key(r["size"]))
    return records


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_stability(records: list[dict], model_family: str, out_path: str) -> None:
    sizes  = [r["size"]       for r in records]
    means  = [r["mean"]       for r in records]
    stds   = [r["std"]        for r in records]
    n_cl   = [r["n_clusters"] for r in records]
    x      = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(max(5, len(sizes) * 1.6), 4.5))
    fig.suptitle(
        f"{model_family} — HDBSCAN Cluster Stability Across Model Sizes\n"
        f"(N = number of clusters per size)",
        fontsize=12,
    )

    # Bars (mean stability)
    ax.bar(x, means, width=0.5, color="steelblue", alpha=0.7,
           label="Mean stability")

    # Error bars (±1 std)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                label="± 1 std")

    # Per-cluster dots with horizontal jitter
    for xi, r in enumerate(records):
        jitter = np.random.default_rng(xi).uniform(-0.18, 0.18, len(r["scores"]))
        ax.scatter(xi + jitter, r["scores"],
                   s=16, color="black", alpha=0.35, zorder=4,
                   label="Per-cluster" if xi == 0 else "_nolegend_")

    # k=N annotation above each bar
    for xi, r in enumerate(records):
        top = r["mean"] + r["std"]
        ylim = ax.get_ylim()
        ax.text(xi, top + (ylim[1] - ylim[0]) * 0.02,
                f"k={r['n_clusters']}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("HDBSCAN Cluster Stability")
    ax.set_title(f"{model_family} — Cluster Persistence Score by Model Size",
                 fontsize=11)
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
    print(f"\n{'='*68}")
    print(f"  {model_family} — HDBSCAN Cluster Stability Summary")
    print(f"{'='*68}")
    print(f"  {'Size':>26}  {'k':>4}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"  {'-'*26}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in records:
        print(f"  {r['size']:>26}  {r['n_clusters']:>4}  "
              f"{r['mean']:>8.4f}  {r['std']:>8.4f}  {r['median']:>8.4f}")
    print(f"{'='*68}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot HDBSCAN cluster stability scores across model sizes"
    )
    parser.add_argument(
        "--model_family", required=True,
        help="Model family name, e.g. Qwen3-Embedding, Pythia, Cerebras-GPT, "
             "Qwen2.5, OpenAI",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for jitter reproducibility (default: 0)",
    )
    args = parser.parse_args()

    out_dir  = os.path.join(BASE_OUT, args.model_family)
    json_out = os.path.join(out_dir, "results.json")
    plot_out = os.path.join(out_dir, "hdbscan_stability.png")

    # ── Load ──────────────────────────────────────────────────────────────────
    records = load_stability(args.model_family)

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(records, args.model_family)

    # ── Save JSON (no per-cluster arrays to keep it compact) ─────────────────
    os.makedirs(out_dir, exist_ok=True)
    save = {
        "family": args.model_family,
        "n_sizes": len(records),
        "sizes": [r["size"] for r in records],
        "per_size": [
            {k: v for k, v in r.items() if k != "scores"}
            for r in records
        ],
    }
    with open(json_out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"[save] Summary → {json_out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    plot_stability(records, args.model_family, plot_out)


if __name__ == "__main__":
    main()
