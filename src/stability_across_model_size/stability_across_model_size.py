"""
stability_across_model_size.py
===============================
Measures how stable each book's neighbourhood is as the embedding model scales up.

For every pair of adjacent model sizes (sorted by parameter count) within a
model family, two metrics are computed **per book** (500 books total):

  1. Rank-correlation  (Spearman ρ)
     Rank all N books by cosine similarity to the query book under each model.
     Compute Spearman ρ between the two rank vectors.
     → Captures global similarity-order stability.

  2. Jaccard similarity (top-k neighbour set overlap)
     Find the k nearest neighbours of the query book under each model.
     Jaccard = |A ∩ B| / |A ∪ B|
     → Captures whether the closest books remain the same.

Per pair, report:
  - all 500 per-book scores (saved to JSON)
  - mean ± std
  - a side-by-side bar chart: Rank-corr | Jaccard

Outputs (all under ./out/stability_across_model_size/<family>/):
  results.json     — full per-book scores + summary stats
  stability.png    — bar chart (mean ± std + scatter)

Usage:
    python src/stability_across_model_size.py --model_family Qwen3-Embedding
    python src/stability_across_model_size.py --model_family Qwen3-Embedding --k 20
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
BASE_OUT = os.path.join(ROOT, "out", "stability_across_model_size")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _size_sort_key(s: str) -> float:
    """'0.6B' → 600, '4B' → 4000, '70M' → 70, etc."""
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def load_model_data(family: str, size: str) -> tuple[np.ndarray, list[str]]:
    """Load (N, D) embeddings and book title list from raw per-book pkl files."""
    base  = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No .pkl files found under {base}. "
            f"Check --model_family / model_size names."
        )
    embeddings, titles = [], []
    for path in paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        emb = d.get("embedding")
        if emb is None:
            emb = d.get("book_embedding")
        if emb is None:
            print(f"  [warn] No embedding in {path}, skipping.")
            continue
        embeddings.append(np.array(emb, dtype=np.float32))
        titles.append(d.get("book_title", os.path.splitext(os.path.basename(path))[0]))
    emb = np.stack(embeddings)           # (N, D)
    # L2-normalise so dot product = cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms, titles


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine similarity matrix (L2-normalised input → dot product)."""
    return emb @ emb.T


def rank_matrix(sim: np.ndarray) -> np.ndarray:
    """
    For each row i, rank all N books by *descending* similarity.
    Self-similarity (diagonal) is set to +inf so book i is always rank 0.
    Returns integer rank matrix (N, N); rank[i, j] = rank of book j from i's perspective.
    """
    N = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, np.inf)          # self → always top rank
    # argsort ascending on (-s) → rank 0 = most similar (excluding self)
    order = np.argsort(-s, axis=1)       # (N, N) — column = book index
    ranks = np.empty_like(order)
    rows  = np.arange(N)[:, None]
    ranks[rows, order] = np.arange(N)   # inverse permutation
    return ranks


# ─── Metrics ──────────────────────────────────────────────────────────────────

def spearman_per_book(ranks_a: np.ndarray, ranks_b: np.ndarray) -> np.ndarray:
    """
    Compute Spearman ρ between the rank vectors of each book across two models.
    ranks_a, ranks_b: (N, N) integer rank matrices.
    Returns length-N array of ρ values.
    """
    N = ranks_a.shape[0]
    rhos = np.empty(N, dtype=np.float64)
    for i in range(N):
        # Exclude the self-rank entry (rank 0 for both → inflates correlation artificially)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        rhos[i] = spearmanr(ranks_a[i, mask], ranks_b[i, mask]).statistic
    return rhos


def jaccard_per_book(sim_a: np.ndarray, sim_b: np.ndarray, k: int) -> np.ndarray:
    """
    For each book i, find its top-k neighbours (excluding itself) under each model
    and compute Jaccard(A, B).
    Returns length-N array of Jaccard scores.
    """
    N = sim_a.shape[0]
    jaccards = np.empty(N, dtype=np.float64)

    # mask out self
    sa, sb = sim_a.copy(), sim_b.copy()
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)

    top_a = np.argsort(-sa, axis=1)[:, :k]   # (N, k) — top-k indices per book
    top_b = np.argsort(-sb, axis=1)[:, :k]

    for i in range(N):
        A = set(top_a[i])
        B = set(top_b[i])
        union = len(A | B)
        jaccards[i] = len(A & B) / union if union else 0.0
    return jaccards


# ─── Load + compute ───────────────────────────────────────────────────────────

def compute_all(family: str, k: int) -> dict:
    """
    Load all model sizes for the family, compute pairwise metrics for every
    adjacent pair, return a results dict.
    """
    size_dirs = sorted(
        [d for d in os.listdir(os.path.join(EMB_DIR, family))
         if os.path.isdir(os.path.join(EMB_DIR, family, d))],
        key=_size_sort_key
    )

    if len(size_dirs) < 2:
        print(f"[error] Need at least 2 model sizes under {EMB_DIR}/{family}/. Found: {size_dirs}")
        sys.exit(1)

    print(f"[load] Found {len(size_dirs)} sizes: {size_dirs}")

    # Load all embeddings once
    embs, titles_ref = {}, None
    for s in size_dirs:
        emb, titles = load_model_data(family, s)
        embs[s] = emb
        if titles_ref is None:
            titles_ref = titles
        else:
            if titles != titles_ref:
                print("[warn] Book order differs between model sizes — aligning by title.")
                # Build mapping: title → index in the reference order
                ref_idx = {t: i for i, t in enumerate(titles_ref)}
                order   = [ref_idx[t] for t in titles]
                embs[s] = emb[order]
    print(f"[load] Embeddings loaded. N={len(titles_ref)} books.")

    # Pre-compute cosine similarity matrices for every size (cached)
    print(f"\n[sim] Pre-computing similarity matrices for all {len(size_dirs)} sizes...")
    sims = {s: cosine_sim_matrix(embs[s]) for s in size_dirs}

    # All unique ordered pairs (i < j) — superset of adjacent pairs
    adjacent_set = {(size_dirs[i], size_dirs[i + 1]) for i in range(len(size_dirs) - 1)}
    all_combos   = [(size_dirs[i], size_dirs[j])
                    for i in range(len(size_dirs))
                    for j in range(i + 1, len(size_dirs))]

    all_pair_results = []

    for sa, sb in all_combos:
        pair_label = f"{sa} → {sb}"
        is_adjacent = (sa, sb) in adjacent_set
        print(f"\n[pair] {pair_label}{'  (adjacent)' if is_adjacent else ''}")

        print(f"  Computing rank correlation (Spearman ρ) ...")
        ranks_a = rank_matrix(sims[sa])
        ranks_b = rank_matrix(sims[sb])
        rhos = spearman_per_book(ranks_a, ranks_b)

        print(f"  Computing Jaccard (k={k}) ...")
        jaccards = jaccard_per_book(sims[sa], sims[sb], k)

        entry = {
            "size_a":      sa,
            "size_b":      sb,
            "label":       pair_label,
            "is_adjacent": is_adjacent,
            "spearman": {
                "per_book": rhos.tolist(),
                "mean":     float(np.mean(rhos)),
                "std":      float(np.std(rhos, ddof=0)),
                "median":   float(np.median(rhos)),
            },
            "jaccard": {
                "per_book": jaccards.tolist(),
                "mean":     float(np.mean(jaccards)),
                "std":      float(np.std(jaccards, ddof=0)),
                "median":   float(np.median(jaccards)),
            },
        }
        all_pair_results.append(entry)

        print(f"  Spearman ρ:    mean={entry['spearman']['mean']:.4f}  "
              f"std={entry['spearman']['std']:.4f}")
        print(f"  Jaccard(k={k}): mean={entry['jaccard']['mean']:.4f}  "
              f"std={entry['jaccard']['std']:.4f}")

    # Adjacent-only subset (used by bar chart / print_summary)
    pair_results = [p for p in all_pair_results if p["is_adjacent"]]

    return {
        "family":    family,
        "k":         k,
        "n_books":   len(titles_ref),
        "sizes":     size_dirs,
        "pairs":     pair_results,      # adjacent only — for bar chart
        "all_pairs": all_pair_results,  # every (i,j) pair — for heatmap
        "titles":    titles_ref,
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: str) -> None:
    pairs     = results["pairs"]
    labels    = [p["label"] for p in pairs]
    x         = np.arange(len(labels))

    sp_means  = [p["spearman"]["mean"] for p in pairs]
    sp_stds   = [p["spearman"]["std"]  for p in pairs]
    jac_means = [p["jaccard"]["mean"]  for p in pairs]
    jac_stds  = [p["jaccard"]["std"]   for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(max(7, len(pairs) * 2.5), 4.5))
    fig.suptitle(
        f"{results['family']} — Neighbourhood Stability Across Model Sizes\n"
        f"(k={results['k']}, N={results['n_books']} books)",
        fontsize=12
    )

    def _panel(ax, means, stds, per_books, title, ylabel, ylim):
        ax.bar(x, means, width=0.5, color="steelblue", alpha=0.7, label="Mean")
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                    label="± 1 std")
        for xi, scores in enumerate(per_books):
            jitter = np.random.default_rng(xi).uniform(-0.18, 0.18, len(scores))
            ax.scatter(xi + jitter, scores, s=8, color="black", alpha=0.2, zorder=4,
                       label="Per-book" if xi == 0 else "_nolegend_")
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + (ylim[1] - ylim[0]) * 0.02, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel("Adjacent Model Pair")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    _panel(axes[0],
           sp_means, sp_stds,
           [p["spearman"]["per_book"] for p in pairs],
           "Rank Correlation (Spearman ρ)",
           "Spearman ρ", (-1.05, 1.05))

    _panel(axes[1],
           jac_means, jac_stds,
           [p["jaccard"]["per_book"] for p in pairs],
           f"Top-{results['k']} Neighbour Set (Jaccard)",
           "Jaccard Similarity", (-0.05, 1.05))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out_path}")


# ─── Heatmap (all pairs) ─────────────────────────────────────────────────────

def plot_heatmap(results: dict, out_path: str) -> None:
    """
    Draw two N×N heatmaps (Spearman ρ and Jaccard) where each cell (i, j)
    shows the mean score between model size i and model size j.
    Diagonal = 1.0 (perfect self-similarity).  Matrix is symmetric.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("[warn] seaborn not installed — skipping heatmap. pip install seaborn")
        return

    sizes   = results["sizes"]
    n       = len(sizes)
    idx     = {s: i for i, s in enumerate(sizes)}

    sp_mat  = np.full((n, n), np.nan)
    jac_mat = np.full((n, n), np.nan)
    np.fill_diagonal(sp_mat,  1.0)
    np.fill_diagonal(jac_mat, 1.0)

    for p in results["all_pairs"]:
        i, j = idx[p["size_a"]], idx[p["size_b"]]
        sp_mat[i, j]  = sp_mat[j, i]  = p["spearman"]["mean"]
        jac_mat[i, j] = jac_mat[j, i] = p["jaccard"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.6) * 2, max(6, n * 1.4)))
    fig.suptitle(
        f"{results['family']} — Pairwise Stability Between All Model Sizes\n"
        f"(k={results['k']}, N={results['n_books']} books)",
        fontsize=13,
    )

    for ax, matrix, title in [
        (axes[0], sp_mat,  "Mean Spearman ρ"),
        (axes[1], jac_mat, f"Mean Jaccard (k={results['k']})"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".3f",
            xticklabels=sizes,
            yticklabels=sizes,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.4,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Model Size")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out_path}")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    k = results["k"]
    print(f"\n{'='*68}")
    print(f"  {results['family']} — Stability Across Model Sizes (k={k})")
    print(f"{'='*68}")
    print(f"  {'Pair':>14}  {'Spearman mean':>14}  {'Spearman std':>13}  "
          f"{'Jaccard mean':>13}  {'Jaccard std':>12}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*13}  {'-'*13}  {'-'*12}")
    for p in results["pairs"]:
        print(f"  {p['label']:>14}  "
              f"{p['spearman']['mean']:>14.4f}  {p['spearman']['std']:>13.4f}  "
              f"{p['jaccard']['mean']:>13.4f}  {p['jaccard']['std']:>12.4f}")
    print(f"{'='*68}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure embedding neighbourhood stability across model sizes"
    )
    parser.add_argument("--model_family", required=True,
                        help="e.g. Qwen3-Embedding")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbours for Jaccard (default: 10)")
    args = parser.parse_args()

    out_dir      = os.path.join(BASE_OUT, args.model_family)
    json_out     = os.path.join(out_dir, "results.json")
    plot_out     = os.path.join(out_dir, "stability.png")
    heatmap_out  = os.path.join(out_dir, "heatmap.png")

    results = compute_all(args.model_family, args.k)

    print_summary(results)

    # Save JSON (drop per_book arrays to keep it compact)
    os.makedirs(out_dir, exist_ok=True)

    def _strip_per_book(p: dict) -> dict:
        return {
            "size_a":      p["size_a"],
            "size_b":      p["size_b"],
            "label":       p["label"],
            "is_adjacent": p["is_adjacent"],
            "spearman":    {k2: v for k2, v in p["spearman"].items() if k2 != "per_book"},
            "jaccard":     {k2: v for k2, v in p["jaccard"].items()  if k2 != "per_book"},
        }

    save = {
        "family":    results["family"],
        "k":         results["k"],
        "n_books":   results["n_books"],
        "sizes":     results["sizes"],
        "pairs":     [_strip_per_book(p) for p in results["pairs"]],
        "all_pairs": [_strip_per_book(p) for p in results["all_pairs"]],
    }
    with open(json_out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"[save] Summary → {json_out}")

    plot_results(results, plot_out)
    plot_heatmap(results, heatmap_out)


if __name__ == "__main__":
    main()
