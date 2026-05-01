"""
mutual_knn_alignment.py
=======================
Computes the mutual k-nearest-neighbor (mNN) alignment metric from
"The Platonic Representation Hypothesis" between every pair of model sizes
within a model family.

For a book i, let A_i and B_i be the sets of k nearest neighbors induced by
two embedding spaces. The per-book alignment is:

    mNN_i = |A_i intersection B_i| / k

The final score is the mean over all books. This is similar to the local
neighborhood overlap used in stability_across_model_size.py, but follows the
paper's normalization by k rather than Jaccard normalization by union size.

Outputs (under out/mutual_knn_alignment/<family>/):
  results.json  - mNN matrix and summary stats for all model-size pairs
  heatmap.png   - heatmap of mean mNN over all pairs
  bar.png       - adjacent-size bar chart with per-book scatter
  largest_model_line.png - mean mNN to the largest model in the family

Usage:
    python src/mutual_knn_alignment/mutual_knn_alignment.py \\
        --model_family Qwen3-Embedding
    python src/mutual_knn_alignment/mutual_knn_alignment.py \\
        --model_family Pythia --k 20 --abtt 2
    python src/mutual_knn_alignment/mutual_knn_alignment.py \\
        --model_family Pythia --embedding_level chunk --sample_n 5000
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_components_removal"))
from ABTT import abtt as apply_abtt

# Paths
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
BASE_OUT = os.path.join(ROOT, "out", "mutual_knn_alignment")
BOOK_LEVEL = "book"
CHUNK_LEVEL = "chunk"


# Helpers

class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2.x pickles from NumPy 1.x runtimes."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def _size_sort_key(s: str) -> float:
    """'0.6B' -> 600, '4B' -> 4000, '70M' -> 70, etc."""
    if s == "text-embedding-3-small":
        return 1e12
    if s == "text-embedding-3-large":
        return 1e12 + 1
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def get_sizes(family: str) -> list[str]:
    base = os.path.join(EMB_DIR, family)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No embedding directory found for family: {base}")
    return sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d)) and d != "chunks"],
        key=_size_sort_key,
    )


def _normalise_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _book_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _chunk_id(book_id: str, chunk_idx: int) -> str:
    return f"{book_id}::chunk_{chunk_idx:05d}"


def collect_embedding_ids(family: str, size: str, embedding_level: str) -> list[str]:
    """Collect stable book or chunk IDs without keeping embedding matrices."""
    base = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No .pkl files found under {base}. "
            f"Check --model_family / model_size names."
        )
    ids = []
    for path in paths:
        d = load_pickle(path)
        if embedding_level == BOOK_LEVEL:
            if d.get("embedding") is not None or d.get("book_embedding") is not None:
                ids.append(d.get("book_title", _book_id_from_path(path)))
            continue

        chunks = d.get("chunk_embeddings")
        if chunks is None:
            print(f"  [warn] No chunk_embeddings in {path}, skipping.")
            continue
        book_id = _book_id_from_path(path)
        ids.extend(_chunk_id(book_id, i) for i in range(len(chunks)))
    return ids


def sample_ids(ids: list[str], sample_n: int, seed: int) -> list[str]:
    if sample_n <= 0 or sample_n >= len(ids):
        return list(ids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=sample_n, replace=False)
    return [ids[i] for i in sorted(idx)]


def load_model_data(
    family: str,
    size: str,
    embedding_level: str = BOOK_LEVEL,
    selected_ids: set[str] | None = None,
    abtt_n: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Load L2-normalised embeddings and their stable IDs."""
    base = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No .pkl files found under {base}. "
            f"Check --model_family / model_size names."
        )

    embeddings, ids = [], []
    for path in paths:
        d = load_pickle(path)
        if embedding_level == BOOK_LEVEL:
            emb = d.get("embedding")
            if emb is None:
                emb = d.get("book_embedding")
            if emb is None:
                print(f"  [warn] No embedding in {path}, skipping.")
                continue
            emb_id = d.get("book_title", _book_id_from_path(path))
            if selected_ids is not None and emb_id not in selected_ids:
                continue
            embeddings.append(np.array(emb, dtype=np.float64))
            ids.append(emb_id)
        else:
            chunks = d.get("chunk_embeddings")
            if chunks is None:
                print(f"  [warn] No chunk_embeddings in {path}, skipping.")
                continue
            book_id = _book_id_from_path(path)
            for chunk_idx, emb in enumerate(chunks):
                emb_id = _chunk_id(book_id, chunk_idx)
                if selected_ids is not None and emb_id not in selected_ids:
                    continue
                embeddings.append(np.array(emb, dtype=np.float64))
                ids.append(emb_id)

    if not embeddings:
        raise ValueError(f"No usable embeddings found under {base}")

    X = _normalise_rows(np.stack(embeddings))
    if abtt_n > 0 and embedding_level == BOOK_LEVEL:
        X = apply_abtt(family, size, X, abtt_n)
        X = _normalise_rows(X)
    return X.astype(np.float32), ids


def align_to_reference(
    emb: np.ndarray,
    ids: list[str],
    ids_ref: list[str],
) -> np.ndarray:
    """Reorder emb so rows match ids_ref."""
    if ids == ids_ref:
        return emb

    cur_idx = {t: i for i, t in enumerate(ids)}
    missing = [t for t in ids_ref if t not in cur_idx]
    ids_ref_set = set(ids_ref)
    extra = [t for t in ids if t not in ids_ref_set]
    if missing or extra:
        raise ValueError(
            "Embedding ID sets differ between model sizes: "
            f"{len(missing)} missing from current, {len(extra)} extra."
        )
    order = [cur_idx[t] for t in ids_ref]
    return emb[order]


# Mutual k-NN metric

def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine similarity matrix (L2-normalised input -> dot product)."""
    return emb @ emb.T


def knn_indices(sim: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbor indices per row, excluding self."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def mutual_knn_per_item(top_a: np.ndarray, top_b: np.ndarray, n_items: int) -> np.ndarray:
    """
    Per-item |NN_k^A(i) intersection NN_k^B(i)| / k.
    top_a and top_b are (N, k) integer arrays.
    """
    k = top_a.shape[1]
    rows = np.arange(n_items)[:, None]
    mask_a = np.zeros((n_items, n_items), dtype=bool)
    mask_b = np.zeros((n_items, n_items), dtype=bool)
    mask_a[rows, top_a] = True
    mask_b[rows, top_b] = True
    return (mask_a & mask_b).sum(axis=1).astype(np.float64) / k


mutual_knn_per_book = mutual_knn_per_item


# Load + compute

def select_common_ids(
    family: str,
    size_dirs: list[str],
    embedding_level: str,
    sample_n: int,
    sample_seed: int,
) -> list[str] | None:
    item_label = "book" if embedding_level == BOOK_LEVEL else "chunk"
    print(f"[sample] Collecting common {item_label} IDs across {len(size_dirs)} sizes...")
    common_ids: set[str] | None = None
    first_order: list[str] | None = None
    for size in size_dirs:
        ids = collect_embedding_ids(family, size, embedding_level)
        if first_order is None:
            first_order = ids
        ids_set = set(ids)
        common_ids = ids_set if common_ids is None else common_ids & ids_set
        print(f"  {size:>28}: {len(ids):>7} {item_label}s")

    if not common_ids:
        raise ValueError(f"No common {item_label}s found across {family} sizes.")

    ordered_common = [eid for eid in first_order if eid in common_ids]
    if embedding_level == BOOK_LEVEL:
        print(f"[sample] Common books: {len(ordered_common)}")
        return ordered_common

    sampled = sample_ids(ordered_common, sample_n, sample_seed)
    print(f"[sample] Common chunks: {len(ordered_common)}; using {len(sampled)}" +
          (f" (seed={sample_seed})" if 0 < sample_n < len(ordered_common) else ""))
    return sampled


def compute_all(
    family: str,
    k: int,
    embedding_level: str = BOOK_LEVEL,
    sample_n: int = 5000,
    sample_seed: int = 42,
    abtt_n: int = 0,
) -> dict:
    size_dirs = get_sizes(family)

    if len(size_dirs) < 2:
        print(f"[error] Need at least 2 model sizes under {EMB_DIR}/{family}/. Found: {size_dirs}")
        sys.exit(1)

    print(f"[load] Found {len(size_dirs)} sizes: {size_dirs}")
    selected_ids_list = select_common_ids(
        family, size_dirs, embedding_level, sample_n, sample_seed
    )
    selected_ids = set(selected_ids_list) if selected_ids_list is not None else None

    embs, ids_ref = {}, None
    for s in size_dirs:
        emb, ids = load_model_data(
            family,
            s,
            embedding_level=embedding_level,
            selected_ids=selected_ids,
            abtt_n=abtt_n,
        )
        if ids_ref is None:
            ids_ref = ids if selected_ids_list is None else selected_ids_list
            if selected_ids_list is not None:
                emb = align_to_reference(emb, ids, ids_ref)
        else:
            if ids != ids_ref:
                print("[warn] Embedding order differs between model sizes - aligning by ID.")
            emb = align_to_reference(emb, ids, ids_ref)
        embs[s] = emb

    n_items = len(ids_ref)
    if k < 1 or k >= n_items:
        raise ValueError(f"k must satisfy 1 <= k < n_embeddings. Got k={k}, n_embeddings={n_items}.")

    item_label = "books" if embedding_level == BOOK_LEVEL else "chunks"
    print(f"[load] Embeddings loaded. N={n_items} {item_label}." +
          (f"  (ABTT n={abtt_n})" if abtt_n > 0 and embedding_level == BOOK_LEVEL else ""))

    print(f"\n[knn]  Pre-computing top-{k} neighbor sets for all {len(size_dirs)} sizes...")
    topk = {}
    for s in size_dirs:
        print(f"  {s:>28}", end=" ... ", flush=True)
        topk[s] = knn_indices(cosine_sim_matrix(embs[s]), k)
        print("done")

    adjacent_set = {(size_dirs[i], size_dirs[i + 1]) for i in range(len(size_dirs) - 1)}
    all_combos = [
        (size_dirs[i], size_dirs[j])
        for i in range(len(size_dirs))
        for j in range(i + 1, len(size_dirs))
    ]

    all_pair_results = []
    n = len(size_dirs)
    matrix = np.eye(n, dtype=np.float64)
    idx = {s: i for i, s in enumerate(size_dirs)}

    print(f"\n[mnn]  Computing mNN for {len(all_combos)} model-size pairs")
    for sa, sb in all_combos:
        scores = mutual_knn_per_item(topk[sa], topk[sb], n_items)
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=0))
        median = float(np.median(scores))
        is_adjacent = (sa, sb) in adjacent_set
        label = f"{sa} -> {sb}"

        i, j = idx[sa], idx[sb]
        matrix[i, j] = matrix[j, i] = mean

        entry = {
            "size_a": sa,
            "size_b": sb,
            "label": label,
            "is_adjacent": is_adjacent,
            "per_book": scores.tolist(),
            "mean": mean,
            "std": std,
            "median": median,
        }
        all_pair_results.append(entry)
        print(f"  {label}: mean={mean:.4f}  std={std:.4f}  median={median:.4f}" +
              ("  (adjacent)" if is_adjacent else ""))

    return {
        "family": family,
        "k": k,
        "abtt_n": abtt_n,
        "embedding_level": embedding_level,
        "sample_n": sample_n if embedding_level == CHUNK_LEVEL else None,
        "sample_seed": sample_seed if embedding_level == CHUNK_LEVEL else None,
        "n_books": n_items if embedding_level == BOOK_LEVEL else None,
        "n_embeddings": n_items,
        "sizes": size_dirs,
        "mnn": matrix.tolist(),
        "pairs": [p for p in all_pair_results if p["is_adjacent"]],
        "all_pairs": all_pair_results,
    }


# Plot

def _count_label(results: dict) -> str:
    item_label = "books" if results["embedding_level"] == BOOK_LEVEL else "chunks"
    return f"N={results['n_embeddings']} {item_label}"


def plot_heatmap(results: dict, out_path: str) -> None:
    try:
        import seaborn as sns
    except ImportError:
        print("[warn] seaborn not installed - skipping heatmap. pip install seaborn")
        return

    matrix = np.array(results["mnn"])
    sizes = results["sizes"]
    n = len(sizes)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.6), max(5, n * 1.4)))
    title = f"{results['family']} - Mutual k-NN Alignment"
    if results["abtt_n"] > 0:
        title += f" (ABTT n={results['abtt_n']})"
    fig.suptitle(
        f"{title}\n(k={results['k']}, {_count_label(results)})",
        fontsize=13,
    )

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
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Model Size")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {out_path}")


def plot_bar(results: dict, out_path: str) -> None:
    pairs = results["pairs"]
    labels = [p["label"] for p in pairs]
    means = [p["mean"] for p in pairs]
    stds = [p["std"] for p in pairs]
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(max(7, len(pairs) * 2.3), 4.5))
    title = f"{results['family']} - Mutual k-NN for Adjacent Model Sizes"
    if results["abtt_n"] > 0:
        title += f" (ABTT n={results['abtt_n']})"
    fig.suptitle(
        f"{title}\n(k={results['k']}, {_count_label(results)})",
        fontsize=12,
    )

    ax.bar(x, means, width=0.5, color="steelblue", alpha=0.7, label="Mean")
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                label="+/- 1 std")

    for xi, p in enumerate(pairs):
        scores = p["per_book"]
        jitter = np.random.default_rng(xi).uniform(-0.18, 0.18, len(scores))
        ax.scatter(xi + jitter, scores, s=8, color="black", alpha=0.2, zorder=4,
                   label="Per-book" if xi == 0 else "_nolegend_")

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, min(1.02, m + s + 0.03), f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_xlabel("Adjacent Model Pair")
    ax.set_ylabel("Mean overlap / k")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {out_path}")


def plot_largest_model_line(results: dict, out_path: str) -> None:
    """Plot the heatmap row comparing every size to the largest model."""
    sizes = results["sizes"]
    matrix = np.array(results["mnn"])
    largest_size = sizes[-1]
    means = matrix[-1, :]
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(max(7, len(sizes) * 1.5), 4.5))
    title = f"{results['family']} - Similarity to Largest Model ({largest_size})"
    if results["abtt_n"] > 0:
        title += f" (ABTT n={results['abtt_n']})"
    fig.suptitle(
        f"{title}\n(k={results['k']}, {_count_label(results)})",
        fontsize=12,
    )

    ax.plot(x, means, marker="o", color="steelblue", linewidth=2)
    for xi, v in zip(x, means):
        ax.text(xi, min(1.02, v + 0.03), f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10, rotation=20, ha="right")
    ax.set_xlabel("Model Size")
    ax.set_ylabel(f"mNN with {largest_size}")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {out_path}")


# Summary table

def print_summary(results: dict) -> None:
    title = f"{results['family']} - Mutual k-NN Alignment (k={results['k']})"
    if results["embedding_level"] == CHUNK_LEVEL:
        title += f" chunk sample n={results['n_embeddings']}"
    if results["abtt_n"] > 0:
        title += f" ABTT n={results['abtt_n']}"
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'Pair':>22}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}")
    for p in results["pairs"]:
        print(f"  {p['label']:>22}  {p['mean']:>8.4f}  "
              f"{p['std']:>8.4f}  {p['median']:>8.4f}")
    print(f"{'='*72}\n")


def strip_per_book(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "per_book"}


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mutual k-NN alignment between model sizes in a family"
    )
    parser.add_argument(
        "--model_family", required=True,
        help="e.g. Qwen3-Embedding, Pythia, Cerebras-GPT, Qwen2.5, OpenAI",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbors for mNN (default: 10)",
    )
    parser.add_argument(
        "--embedding_level", choices=[BOOK_LEVEL, CHUNK_LEVEL], default=BOOK_LEVEL,
        help="Use whole-book embeddings or individual chunk embeddings (default: book)",
    )
    parser.add_argument(
        "--sample_n", type=int, default=5000,
        help="Number of common chunk embeddings to sample in chunk mode; 0 = all (default: 5000)",
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42,
        help="Random seed for chunk sampling (default: 42)",
    )
    parser.add_argument(
        "--abtt", type=int, default=0, metavar="N",
        help="Book mode only: apply ABTT by removing top-N principal directions (0 = disabled)",
    )
    args = parser.parse_args()

    if args.embedding_level == CHUNK_LEVEL and args.abtt > 0:
        raise ValueError("ABTT is only available for book embeddings; use --abtt 0 with --embedding_level chunk.")

    if args.embedding_level == CHUNK_LEVEL:
        sample_suffix = "all" if args.sample_n <= 0 else str(args.sample_n)
        suffix = f"_chunks_n{sample_suffix}"
    else:
        suffix = "_abtt" if args.abtt > 0 else ""
    out_dir = os.path.join(BASE_OUT, args.model_family + suffix)
    json_out = os.path.join(out_dir, "results.json")
    heatmap_out = os.path.join(out_dir, "heatmap.png")
    bar_out = os.path.join(out_dir, "bar.png")
    largest_line_out = os.path.join(out_dir, "largest_model_line.png")

    results = compute_all(
        args.model_family,
        args.k,
        embedding_level=args.embedding_level,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        abtt_n=args.abtt,
    )
    print_summary(results)

    os.makedirs(out_dir, exist_ok=True)
    save = {
        "family": results["family"],
        "k": results["k"],
        "abtt_n": results["abtt_n"],
        "embedding_level": results["embedding_level"],
        "sample_n": results["sample_n"],
        "sample_seed": results["sample_seed"],
        "n_books": results["n_books"],
        "n_embeddings": results["n_embeddings"],
        "sizes": results["sizes"],
        "mnn": results["mnn"],
        "largest_model_alignment": {
            "reference_size": results["sizes"][-1],
            "sizes": results["sizes"],
            "means": np.array(results["mnn"])[-1, :].tolist(),
        },
        "pairs": [strip_per_book(p) for p in results["pairs"]],
        "all_pairs": [strip_per_book(p) for p in results["all_pairs"]],
    }
    with open(json_out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"[save] Summary -> {json_out}")

    plot_heatmap(results, heatmap_out)
    plot_bar(results, bar_out)
    plot_largest_model_line(results, largest_line_out)


if __name__ == "__main__":
    main()
