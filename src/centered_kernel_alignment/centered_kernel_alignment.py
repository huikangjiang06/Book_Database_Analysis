"""
centered_kernel_alignment.py
=============================
Computes CKA (Centered Kernel Alignment) between every pair of model sizes
within a model family, using multiple kernels.

CKA measures how similar two representation matrices are, invariant to
orthogonal transforms and isotropic scaling.  Given embeddings X (n×p) and
Y (n×q) for the same n books:

  1. Build kernel matrices  K = k(X, X),  L = k(Y, Y)   — shape (n, n)
  2. Double-centre:         K̃ = HKH,  L̃ = HLH   where H = I − (1/n)11ᵀ
  3. HSIC(K, L) = tr(K̃ L̃) / (n−1)²
  4. CKA(K, L) = HSIC(K, L) / √( HSIC(K,K) · HSIC(L,L) )

CKA ∈ [0, 1].  1 = geometrically identical representations (up to above transforms).

Three kernels are supported:
  linear  — K = X Xᵀ
               Measures raw dot-product similarity in embedding space.
  cosine  — K = X̂ X̂ᵀ  where X̂ is L2-normalised X
               Same as linear but invariant to per-vector magnitude.
  rbf     — K_ij = exp(−‖xᵢ−xⱼ‖² / (2σ²))
               σ = median pairwise distance (median heuristic), computed
               independently per model size.  Captures local Gaussian structure.

Outputs (under out/centered_kernel_alignment/<family>/):
  results.json  — CKA matrix for every kernel (n_sizes × n_sizes)
  heatmap.png   — seaborn heatmaps, one panel per kernel

Usage:
    python src/centered_kernel_alignment/centered_kernel_alignment.py \\
        --model_family Qwen3-Embedding
    python src/centered_kernel_alignment/centered_kernel_alignment.py \\
        --model_family Pythia --kernels linear cosine rbf
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

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
BASE_OUT = os.path.join(ROOT, "out", "centered_kernel_alignment")

AVAILABLE_KERNELS = ("linear", "cosine", "rbf")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _size_sort_key(s: str) -> float:
    """'0.6B' → 600, '4B' → 4000, '70M' → 70, etc."""
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def load_model_data(family: str, size: str) -> tuple[np.ndarray, list[str]]:
    """Load (N, D) embedding matrix and book title list from raw .pkl files."""
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
        embeddings.append(np.array(emb, dtype=np.float64))
        titles.append(d.get("book_title", os.path.splitext(os.path.basename(path))[0]))
    return np.stack(embeddings), titles   # (N, D), raw — NOT normalised here


# ─── Kernels ──────────────────────────────────────────────────────────────────

def _sq_dists(X: np.ndarray) -> np.ndarray:
    """Compute (N, N) matrix of squared Euclidean distances efficiently."""
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    sq_norms = (X ** 2).sum(axis=1)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    return np.maximum(D2, 0.0)   # numerical safety: clip tiny negatives


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """K = X Xᵀ  — dot-product kernel."""
    return X @ X.T


def cosine_kernel(X: np.ndarray) -> np.ndarray:
    """K = X̂ X̂ᵀ where X̂ = X / ||X||₂  — cosine similarity kernel."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_hat = X / norms
    return X_hat @ X_hat.T


def rbf_kernel(X: np.ndarray) -> np.ndarray:
    """
    K_ij = exp(−‖xᵢ−xⱼ‖² / (2σ²))
    σ is chosen by the median heuristic:
        σ = median of all pairwise distances √(‖xᵢ−xⱼ‖²) for i < j
    This makes the kernel self-calibrating to the scale of each embedding space.
    """
    D2 = _sq_dists(X)
    # Extract upper-triangle pairwise distances (excluding diagonal)
    triu = np.triu_indices(len(X), k=1)
    pairwise_dists = np.sqrt(D2[triu])
    sigma = np.median(pairwise_dists)
    if sigma < 1e-10:
        sigma = 1e-10
    return np.exp(-D2 / (2.0 * sigma ** 2))


_KERNEL_FNS = {
    "linear": linear_kernel,
    "cosine": cosine_kernel,
    "rbf":    rbf_kernel,
}

_KERNEL_LABELS = {
    "linear": "Linear  (K = XXᵀ)",
    "cosine": "Cosine  (L2-normalised)",
    "rbf":    "RBF  (median-heuristic σ)",
}


# ─── CKA ──────────────────────────────────────────────────────────────────────

def centre_kernel(K: np.ndarray) -> np.ndarray:
    """Double-centre K: H K H  where H = I − (1/n) 11ᵀ."""
    row_mean   = K.mean(axis=1, keepdims=True)
    col_mean   = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Biased HSIC estimator:  tr(K̃ L̃) / (n−1)²
    K̃ and L̃ are the double-centred kernels.
    """
    Kc = centre_kernel(K)
    Lc = centre_kernel(L)
    n  = K.shape[0]
    return float(np.sum(Kc * Lc)) / (n - 1) ** 2


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """
    CKA(K, L) = HSIC(K, L) / √(HSIC(K,K) · HSIC(L,L))

    Returns a value in [0, 1].  1 = representations are identical up to
    orthogonal transforms and isotropic scaling.
    """
    h_kl = hsic(K, L)
    h_kk = hsic(K, K)
    h_ll = hsic(L, L)
    denom = np.sqrt(h_kk * h_ll)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(h_kl / denom, 0.0, 1.0))


# ─── Load + compute ───────────────────────────────────────────────────────────

def compute_all(family: str, kernels: list[str]) -> dict:
    """
    Load embeddings for every model size, compute kernels, then calculate
    CKA for all N×N pairs (symmetric, diagonal = 1.0).
    Returns a results dict with a separate CKA matrix per kernel.
    """
    size_dirs = sorted(
        [d for d in os.listdir(os.path.join(EMB_DIR, family))
         if os.path.isdir(os.path.join(EMB_DIR, family, d))],
        key=_size_sort_key
    )

    if len(size_dirs) < 2:
        print(f"[error] Need at least 2 model sizes under {EMB_DIR}/{family}/. "
              f"Found: {size_dirs}")
        sys.exit(1)

    print(f"[load] Found {len(size_dirs)} sizes: {size_dirs}")

    # ── Load embeddings ────────────────────────────────────────────────────────
    embs, titles_ref = {}, None
    for s in size_dirs:
        emb, titles = load_model_data(family, s)
        embs[s] = emb
        if titles_ref is None:
            titles_ref = titles
        else:
            if titles != titles_ref:
                print("[warn] Book order differs — aligning by title.")
                ref_idx  = {t: i for i, t in enumerate(titles_ref)}
                order    = [ref_idx[t] for t in titles]
                embs[s]  = emb[order]
    N = len(titles_ref)
    print(f"[load] {N} books loaded.")

    # ── Build kernel matrices (one per size per kernel) ────────────────────────
    print(f"\n[kernel] Computing kernel matrices for kernels: {kernels}")
    Ks: dict[str, dict[str, np.ndarray]] = {k: {} for k in kernels}
    for s in size_dirs:
        for kname in kernels:
            print(f"  {s:>28}  kernel={kname}", end="  ... ", flush=True)
            Ks[kname][s] = _KERNEL_FNS[kname](embs[s])
            print("done")

    # ── CKA for all pairs ─────────────────────────────────────────────────────
    print(f"\n[cka]   Computing CKA for all {len(size_dirs)*(len(size_dirs)-1)//2}"
          f" pairs × {len(kernels)} kernels")

    # cka_matrix[kname] is a symmetric n_sizes × n_sizes matrix
    n = len(size_dirs)
    cka_matrices = {kname: np.eye(n) for kname in kernels}

    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = size_dirs[i], size_dirs[j]
            for kname in kernels:
                val = cka(Ks[kname][sa], Ks[kname][sb])
                cka_matrices[kname][i, j] = val
                cka_matrices[kname][j, i] = val
            vals_str = "  ".join(f"{kname}={cka_matrices[kname][i,j]:.4f}"
                                  for kname in kernels)
            print(f"  {sa} → {sb}:  {vals_str}")

    return {
        "family":  family,
        "kernels": kernels,
        "n_books": N,
        "sizes":   size_dirs,
        "cka":     {kname: cka_matrices[kname].tolist() for kname in kernels},
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_heatmap(results: dict, out_path: str) -> None:
    """
    Draw one seaborn heatmap per kernel side-by-side.
    Rows/columns = model sizes sorted by parameter count.
    Cell value = CKA between size i and size j.
    Diagonal is always 1.0 (self-similarity).
    """
    try:
        import seaborn as sns
    except ImportError:
        print("[warn] seaborn not installed — skipping heatmap.  pip install seaborn")
        return

    kernels = results["kernels"]
    sizes   = results["sizes"]
    n_k     = len(kernels)

    fig, axes = plt.subplots(
        1, n_k,
        figsize=(max(5, len(sizes) * 1.5) * n_k, max(4, len(sizes) * 1.4)),
    )
    if n_k == 1:
        axes = [axes]

    fig.suptitle(
        f"{results['family']} — Centered Kernel Alignment (CKA) Across Model Sizes\n"
        f"(N={results['n_books']} books)",
        fontsize=13,
    )

    for ax, kname in zip(axes, kernels):
        matrix = np.array(results["cka"][kname])
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".3f",
            xticklabels=sizes,
            yticklabels=sizes,
            cmap="YlOrRd",
            vmin=0.8,
            vmax=1.0,
            linewidths=0.4,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(_KERNEL_LABELS[kname], fontsize=11)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Model Size")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out_path}")


# ─── Bar chart (adjacent pairs) ─────────────────────────────────────────────

def plot_bar(results: dict, out_path: str) -> None:
    """
    Bar chart of CKA for every adjacent model-size pair, grouped by kernel.
    One subplot per kernel — mirrors the style of stability_across_model_size.py.
    """
    sizes   = results["sizes"]
    kernels = results["kernels"]
    n_k     = len(kernels)
    n       = len(sizes)
    idx     = {s: i for i, s in enumerate(sizes)}

    # Collect adjacent pairs in size order
    adj_pairs  = [(sizes[i], sizes[i + 1]) for i in range(n - 1)]
    adj_labels = [f"{a} → {b}" for a, b in adj_pairs]
    x          = np.arange(len(adj_pairs))

    fig, axes = plt.subplots(
        1, n_k,
        figsize=(max(6, len(adj_pairs) * 1.8) * n_k, 4.5),
        sharey=False,
    )
    if n_k == 1:
        axes = [axes]

    fig.suptitle(
        f"{results['family']} — CKA for Adjacent Model Sizes\n"
        f"(N={results['n_books']} books)",
        fontsize=12,
    )

    # Determine a shared y-range across all kernels for easy comparison
    all_vals = [
        results["cka"][kname][idx[a]][idx[b]]
        for kname in kernels
        for a, b in adj_pairs
    ]
    y_min = max(0.0, min(all_vals) - 0.05)
    y_max = min(1.0, max(all_vals) + 0.05)

    for ax, kname in zip(axes, kernels):
        vals = [results["cka"][kname][idx[a]][idx[b]] for a, b in adj_pairs]

        bars = ax.bar(x, vals, width=0.5, color="steelblue", alpha=0.7)

        # Value annotation above each bar
        for xi, v in enumerate(vals):
            ax.text(xi, v + (y_max - y_min) * 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(adj_labels, fontsize=9, rotation=15, ha="right")
        ax.set_xlabel("Adjacent Model Pair")
        ax.set_ylabel("CKA")
        ax.set_title(_KERNEL_LABELS[kname], fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out_path}")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    sizes   = results["sizes"]
    kernels = results["kernels"]
    n       = len(sizes)

    for kname in kernels:
        matrix = results["cka"][kname]
        print(f"\n{'='*60}")
        print(f"  {results['family']} — CKA ({_KERNEL_LABELS[kname]})")
        print(f"{'='*60}")

        # Header row
        col_w = max(len(s) for s in sizes) + 2
        header = f"  {'':>{col_w}}" + "".join(f"  {s:>{col_w}}" for s in sizes)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for i, row_size in enumerate(sizes):
            row_vals = "".join(f"  {matrix[i][j]:>{col_w}.3f}" for j in range(n))
            print(f"  {row_size:>{col_w}}{row_vals}")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CKA between all pairs of model sizes in a family"
    )
    parser.add_argument(
        "--model_family", required=True,
        help="e.g. Qwen3-Embedding, Pythia, Cerebras-GPT, Qwen2.5, OpenAI",
    )
    parser.add_argument(
        "--kernels", nargs="+", default=list(AVAILABLE_KERNELS),
        choices=AVAILABLE_KERNELS,
        help=f"Kernels to use (default: all — {AVAILABLE_KERNELS})",
    )
    args = parser.parse_args()

    out_dir   = os.path.join(BASE_OUT, args.model_family)
    json_out  = os.path.join(out_dir, "results.json")
    heat_out  = os.path.join(out_dir, "heatmap.png")
    bar_out   = os.path.join(out_dir, "bar.png")

    results = compute_all(args.model_family, args.kernels)
    print_table(results)

    os.makedirs(out_dir, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save]  Summary → {json_out}")

    plot_heatmap(results, heat_out)
    plot_bar(results, bar_out)


if __name__ == "__main__":
    main()
