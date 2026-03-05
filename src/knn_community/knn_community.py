"""
knn_community.py
=================
Graph community-structure analysis over book embeddings.

Pipeline (for each model size in a family, sweeping k and γ):

  1. Build mutual kNN graph — nodes = books, edges = mutual nearest neighbours,
     weight = max(cosine_similarity, 0).

  2. Community detection — run Louvain (resolution γ) with multiple seeds;
     record best modularity Q and per-community conductance.

  3. Spectral analysis — normalised Laplacian → λ₂ (Fiedler value) and
     leading eigengaps (proxy for clusterability).

  4. Persistence across k — ARI between partitions at adjacent k-values for
     a fixed reference resolution (γ_ref=1.0 by default).

Emergence hypothesis: larger/better models produce embeddings with stronger
community structure (higher Q, lower conductance, larger λ₂, higher persistence)
across a broad (k, γ) region rather than at one fragile sweet-spot.

Outputs (under out/knn_community/<family>/):
  results.json             — complete stats for every (size, k, γ)
  heatmap_modularity.png   — k×γ heatmap of max-Q, one panel per model size
  heatmap_conductance.png  — k×γ heatmap of median conductance
  spectral.png             — λ₂ vs k, all sizes overlaid
  persistence.png          — ARI (adjacent k, γ_ref) vs k-pair, all sizes overlaid

Usage:
    python src/knn_community/knn_community.py --model_family Qwen3-Embedding
    python src/knn_community/knn_community.py --model_family Pythia \\
        --k_min 5 --k_max 80 --n_k 8 --n_gamma 6 --n_seeds 3
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
BASE_OUT = os.path.join(ROOT, "out", "knn_community")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _size_sort_key(s: str) -> float:
    """'0.6B' → 600, '4B' → 4000, '70M' → 70, etc."""
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def load_model_data(family: str, size: str) -> tuple[np.ndarray, list[str]]:
    """Load L2-normalised (N, D) embeddings and book title list."""
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
            continue
        embeddings.append(np.array(emb, dtype=np.float64))
        titles.append(d.get("book_title", os.path.splitext(os.path.basename(path))[0]))
    X = np.stack(embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms, titles


# ─── Graph construction ───────────────────────────────────────────────────────

def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """Full (N, N) cosine similarity matrix from L2-normalised embeddings."""
    return emb @ emb.T


def build_mutual_knn_graph(sim: np.ndarray, k: int) -> nx.Graph:
    """
    Build an undirected weighted mutual kNN graph.
    Edge (i,j) exists iff j ∈ kNN(i) AND i ∈ kNN(j).
    Edge weight = max(cos(i,j), 0).
    """
    N = sim.shape[0]
    k = min(k, N - 1)

    # top-k neighbours per node (excluding self)
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    top_k = np.argsort(-s, axis=1)[:, :k]    # (N, k)

    # kNN sets
    knn_sets = [set(top_k[i]) for i in range(N)]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in knn_sets[i]:
            if j > i and i in knn_sets[j]:   # mutual + upper-triangle dedup
                w = max(float(sim[i, j]), 0.0)
                G.add_edge(i, j, weight=w)

    return G


# ─── Conductance ─────────────────────────────────────────────────────────────

def compute_conductance(G: nx.Graph, partition: dict) -> list[float]:
    """
    Conductance φ(S) = cut(S, S̄) / min(vol(S), vol(S̄))
    for each community S in the partition.
    Returns a list of conductance values (one per community, noise-free).
    """
    # Group nodes by community id
    communities: dict[int, set] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)

    degree = dict(G.degree(weight="weight"))
    total_vol = sum(degree.values())
    conductances = []

    for cid, members in communities.items():
        if len(members) == 0 or len(members) == G.number_of_nodes():
            continue
        vol_s = sum(degree.get(v, 0.0) for v in members)
        vol_c = total_vol - vol_s
        if vol_s < 1e-12 or vol_c < 1e-12:
            conductances.append(0.0)
            continue
        cut = sum(
            G[u][v].get("weight", 1.0)
            for u in members
            for v in G[u]
            if v not in members
        )
        conductances.append(cut / min(vol_s, vol_c))

    return conductances if conductances else [0.0]


# ─── Community detection ──────────────────────────────────────────────────────

def run_louvain(G: nx.Graph, gamma: float, n_seeds: int = 3) -> tuple[dict, float]:
    """
    Run Louvain (python-louvain) with resolution γ; repeat n_seeds times and
    return the partition with the highest modularity.
    Returns: (partition dict {node: community_id}, best_modularity).
    """
    try:
        import community as community_louvain
    except ImportError:
        print("[error] python-louvain not installed. Run: pip install python-louvain")
        sys.exit(1)

    if G.number_of_edges() == 0:
        # Trivial: every node is its own community
        return {n: n for n in G.nodes()}, 0.0

    best_Q, best_part = -np.inf, None
    for seed in range(n_seeds):
        try:
            part = community_louvain.best_partition(
                G, resolution=gamma, weight="weight", random_state=seed
            )
            Q = community_louvain.modularity(part, G, weight="weight")
            if Q > best_Q:
                best_Q, best_part = Q, part
        except Exception:
            continue

    if best_part is None:
        return {n: 0 for n in G.nodes()}, 0.0

    return best_part, float(best_Q)


# ─── Spectral analysis ───────────────────────────────────────────────────────

def compute_spectral(G: nx.Graph, n_eigs: int = 30) -> dict:
    """
    Compute the first n_eigs eigenvalues of the normalised Laplacian L_sym.
    Returns:
      lambda2      — Fiedler value (2nd smallest eigenvalue)
      eigenvalues  — list of n_eigs smallest eigenvalues
      eigengaps    — consecutive differences λ_{i+1} − λ_i
      n_components — number of connected components (= #zero eigenvalues)
    """
    n = G.number_of_nodes()
    if n < 4 or G.number_of_edges() == 0:
        return {"lambda2": 0.0, "eigenvalues": [], "eigengaps": [], "n_components": n}

    n_eigs = min(n_eigs, n - 2)

    try:
        L = nx.normalized_laplacian_matrix(
            G, nodelist=sorted(G.nodes()), weight="weight"
        ).astype(float)
        # Use shift-invert mode for smallest eigenvalues; add sigma=0 jitter
        vals, _ = eigsh(L, k=n_eigs, which="SM", tol=1e-5, maxiter=3000)
        vals = np.sort(np.real(vals))
        vals = np.maximum(vals, 0.0)
    except Exception as e:
        return {"lambda2": 0.0, "eigenvalues": [], "eigengaps": [], "n_components": 0}

    n_zero = int(np.sum(vals < 1e-6))
    lambda2 = float(vals[1]) if len(vals) > 1 else 0.0
    eigengaps = [float(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]

    return {
        "lambda2":     lambda2,
        "eigenvalues": vals.tolist(),
        "eigengaps":   eigengaps,
        "n_components": n_zero,
    }


# ─── Main sweep ──────────────────────────────────────────────────────────────

def compute_all(
    family:   str,
    k_list:   list[int],
    gamma_list: list[float],
    n_seeds:  int,
    n_eigs:   int,
    gamma_ref: float,
) -> dict:
    """
    For every model size in the family: sweep k × γ, record all metrics.
    """
    size_dirs = sorted(
        [d for d in os.listdir(os.path.join(EMB_DIR, family))
         if os.path.isdir(os.path.join(EMB_DIR, family, d))],
        key=_size_sort_key,
    )
    if not size_dirs:
        print(f"[error] No size directories found under {EMB_DIR}/{family}/")
        sys.exit(1)

    print(f"[load] Found {len(size_dirs)} sizes: {size_dirs}")
    print(f"[sweep] k-grid={k_list}")
    print(f"[sweep] γ-grid={[round(g,3) for g in gamma_list]}")
    print(f"[sweep] seeds={n_seeds}, n_eigs={n_eigs}, γ_ref={gamma_ref}\n")

    size_results = {}

    for size in size_dirs:
        print(f"\n{'─'*60}")
        print(f"[size] {family} / {size}")
        print(f"{'─'*60}")

        emb, titles = load_model_data(family, size)
        N = len(titles)
        print(f"[load] {N} books")

        sim = cosine_sim_matrix(emb)

        # Data structures:  { k → { gamma → {...stats...} } }
        k_gamma_stats: dict[int, dict[float, dict]] = {}
        # Store partitions for persistence computation
        partitions: dict[tuple, dict] = {}    # (k, gamma) → partition

        for k in k_list:
            print(f"  k={k:<4}  building graph ...", end="", flush=True)
            G = build_mutual_knn_graph(sim, k)
            n_edges  = G.number_of_edges()
            n_comp   = nx.number_connected_components(G)
            gc_size  = max(len(c) for c in nx.connected_components(G))
            print(f"  edges={n_edges}, components={n_comp}, giant={gc_size}")

            # ── Spectral (per k, not per γ) ───────────────────────────────
            sp = compute_spectral(G, n_eigs=n_eigs)

            # ── Community sweep over γ ────────────────────────────────────
            gamma_stats: dict[float, dict] = {}
            for gamma in gamma_list:
                partition, Q = run_louvain(G, gamma, n_seeds=n_seeds)
                conds = compute_conductance(G, partition)
                n_comm = len(set(partition.values()))
                partitions[(k, gamma)] = partition

                gamma_stats[gamma] = {
                    "modularity":           Q,
                    "n_communities":        n_comm,
                    "conductance_median":   float(np.median(conds)),
                    "conductance_p10":      float(np.percentile(conds, 10)),
                    "conductance_mean":     float(np.mean(conds)),
                }

            k_gamma_stats[k] = {
                "graph": {
                    "n_edges":    n_edges,
                    "n_components": n_comp,
                    "giant_component_size": gc_size,
                },
                "spectral": sp,
                "gamma":   gamma_stats,
            }

            # Print per-k summary
            best_Q = max(gamma_stats[g]["modularity"] for g in gamma_list)
            best_cond = min(gamma_stats[g]["conductance_median"] for g in gamma_list)
            print(f"         λ₂={sp['lambda2']:.4f}  "
                  f"best_Q={best_Q:.4f}  best_cond={best_cond:.4f}")

        # ── Persistence across k at gamma_ref ────────────────────────────
        # Find the closest γ in gamma_list to gamma_ref
        gamma_ref_actual = min(gamma_list, key=lambda g: abs(g - gamma_ref))
        ari_list = []
        for i in range(len(k_list) - 1):
            ka, kb = k_list[i], k_list[i + 1]
            pa = partitions.get((ka, gamma_ref_actual))
            pb = partitions.get((kb, gamma_ref_actual))
            if pa is None or pb is None:
                ari_list.append(None)
                continue
            nodes = sorted(pa.keys())
            la = [pa[v] for v in nodes]
            lb = [pb[v] for v in nodes]
            ari_list.append(float(adjusted_rand_score(la, lb)))

        size_results[size] = {
            "n_books":       N,
            "k_gamma":       k_gamma_stats,
            "persistence":   {
                "gamma_ref": gamma_ref_actual,
                "k_pairs":   [f"{k_list[i]}→{k_list[i+1]}" for i in range(len(k_list)-1)],
                "ari":       ari_list,
            },
        }

    return {
        "family":    family,
        "k_list":    k_list,
        "gamma_list": gamma_list,
        "gamma_ref": gamma_ref,
        "n_seeds":   n_seeds,
        "sizes":     size_dirs,
        "per_size":  size_results,
    }


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _k_gamma_matrix(size_data: dict, k_list: list, gamma_list: list, metric: str) -> np.ndarray:
    """Extract a len(k_list) × len(gamma_list) matrix for a given metric."""
    mat = np.full((len(k_list), len(gamma_list)), np.nan)
    for ki, k in enumerate(k_list):
        for gi, g in enumerate(gamma_list):
            v = size_data["k_gamma"].get(k, {}).get("gamma", {}).get(g, {}).get(metric)
            if v is not None:
                mat[ki, gi] = v
    return mat


# ─── Plot: k×γ heatmaps ──────────────────────────────────────────────────────

def plot_heatmaps(results: dict, out_path_mod: str, out_path_cond: str) -> None:
    """
    Two figures: modularity and conductance k×γ heatmaps.
    Each figure = one panel per model size.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("[warn] seaborn not installed — skipping heatmaps. pip install seaborn")
        return

    sizes   = results["sizes"]
    k_list  = results["k_list"]
    g_list  = results["gamma_list"]
    n_s     = len(sizes)

    row_labels = [str(k) for k in k_list]
    col_labels = [f"{g:.2f}" for g in g_list]

    for metric, out_path, title_str, cmap, vmin, vmax in [
        ("modularity",         out_path_mod,  "Modularity Q",             "YlOrRd", None, None),
        ("conductance_median", out_path_cond, "Conductance (median)",     "YlOrRd_r", None, None),
    ]:
        panel_w = max(3.5, len(g_list) * 0.7)
        panel_h = max(3.0, len(k_list) * 0.55)
        fig, axes = plt.subplots(
            1, n_s,
            figsize=(panel_w * n_s, panel_h + 1.2),
            squeeze=False,
        )
        fig.suptitle(
            f"{results['family']} — {title_str} (k × γ heatmap)\n"
            f"(N={list(results['per_size'].values())[0]['n_books']} books)",
            fontsize=12,
        )

        for si, size in enumerate(sizes):
            ax  = axes[0, si]
            mat = _k_gamma_matrix(results["per_size"][size], k_list, g_list, metric)

            # Auto range if not specified
            _vmin = float(np.nanmin(mat)) if vmin is None else vmin
            _vmax = float(np.nanmax(mat)) if vmax is None else vmax

            sns.heatmap(
                mat,
                ax=ax,
                annot=True,
                fmt=".3f",
                xticklabels=col_labels,
                yticklabels=row_labels,
                cmap=cmap,
                vmin=_vmin,
                vmax=_vmax,
                linewidths=0.3,
                linecolor="white",
                cbar_kws={"shrink": 0.75},
            )
            ax.set_title(size, fontsize=11)
            ax.set_xlabel("γ (resolution)")
            ax.set_ylabel("k" if si == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot]  Saved → {out_path}")


# ─── Plot: spectral ───────────────────────────────────────────────────────────

def plot_spectral(results: dict, out_path: str) -> None:
    """
    Two panels:
      Left  — λ₂ (Fiedler value) vs k, one line per model size.
      Right — max eigengap (top-5 gaps) vs k, one line per model size.
    """
    sizes  = results["sizes"]
    k_list = results["k_list"]
    x      = np.arange(len(k_list))

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(k_list) * 1.1) * 1.5, 4.5))
    fig.suptitle(
        f"{results['family']} — Spectral Properties of Mutual kNN Graph",
        fontsize=12,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sizes), 1)))

    for si, size in enumerate(sizes):
        kd = results["per_size"][size]["k_gamma"]

        lambda2s  = [kd[k]["spectral"]["lambda2"] for k in k_list]
        # Max eigengap among positions 1..5 (k-way clusterability signal)
        max_gaps = []
        for k in k_list:
            gaps = kd[k]["spectral"].get("eigengaps", [])
            max_gaps.append(max(gaps[1:6]) if len(gaps) >= 6 else (max(gaps[1:]) if len(gaps) > 1 else 0.0))

        kw = dict(color=colors[si], marker="o", markersize=5, label=size)
        axes[0].plot(x, lambda2s, **kw)
        axes[1].plot(x, max_gaps, **kw)

    for ax, title, ylabel in [
        (axes[0], "Fiedler Value (λ₂)",             "λ₂"),
        (axes[1], "Max Eigengap (positions 1–5)",    "Eigengap"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_list], fontsize=9)
        ax.set_xlabel("k (kNN)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out_path}")


# ─── Plot: persistence ────────────────────────────────────────────────────────

def plot_persistence(results: dict, out_path: str) -> None:
    """
    ARI between adjacent k-partitions at γ_ref, one line per model size.
    """
    sizes  = results["sizes"]
    k_list = results["k_list"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sizes), 1)))

    fig, ax = plt.subplots(figsize=(max(6, (len(k_list) - 1) * 1.4), 4.5))
    fig.suptitle(
        f"{results['family']} — Partition Persistence Across k\n"
        f"(ARI between adjacent k-partitions, γ_ref={results['gamma_ref']})",
        fontsize=12,
    )

    x       = np.arange(len(k_list) - 1)
    k_pairs = [f"{k_list[i]}→{k_list[i+1]}" for i in range(len(k_list) - 1)]

    for si, size in enumerate(sizes):
        aris = results["per_size"][size]["persistence"]["ari"]
        aris_clean = [v if v is not None else float("nan") for v in aris]
        ax.plot(x, aris_clean, color=colors[si], marker="o", markersize=6, label=size)

    ax.set_xticks(x)
    ax.set_xticklabels(k_pairs, fontsize=9, rotation=15, ha="right")
    ax.set_xlabel("Adjacent k-pair")
    ax.set_ylabel("ARI")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out_path}")


# ─── Plot: best-Q and best-conductance vs k (summary bar) ────────────────────

def plot_summary_bar(results: dict, out_path: str) -> None:
    """
    Two bar charts comparing model sizes:
      Left  — best (max over γ) modularity per k, one group per size.
      Right — best (min over γ) median conductance per k, one group per size.
    """
    sizes  = results["sizes"]
    k_list = results["k_list"]
    g_list = results["gamma_list"]
    n_s    = len(sizes)
    n_k    = len(k_list)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_s, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(max(9, n_k * n_s * 0.35), 4.5))
    fig.suptitle(
        f"{results['family']} — Best Modularity & Conductance Across k\n"
        f"(optimised over γ ∈ [{min(g_list):.2f}, {max(g_list):.2f}])",
        fontsize=12,
    )

    bar_w   = 0.8 / max(n_s, 1)
    x       = np.arange(n_k)
    offsets = np.linspace(-(n_s - 1) * bar_w / 2, (n_s - 1) * bar_w / 2, n_s)

    for si, size in enumerate(sizes):
        kd = results["per_size"][size]["k_gamma"]
        best_Q    = [max(kd[k]["gamma"][g]["modularity"]         for g in g_list) for k in k_list]
        best_cond = [min(kd[k]["gamma"][g]["conductance_median"] for g in g_list) for k in k_list]

        kw = dict(width=bar_w, color=colors[si], alpha=0.8, label=size)
        axes[0].bar(x + offsets[si], best_Q,    **kw)
        axes[1].bar(x + offsets[si], best_cond, **kw)

    for ax, title, ylabel in [
        (axes[0], "Best Modularity Q (max over γ)",          "Modularity Q"),
        (axes[1], "Best Conductance (min median over γ)",    "Conductance"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_list], fontsize=9)
        ax.set_xlabel("k (kNN)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out_path}")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    sizes  = results["sizes"]
    k_list = results["k_list"]
    g_list = results["gamma_list"]

    print(f"\n{'='*72}")
    print(f"  {results['family']} — kNN Community Summary")
    print(f"{'='*72}")
    print(f"  {'Size':>26}  {'k':>4}  {'best_Q':>8}  {'best_cond':>10}  {'λ₂':>8}  {'ARI(adj)':>9}")
    print(f"  {'-'*26}  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*9}")

    for size in sizes:
        sd    = results["per_size"][size]
        kd    = sd["k_gamma"]
        aris  = sd["persistence"]["ari"]
        mean_ari = float(np.nanmean([a for a in aris if a is not None])) if aris else float("nan")

        for ki, k in enumerate(k_list):
            best_Q    = max(kd[k]["gamma"][g]["modularity"]         for g in g_list)
            best_cond = min(kd[k]["gamma"][g]["conductance_median"] for g in g_list)
            lam2      = kd[k]["spectral"]["lambda2"]
            size_lbl  = size if ki == 0 else ""
            ari_lbl   = f"{mean_ari:>9.4f}" if ki == 0 else " " * 9
            print(f"  {size_lbl:>26}  {k:>4}  {best_Q:>8.4f}  {best_cond:>10.4f}  {lam2:>8.4f}  {ari_lbl}")

    print(f"{'='*72}\n")


# ─── Save JSON (strip per-book arrays) ───────────────────────────────────────

def _serialisable(obj):
    """Recursively convert numpy types → native Python for json.dump."""
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, dict):         return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [_serialisable(v) for v in obj]
    return obj


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="kNN graph community structure analysis across model sizes"
    )
    parser.add_argument("--model_family", required=True,
                        help="e.g. Qwen3-Embedding, Pythia, Cerebras-GPT")
    parser.add_argument("--k_min",    type=int,   default=5)
    parser.add_argument("--k_max",    type=int,   default=80)
    parser.add_argument("--n_k",      type=int,   default=8,
                        help="Number of k values (log-spaced)")
    parser.add_argument("--gamma_min", type=float, default=0.5)
    parser.add_argument("--gamma_max", type=float, default=2.0)
    parser.add_argument("--n_gamma",  type=int,   default=7,
                        help="Number of γ resolution values (linear-spaced)")
    parser.add_argument("--n_seeds",  type=int,   default=3,
                        help="Louvain seeds per (k, γ) cell")
    parser.add_argument("--n_eigs",   type=int,   default=30,
                        help="Eigenvalues to compute for spectral analysis")
    parser.add_argument("--gamma_ref", type=float, default=1.0,
                        help="Reference γ for persistence across k")
    args = parser.parse_args()

    # Build grids
    k_list     = sorted(set(
        int(round(v))
        for v in np.geomspace(args.k_min, args.k_max, args.n_k)
    ))
    gamma_list = [
        round(v, 4)
        for v in np.linspace(args.gamma_min, args.gamma_max, args.n_gamma)
    ]

    out_dir = os.path.join(BASE_OUT, args.model_family)
    os.makedirs(out_dir, exist_ok=True)

    # ── Compute ───────────────────────────────────────────────────────────────
    results = compute_all(
        family=args.model_family,
        k_list=k_list,
        gamma_list=gamma_list,
        n_seeds=args.n_seeds,
        n_eigs=args.n_eigs,
        gamma_ref=args.gamma_ref,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print_table(results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_out = os.path.join(out_dir, "results.json")
    with open(json_out, "w") as f:
        json.dump(_serialisable(results), f, indent=2)
    print(f"[save]  Summary → {json_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_heatmaps(
        results,
        out_path_mod  = os.path.join(out_dir, "heatmap_modularity.png"),
        out_path_cond = os.path.join(out_dir, "heatmap_conductance.png"),
    )
    plot_spectral(  results, os.path.join(out_dir, "spectral.png"))
    plot_persistence(results, os.path.join(out_dir, "persistence.png"))
    plot_summary_bar(results, os.path.join(out_dir, "summary_bar.png"))


if __name__ == "__main__":
    main()
