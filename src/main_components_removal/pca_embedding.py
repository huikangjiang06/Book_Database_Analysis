"""
pca_embedding.py
================
For every model family and size:
  1. Load all 500 L2-normalised book embeddings.
  2. Mean-centre them (subtract the per-model mean vector).
  3. Run PCA via thin SVD on the centred matrix (efficient when N < D).
  4. Store results.

Outputs:
  out/main_component_removal/principal_directions.pkl
      {family: {size: np.ndarray shape (K, D)}}
      Rows are principal directions ordered by descending variance.
      K = min(N, D) − 1  =  499  (rank of centred N×D matrix with N=500).

  out/main_component_removal/eigenvalues.json
      {family: {size: [λ1, λ2, …, λ_K]}}
      Eigenvalues of the (unbiased) covariance matrix: λ_i = s_i² / (N−1).
      Sum of all λ_i equals the total variance (trace of covariance).
"""

import glob
import json
import os
import pickle
import re

import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR   = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
OUT_DIR   = os.path.join(ROOT, "out", "main_component_removal")
OUT_PKL        = os.path.join(OUT_DIR, "principal_directions.pkl")
OUT_JSON       = os.path.join(OUT_DIR, "eigenvalues.json")
OUT_JSON_TOP20 = os.path.join(OUT_DIR, "eigenvalues_top_20.json")

FAMILIES  = ["Cerebras-GPT", "OpenAI", "Pythia", "Qwen2.5", "Qwen3-Embedding"]


def _size_sort_key(s: str) -> float:
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def get_sizes(family: str) -> list[str]:
    base = os.path.join(EMB_DIR, family)
    return sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d)) and d != "chunks"],
        key=_size_sort_key,
    )


def load_embeddings(family: str, size: str) -> np.ndarray:
    """Return L2-normalised (N, D) float64 matrix."""
    paths = sorted(glob.glob(
        os.path.join(EMB_DIR, family, size, "**", "*.pkl"), recursive=True
    ))
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        emb = d.get("embedding")
        if emb is None:
            emb = d.get("book_embedding")
        if emb is None:
            continue
        emb = np.array(emb, dtype=np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        rows.append(emb)
    return np.stack(rows)   # (N, D)


def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-centre X and return (principal_directions, eigenvalues).

    principal_directions : (K, D)  — rows ordered by descending variance
    eigenvalues          : (K,)    — λ_i = s_i² / (N−1), descending
    where K = min(N, D) for the thin SVD (effectively rank = N−1 after centring).
    """
    N = X.shape[0]
    X_c = X - X.mean(axis=0)           # (N, D) mean-centred

    # Thin SVD: X_c = U @ diag(s) @ Vt
    # Vt rows are principal directions; s² / (N-1) are eigenvalues.
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)  # shapes: s (K,), Vt (K,D)

    eigenvalues = (s ** 2) / (N - 1)
    return Vt, eigenvalues   # principal_directions (K,D), eigenvalues (K,)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_directions: dict[str, dict[str, np.ndarray]] = {}
    all_eigenvalues: dict[str, dict[str, list[float]]] = {}

    for family in FAMILIES:
        all_directions[family]  = {}
        all_eigenvalues[family] = {}
        sizes = get_sizes(family)
        print(f"\n[{family}]  {len(sizes)} sizes")

        for size in sizes:
            X = load_embeddings(family, size)
            Vt, eigs = run_pca(X)

            all_directions[family][size]  = Vt.astype(np.float32)   # save as float32 to halve file size
            all_eigenvalues[family][size] = [round(float(v), 8) for v in eigs]

            explained_top1  = float(eigs[0]  / eigs.sum()) * 100
            explained_top10 = float(eigs[:10].sum() / eigs.sum()) * 100
            print(
                f"  {size:>25}  N={X.shape[0]:>4}  D={X.shape[1]:>5}  K={Vt.shape[0]:>4}"
                f"  top-1={explained_top1:.1f}%  top-10={explained_top10:.1f}%"
            )

    with open(OUT_PKL, "wb") as fh:
        pickle.dump(all_directions, fh)
    print(f"\n[done]  principal_directions → {OUT_PKL}")

    with open(OUT_JSON, "w") as fh:
        json.dump(all_eigenvalues, fh, indent=2)
    print(f"[done]  eigenvalues          → {OUT_JSON}")

    top20 = {
        fam: {size: vals[:20] for size, vals in sizes.items()}
        for fam, sizes in all_eigenvalues.items()
    }
    with open(OUT_JSON_TOP20, "w") as fh:
        json.dump(top20, fh, indent=2)
    print(f"[done]  eigenvalues_top_20   → {OUT_JSON_TOP20}")


if __name__ == "__main__":
    main()
