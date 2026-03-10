"""
average_cosine_similarity.py
=============================
Sweep all model families and sizes. For each model, compute the average
cosine similarity over every pair of book embeddings (upper-triangle only).

Output:
  out/main_component_removal/avg_cos_sim.json
  {
    "Cerebras-GPT": {"111M": 0.123, "256M": 0.118, ...},
    "OpenAI":       {"text-embedding-3-large": 0.031, ...},
    ...
  }

High average cosine similarity → anisotropic space (embeddings cluster in a cone).
"""

import glob
import json
import os
import pickle
import re

import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
OUT_FILE = os.path.join(ROOT, "out", "main_component_removal", "avg_cos_sim.json")

FAMILIES = ["Cerebras-GPT", "OpenAI", "Pythia", "Qwen2.5", "Qwen3-Embedding"]


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
    """Return L2-normalised (N, D) matrix."""
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


def avg_cosine_similarity(X: np.ndarray) -> float:
    """
    Mean cosine similarity over all N*(N-1)/2 upper-triangle pairs.
    Since X is already L2-normalised, cos(i,j) = dot(X[i], X[j]).
    Mean of upper triangle = (sum of all dot products - trace) / (N*(N-1))
    """
    N = X.shape[0]
    gram = X @ X.T          # (N, N)  — all pairwise cos sims
    total = gram.sum() - np.trace(gram)   # exclude diagonal (self-sim = 1)
    return float(total / (N * (N - 1)))   # each pair counted twice


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    results: dict[str, dict[str, float]] = {}

    for family in FAMILIES:
        results[family] = {}
        sizes = get_sizes(family)
        print(f"\n[{family}]  {len(sizes)} sizes")
        for size in sizes:
            X = load_embeddings(family, size)
            acs = avg_cosine_similarity(X)
            results[family][size] = round(acs, 6)
            print(f"  {size:>25}  N={X.shape[0]:>4}  D={X.shape[1]:>5}  avg_cos={acs:.6f}")

    with open(OUT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done]  → {OUT_FILE}")


if __name__ == "__main__":
    main()
