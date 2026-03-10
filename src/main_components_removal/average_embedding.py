"""
average_embedding.py
====================
Sweep all model families and sizes. For each model, compute the mean
embedding vector (centroid) across all book embeddings.

Output:
  out/main_component_removal/avg_embed.pkl
  {
    "Cerebras-GPT": {
        "111M":  np.ndarray shape (D,),   # raw (un-normalised) mean
        "256M":  np.ndarray shape (D,),
        ...
    },
    "OpenAI": { ... },
    ...
  }

The mean embedding is the first-order statistic of anisotropy: if it has
large norm the space has a strong directional bias. It is later subtracted
("mean-centering") as the simplest form of main-component removal.
"""

import glob
import os
import pickle
import re

import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR  = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
OUT_FILE = os.path.join(ROOT, "out", "main_component_removal", "avg_embed.pkl")

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


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    results: dict[str, dict[str, np.ndarray]] = {}

    for family in FAMILIES:
        results[family] = {}
        sizes = get_sizes(family)
        print(f"\n[{family}]  {len(sizes)} sizes")
        for size in sizes:
            X = load_embeddings(family, size)
            mean_emb = X.mean(axis=0)          # shape (D,)
            mean_norm = float(np.linalg.norm(mean_emb))
            results[family][size] = mean_emb
            print(f"  {size:>25}  N={X.shape[0]:>4}  D={X.shape[1]:>5}"
                  f"  ||mean||={mean_norm:.6f}")

    with open(OUT_FILE, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[done]  → {OUT_FILE}")


if __name__ == "__main__":
    main()
