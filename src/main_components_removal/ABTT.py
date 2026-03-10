"""
ABTT.py — All-But-The-Top postprocessing for embedding spaces.
===============================================================
Reference: Mu et al., "All-but-the-Top: Simple and Effective Postprocessing
for Word Representations", ICLR 2018.

Algorithm (applied to a batch of embeddings):
  1. Subtract the global mean embedding  →  mean-centred embeddings
  2. Project out the top-n principal directions computed over the full corpus

The mean vector and principal directions are loaded from the precomputed files
produced by average_embedding.py and pca_embedding.py:
  out/main_component_removal/avg_embed.pkl
  out/main_component_removal/principal_directions.pkl

Public API
----------
  abtt(family, size, embeddings, n) -> np.ndarray
      family     : str   — model family, e.g. "Qwen3-Embedding"
      size       : str   — model size,   e.g. "8B"
      embeddings : array-like, shape (N, D) or (D,)
      n          : int   — number of top principal directions to remove
      returns    : np.ndarray, same shape as input, dtype float64
"""

import os
import pickle

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_DIR = os.path.join(_ROOT, "out", "main_component_removal")

_AVG_EMBED_PATH = os.path.join(_OUT_DIR, "avg_embed.pkl")
_PCA_PATH       = os.path.join(_OUT_DIR, "principal_directions.pkl")

# ─── Lazy-loaded caches ───────────────────────────────────────────────────────
_avg_embed_cache: dict | None = None
_pca_cache:       dict | None = None


def _load_avg_embed() -> dict:
    global _avg_embed_cache
    if _avg_embed_cache is None:
        with open(_AVG_EMBED_PATH, "rb") as f:
            _avg_embed_cache = pickle.load(f)
    return _avg_embed_cache


def _load_pca() -> dict:
    global _pca_cache
    if _pca_cache is None:
        with open(_PCA_PATH, "rb") as f:
            _pca_cache = pickle.load(f)
    return _pca_cache


# ─── Public function ──────────────────────────────────────────────────────────

def abtt(
    family: str,
    size: str,
    embeddings: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Apply All-But-The-Top postprocessing.

    Parameters
    ----------
    family     : model family name (must match avg_embed.pkl / principal_directions.pkl)
    size       : model size name
    embeddings : (N, D) or (D,) array of raw (or L2-normalised) embeddings
    n          : number of top principal directions to project out (≥ 0)

    Returns
    -------
    np.ndarray of the same shape as `embeddings`, dtype float64.
    The returned vectors are NOT re-normalised; callers may normalise if needed.
    """
    X = np.array(embeddings, dtype=np.float64)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[np.newaxis, :]   # treat single vector as (1, D)

    # 1. Mean-centre
    mean_vec = np.array(_load_avg_embed()[family][size], dtype=np.float64)
    X_c = X - mean_vec   # broadcast over N

    # 2. Project out top-n principal directions
    if n > 0:
        # directions: (K, D), rows are unit principal vectors (from SVD of Vt)
        directions = np.array(_load_pca()[family][size], dtype=np.float64)
        top_dirs = directions[:n]                  # (n, D)
        # Projection: X_c -= (X_c @ top_dirs.T) @ top_dirs
        # shape: (N, n) @ (n, D) = (N, D)
        X_c -= (X_c @ top_dirs.T) @ top_dirs

    return X_c.squeeze() if squeeze else X_c


# ─── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    family = "Qwen3-Embedding"
    size   = "0.6B"
    n      = 5

    emb_dir = os.path.join(_ROOT, "outputs_embeddings_all_with_chunks", family, size)
    paths   = sorted(glob.glob(os.path.join(emb_dir, "**", "*.pkl"), recursive=True))[:10]

    raw = []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        e = d.get("embedding")
        if e is not None:
            raw.append(np.array(e, dtype=np.float64))

    X_raw  = np.stack(raw)
    X_abtt = abtt(family, size, X_raw, n)

    print(f"Input  shape : {X_raw.shape}")
    print(f"Output shape : {X_abtt.shape}")
    print(f"Input  avg‖·‖ : {np.linalg.norm(X_raw,  axis=1).mean():.4f}")
    print(f"Output avg‖·‖ : {np.linalg.norm(X_abtt, axis=1).mean():.4f}")
    print(f"Output mean (should be ≈0): {np.abs(X_abtt.mean(axis=0)).max():.6f}")
