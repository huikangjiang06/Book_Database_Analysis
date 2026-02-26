"""
Emergence Pipeline
==================
Step 1: Load book embeddings for a given model.
Step 2: UMAP dimensionality reduction (→ 5D).
Step 3: HDBSCAN clustering on reduced embeddings.
Step 4: Save results (embeddings, UMAP coords, cluster labels, metadata) to out/.

Usage:
    python pipeline.py --model_family Qwen3-Embedding --model_size 0.6B
    python pipeline.py --model_family OpenAI --model_size text-embedding-3-small
"""

import argparse
import glob
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_DIR = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Step 1: Load Embeddings ──────────────────────────────────────────────────

def load_embeddings(model_family: str, model_size: str) -> pd.DataFrame:
    """
    Load all .pkl files for the given model family/size.
    Returns a DataFrame with columns:
        book_title, genre, pkl_path, embedding (np.ndarray)
    """
    base = os.path.join(EMBEDDINGS_DIR, model_family, model_size)
    pkl_files = glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True)

    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl files found under {base}. "
            f"Check --model_family and --model_size."
        )

    records = []
    for path in sorted(pkl_files):
        with open(path, "rb") as f:
            d = pickle.load(f)

        emb = d.get("embedding")
        if emb is None:
            # fallback key used in some older pkl files
            emb = d.get("book_embedding")
        if emb is None:
            print(f"  [WARN] No embedding in {path}, skipping.")
            continue

        records.append(
            {
                "book_title": d.get("book_title", os.path.splitext(os.path.basename(path))[0]),
                "genre": d.get("genre", "Unknown"),
                "pkl_path": path,
                "embedding": np.array(emb, dtype=np.float32),
                "model_family": d.get("model_family", model_family),
                "model_size": d.get("model_size", model_size),
            }
        )

    print(f"[load] Loaded {len(records)} books from {base}")
    return pd.DataFrame(records)


# ─── Step 2: UMAP ─────────────────────────────────────────────────────────────

def run_umap(
    df: pd.DataFrame,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embeddings to `n_components` dimensions using UMAP.
    Returns reduced array of shape (n_books, n_components).

    Notes on parameter choices:
    - n_components=5: Low enough to avoid curse of dimensionality for HDBSCAN,
      high enough to preserve structure better than 2D.
    - min_dist=0.0: Maximises clustering fidelity (points in the same cluster
      compress tightly). We use a separate 2D run for visualisation.
    - metric=cosine: Book embeddings are L2-normalised, so cosine distance is
      the natural similarity measure.
    """
    print(f"[umap] Reducing {len(df)} × {df['embedding'].iloc[0].shape[0]}D "
          f"→ {n_components}D  (n_neighbors={n_neighbors}, metric={metric})")

    import umap
    
    X = np.stack(df["embedding"].values)  # (N, D)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    X_reduced = reducer.fit_transform(X)
    print(f"[umap] Done. Output shape: {X_reduced.shape}")
    return X_reduced


def run_umap_2d(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Separate 2D UMAP purely for visualisation (min_dist=0.1 for readability).
    """
    import umap
    
    print(f"[umap-2d] Reducing to 2D for visualisation ...")
    X = np.stack(df["embedding"].values)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(X)


# ─── Step 3: HDBSCAN ──────────────────────────────────────────────────────────

def run_hdbscan(
    X_reduced: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Cluster the UMAP-reduced embeddings with HDBSCAN.
    Returns integer label array; -1 = noise/outlier.

    Notes:
    - min_cluster_size=5: With 500 books, a cluster of 5 is a reasonable floor.
    - min_samples=3: Controls noise sensitivity; lower → fewer noise points.
    - We use euclidean on the UMAP space (not cosine) because UMAP has already
      encoded the cosine structure into Euclidean distances.
    """
    print(f"[hdbscan] Clustering {X_reduced.shape[0]} points "
          f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})")

    import hdbscan
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",  # Excess of Mass: stable, general-purpose
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[hdbscan] Found {n_clusters} clusters, {n_noise} noise points "
          f"({100*n_noise/len(labels):.1f}% of books)")
    return labels, clusterer


# ─── Step 4: Save Results ─────────────────────────────────────────────────────

def save_results(
    df: pd.DataFrame,
    X_umap5: np.ndarray,
    X_umap2: np.ndarray,
    labels: np.ndarray,
    model_family: str,
    model_size: str,
    clusterer=None,
) -> str:
    """
    Save a comprehensive results file to out/<model_family>_<model_size>_clusters.pkl
    and a human-readable summary to out/<model_family>_<model_size>_clusters.csv
    """
    tag = f"{model_family}_{model_size}".replace("/", "_").replace(" ", "_")
    run_dir = os.path.join(OUT_DIR, model_family, model_size)
    os.makedirs(run_dir, exist_ok=True)
    out_pkl = os.path.join(run_dir, "clusters.pkl")
    out_csv = os.path.join(run_dir, "clusters.csv")
    out_json = os.path.join(run_dir, "summary.json")

    # Build result DataFrame
    result = df[["book_title", "genre", "pkl_path", "model_family", "model_size"]].copy()
    result["cluster"] = labels
    result["umap2_x"] = X_umap2[:, 0]
    result["umap2_y"] = X_umap2[:, 1]
    for i in range(X_umap5.shape[1]):
        result[f"umap5_{i}"] = X_umap5[:, i]

    # Extract condensed tree as a DataFrame (columns: parent, child, lambda_val, child_size)
    condensed_tree_df = None
    if clusterer is not None:
        condensed_tree_df = clusterer.condensed_tree_.to_pandas()
        out_ctree = os.path.join(run_dir, "condensed_tree.csv")
        condensed_tree_df.to_csv(out_ctree, index=False)
        print(f"[save] condensed_tree.csv  ({len(condensed_tree_df)} rows)")

    # Compute stability scores before using them
    stability_scores = {}
    if clusterer is not None and hasattr(clusterer, "cluster_persistence_"):
        for cid, score in enumerate(clusterer.cluster_persistence_):
            stability_scores[str(cid)] = round(float(score), 6)

    # Save full pickle (includes original embeddings for later steps)
    full = {
        "result_df": result,
        "embeddings": np.stack(df["embedding"].values),
        "umap5": X_umap5,
        "umap2": X_umap2,
        "labels": labels,
        "model_family": model_family,
        "model_size": model_size,
        "condensed_tree": condensed_tree_df,  # pd.DataFrame or None
        "cluster_stability": stability_scores,  # {str(cid): float}
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(full, f)

    # Save CSV (human readable)
    result.to_csv(out_csv, index=False)

    # Save JSON summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    cluster_sizes = result[result["cluster"] >= 0]["cluster"].value_counts().to_dict()

    summary = {
        "model_family": model_family,
        "model_size": model_size,
        "n_books": len(result),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": {str(k): int(v) for k, v in sorted(cluster_sizes.items())},
        "cluster_stability": stability_scores,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[save] Results saved to: {run_dir}/\n  clusters.pkl\n  clusters.csv\n  summary.json\n  condensed_tree.csv")

    # Print cluster preview
    print("\n[preview] Cluster composition (top 3 books per cluster):")
    for cid in sorted(set(labels)):
        label = f"Cluster {cid}" if cid >= 0 else "Noise"
        books = result[result["cluster"] == cid]["book_title"].tolist()
        print(f"  {label} ({len(books)} books): {books[:3]}")

    return run_dir


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emergence Embedding Pipeline")
    parser.add_argument("--model_family", type=str, default="Qwen3-Embedding",
                        help="Model family folder name (e.g. Qwen3-Embedding, OpenAI, Pythia)")
    parser.add_argument("--model_size", type=str, default="0.6B",
                        help="Model size subfolder (e.g. 0.6B, 4B, text-embedding-3-small)")
    parser.add_argument("--umap_components", type=int, default=5)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--hdbscan_min_cluster", type=int, default=5)
    parser.add_argument("--hdbscan_min_samples", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Emergence Pipeline: {args.model_family} / {args.model_size}")
    print(f"{'='*60}\n")

    # Step 1: Load
    df = load_embeddings(args.model_family, args.model_size)

    # Step 2: UMAP (5D for clustering + 2D for visualisation)
    X_umap5 = run_umap(df, n_components=args.umap_components,
                       n_neighbors=args.umap_neighbors)
    X_umap2 = run_umap_2d(df, n_neighbors=args.umap_neighbors)

    # Step 3: HDBSCAN
    labels, clusterer = run_hdbscan(
        X_umap5,
        min_cluster_size=args.hdbscan_min_cluster,
        min_samples=args.hdbscan_min_samples,
    )

    # Step 4: Save
    out_path = save_results(df, X_umap5, X_umap2, labels,
                            args.model_family, args.model_size,
                            clusterer=clusterer)

    print(f"\n[done] Pipeline complete. Next step: c-TF-IDF topic extraction.")
    return out_path


if __name__ == "__main__":
    main()
