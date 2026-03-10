"""
retrieval_score.py
==================
Embedding-based retrieval accuracy across model sizes.

For each grouping (genre, bookshelf, subject):
  - Randomly select num_instances query books.
  - For each query:
      • target     = 1 book that shares ≥1 category with the query
      • distractors = num_candidates books that share NO category with the query
      • candidate pool = [target] + distractors  (shuffled internally)
  - Score = fraction of queries where the query's nearest neighbour in the pool
            is the target.

Category sources:
  genre     — directory name under processed_books/ (always available)
  bookshelf — "bookshelves" list from out/gutenberg_meta/meta.jsonl
  subject   — "subjects" list from out/gutenberg_meta/meta.jsonl

Outputs (under out/retrieval_score/<family>/):
  results.json          — accuracy per (grouping, size)
  retrieval_score.png   — line plot of accuracy vs model size, one line per grouping

Usage:
    python src/retrieval_score/retrieval_score.py --model_family Qwen3-Embedding
    python src/retrieval_score/retrieval_score.py --model_family Pythia \\
        --num_instances 200 --num_candidates 19
"""

import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_components_removal"))
from ABTT import abtt as apply_abtt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR   = os.path.join(ROOT, "outputs_embeddings_all_with_chunks")
META_FILE = os.path.join(ROOT, "out", "gutenberg_meta", "meta.jsonl")
BASE_OUT  = os.path.join(ROOT, "out", "retrieval_score")

GROUPINGS = ["genre", "bookshelf", "subject"]

GROUPING_COLORS = {
    "genre":     "#2196F3",
    "bookshelf": "#FF9800",
    "subject":   "#4CAF50",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _size_sort_key(s: str) -> float:
    m = re.match(r"([\d.]+)\s*([BbMmKk]?)", s.strip())
    if not m:
        return float("inf")
    v, suffix = float(m.group(1)), m.group(2).upper()
    return v * (1000 if suffix == "B" else 1 if suffix == "M" else 0.001 if suffix == "K" else 1)


def get_sizes(family: str) -> list[str]:
    base = os.path.join(EMB_DIR, family)
    sizes = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d != "chunks"
    ]
    return sorted(sizes, key=_size_sort_key)


def load_meta() -> dict[str, dict]:
    """book_name (NFC) → {genre: str, bookshelves: set, subjects: set}"""
    meta: dict[str, dict] = {}
    if not os.path.exists(META_FILE):
        return meta
    with open(META_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bn = nfc(rec["book_name"])
            meta[bn] = {
                "genre":      nfc(rec.get("genre", "")),
                "bookshelves": set(rec.get("bookshelves", [])),
                "subjects":    set(rec.get("subjects", [])),
            }
    return meta


def load_book_names_with_genre(family: str, size: str) -> list[tuple[str, str]]:
    """Return [(book_name, genre), …] from the pkl directory tree."""
    base  = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    out = []
    for p in paths:
        bn    = nfc(os.path.splitext(os.path.basename(p))[0])
        genre = nfc(os.path.basename(os.path.dirname(p)))
        out.append((bn, genre))
    return out


def load_embeddings(family: str, size: str, abtt_n: int = 0) -> dict[str, np.ndarray]:
    """book_name (NFC) → L2-normalised embedding (with optional ABTT postprocessing)."""
    base  = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    embs: dict[str, np.ndarray] = {}
    book_names: list[str] = []
    raw: list[np.ndarray] = []
    for path in paths:
        with open(path, "rb") as f:
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
        bn = nfc(os.path.splitext(os.path.basename(path))[0])
        book_names.append(bn)
        raw.append(emb)

    if not raw:
        return embs

    X = np.stack(raw)   # (N, D)
    if abtt_n > 0:
        X = apply_abtt(family, size, X, abtt_n)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    for bn, emb in zip(book_names, X):
        embs[bn] = emb
    return embs


# ─── Category helpers ─────────────────────────────────────────────────────────

def cats_for(book: str, grouping: str, meta: dict) -> frozenset:
    rec = meta.get(book, {})
    if grouping == "genre":
        g = rec.get("genre", "")
        return frozenset([g]) if g else frozenset()
    elif grouping == "bookshelf":
        return frozenset(rec.get("bookshelves", set()))
    else:
        return frozenset(rec.get("subjects", set()))


# ─── Instance building ────────────────────────────────────────────────────────

def build_instances(
    all_books: list[str],
    grouping: str,
    meta: dict,
    num_instances: int,
    num_candidates: int,
    seed: int,
) -> list[tuple[str, str, list[str]]]:
    """
    Returns list of (query, target, [distractor × num_candidates]).
    Tries each book as query once; skips books without valid sharers/non-sharers.
    """
    rng = random.Random(seed)
    shuffled = list(all_books)
    rng.shuffle(shuffled)

    instances: list[tuple[str, str, list[str]]] = []

    for query in shuffled:
        if len(instances) >= num_instances:
            break

        q_cats = cats_for(query, grouping, meta)
        if not q_cats:
            continue

        sharers     = [b for b in all_books if b != query and q_cats & cats_for(b, grouping, meta)]
        non_sharers = [b for b in all_books if b != query and not (q_cats & cats_for(b, grouping, meta))]

        if not sharers or len(non_sharers) < num_candidates:
            continue

        target      = rng.choice(sharers)
        distractors = rng.sample(non_sharers, num_candidates)
        instances.append((query, target, distractors))

    return instances


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_instances(
    embs: dict[str, np.ndarray],
    instances: list[tuple[str, str, list[str]]],
) -> float:
    correct = total = 0
    for query, target, distractors in instances:
        if query not in embs or target not in embs:
            continue
        if any(d not in embs for d in distractors):
            continue
        q    = embs[query]
        pool = [target] + distractors   # target always at index 0
        sims = np.array([float(q @ embs[b]) for b in pool])
        if int(np.argmax(sims)) == 0:
            correct += 1
        total += 1
    return correct / total if total > 0 else float("nan")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(
    results: dict[str, dict[str, float]],
    sizes: list[str],
    out_dir: str,
    family: str,
    num_candidates: int,
):
    chance = 1.0 / (num_candidates + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for grouping in GROUPINGS:
        size_acc = results[grouping]
        ys = [size_acc.get(s, float("nan")) for s in sizes]
        ax.plot(sizes, ys, marker="o", label=grouping,
                color=GROUPING_COLORS[grouping], linewidth=2)

    ax.axhline(chance, color="gray", linestyle="--", linewidth=1,
               label=f"random ({chance:.0%})")

    ax.set_title(f"Retrieval Accuracy — {family}")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "retrieval_score.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family",   required=True)
    parser.add_argument("--num_instances",  type=int, default=350,
                        help="Number of query instances per grouping")
    parser.add_argument("--num_candidates", type=int, default=49,
                        help="Number of distractor books (pool size = num_candidates + 1)")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--abtt",           type=int, default=0, metavar="N",
                        help="Apply ABTT: remove top-N principal directions (0 = disabled)")
    args = parser.parse_args()

    family  = args.model_family
    suffix  = f"_abtt" if args.abtt > 0 else ""
    out_dir = os.path.join(BASE_OUT, family + suffix)
    os.makedirs(out_dir, exist_ok=True)

    sizes = get_sizes(family)
    print(f"[load] Found {len(sizes)} sizes: {sizes}")

    # Load Gutenberg metadata (bookshelves / subjects)
    meta = load_meta()
    print(f"[load] Gutenberg meta: {len(meta)} books")

    # Get full book list + genre from first size directory
    book_genre_pairs = load_book_names_with_genre(family, sizes[0])
    all_books = [bn for bn, _ in book_genre_pairs]
    print(f"[load] {len(all_books)} books in {family}/{sizes[0]}")

    # Backfill genre in meta from pkl directory structure (covers books with no meta)
    for bn, genre in book_genre_pairs:
        if bn not in meta:
            meta[bn] = {"genre": genre, "bookshelves": set(), "subjects": set()}
        elif not meta[bn].get("genre"):
            meta[bn]["genre"] = genre

    # ── Build instances once (same for every model size) ─────────────────────
    instances_map: dict[str, list] = {}
    for g in GROUPINGS:
        inst = build_instances(all_books, g, meta,
                               args.num_instances, args.num_candidates, args.seed)
        instances_map[g] = inst
        print(f"[instances] {g:>10}: {len(inst):>4} valid instances")

    # ── Score each size ───────────────────────────────────────────────────────
    results: dict[str, dict[str, float]] = {g: {} for g in GROUPINGS}

    col_w = 12
    header = f"{'Size':>{col_w}}" + "".join(f"  {g:>{col_w}}" for g in GROUPINGS)
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for size in sizes:
        print(f"  loading {family}/{size} …", end="", flush=True)
        embs = load_embeddings(family, size, abtt_n=args.abtt)
        print(f"  {len(embs)} embs")
        row = f"{size:>{col_w}}"
        for g in GROUPINGS:
            acc = score_instances(embs, instances_map[g])
            results[g][size] = acc
            row += f"  {acc:>{col_w}.1%}" if not np.isnan(acc) else f"  {'N/A':>{col_w}}"
        print(row)

    print(sep)

    # ── Save results ─────────────────────────────────────────────────────────
    out_json = os.path.join(out_dir, "results.json")
    serial = {
        "family":         family,
        "num_instances":  args.num_instances,
        "num_candidates": args.num_candidates,
        "seed":           args.seed,
        "results": {
            g: {s: (v if not np.isnan(v) else None) for s, v in sv.items()}
            for g, sv in results.items()
        },
    }
    with open(out_json, "w") as fh:
        json.dump(serial, fh, indent=2)
    print(f"\n[save]  Results → {out_json}")

    plot_results(results, sizes, out_dir, family, args.num_candidates)


if __name__ == "__main__":
    main()
