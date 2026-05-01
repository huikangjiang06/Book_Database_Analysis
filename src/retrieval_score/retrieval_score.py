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
    python src/retrieval_score/retrieval_score.py --model_family Pythia \\
        --embedding_level chunk --sample_n 5000
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
BOOK_LEVEL = "book"
CHUNK_LEVEL = "chunk"

GROUPING_COLORS = {
    "genre":     "#2196F3",
    "bookshelf": "#FF9800",
    "subject":   "#4CAF50",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


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


def _normalise_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def chunk_id(book_name: str, chunk_idx: int) -> str:
    return f"{book_name}::chunk_{chunk_idx:05d}"


def collect_chunk_ids(family: str, size: str) -> dict[str, list[str]]:
    """book_name -> chunk IDs available for this model size."""
    base = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    out: dict[str, list[str]] = {}
    for path in paths:
        d = load_pickle(path)
        chunks = d.get("chunk_embeddings")
        if chunks is None:
            continue
        book_name = nfc(os.path.splitext(os.path.basename(path))[0])
        out[book_name] = [chunk_id(book_name, i) for i in range(len(chunks))]
    return out


def sample_common_chunk_ids(
    family: str,
    sizes: list[str],
    sample_n: int,
    seed: int,
) -> set[str]:
    """Stratified sample of chunk IDs common to every size, preserving book coverage."""
    print(f"[sample] Collecting common chunk IDs across {len(sizes)} sizes...")
    first_by_book: dict[str, list[str]] | None = None
    common: set[str] | None = None
    for size in sizes:
        by_book = collect_chunk_ids(family, size)
        ids = {cid for ids_for_book in by_book.values() for cid in ids_for_book}
        if first_by_book is None:
            first_by_book = by_book
        common = ids if common is None else common & ids
        print(f"  {size:>25}: {len(ids):>7} chunks")

    if not common:
        raise ValueError(f"No common chunk IDs found for {family}.")

    ordered_by_book = {
        book: [cid for cid in ids if cid in common]
        for book, ids in first_by_book.items()
    }
    ordered_by_book = {book: ids for book, ids in ordered_by_book.items() if ids}
    ordered_all = [cid for ids in ordered_by_book.values() for cid in ids]

    if sample_n <= 0 or sample_n >= len(ordered_all):
        selected = set(ordered_all)
        print(f"[sample] Common chunks: {len(ordered_all)}; using all")
        return selected

    rng = random.Random(seed)
    selected: list[str] = []

    # First pass: one chunk per book when the budget allows it.
    books = list(ordered_by_book)
    rng.shuffle(books)
    for book in books[:min(sample_n, len(books))]:
        selected.append(rng.choice(ordered_by_book[book]))

    remaining_budget = sample_n - len(selected)
    if remaining_budget > 0:
        selected_set = set(selected)
        remaining = [cid for cid in ordered_all if cid not in selected_set]
        selected.extend(rng.sample(remaining, min(remaining_budget, len(remaining))))

    print(f"[sample] Common chunks: {len(ordered_all)}; using {len(selected)} (seed={seed})")
    return set(selected)


def load_embeddings(
    family: str,
    size: str,
    abtt_n: int = 0,
    embedding_level: str = BOOK_LEVEL,
    selected_chunk_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """book_name (NFC) → L2-normalised embedding (with optional ABTT postprocessing)."""
    base  = os.path.join(EMB_DIR, family, size)
    paths = sorted(glob.glob(os.path.join(base, "**", "*.pkl"), recursive=True))
    embs: dict[str, np.ndarray] = {}
    book_names: list[str] = []
    raw: list[np.ndarray] = []
    for path in paths:
        bn = nfc(os.path.splitext(os.path.basename(path))[0])
        d = load_pickle(path)

        if embedding_level == BOOK_LEVEL:
            emb = d.get("embedding")
            if emb is None:
                emb = d.get("book_embedding")
            if emb is None:
                continue
            emb = np.array(emb, dtype=np.float64)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            book_names.append(bn)
            raw.append(emb)
        else:
            chunks = d.get("chunk_embeddings")
            if chunks is None:
                continue
            rows = []
            for chunk_idx, emb in enumerate(chunks):
                cid = chunk_id(bn, chunk_idx)
                if selected_chunk_ids is not None and cid not in selected_chunk_ids:
                    continue
                rows.append(np.array(emb, dtype=np.float64))
            if rows:
                embs[bn] = _normalise_rows(np.stack(rows)).astype(np.float32)

    if embedding_level == CHUNK_LEVEL:
        return embs

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


def score_instances_chunk(
    embs: dict[str, np.ndarray],
    instances: list[tuple[str, str, list[str]]],
) -> float:
    correct = total = 0
    for query, target, distractors in instances:
        if query not in embs or target not in embs:
            continue
        if any(d not in embs for d in distractors):
            continue

        q = embs[query]
        pool = [target] + distractors
        sims = []
        for book in pool:
            # Candidate score: average, over sampled query chunks, of the best
            # matching sampled chunk in the candidate book.
            sims.append(float((q @ embs[book].T).max(axis=1).mean()))
        if int(np.argmax(np.array(sims))) == 0:
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
    embedding_level: str,
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

    suffix = "chunk embeddings" if embedding_level == CHUNK_LEVEL else "book embeddings"
    ax.set_title(f"Retrieval Accuracy — {family} ({suffix})")
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
    parser.add_argument("--embedding_level", choices=[BOOK_LEVEL, CHUNK_LEVEL], default=BOOK_LEVEL,
                        help="Use whole-book embeddings or sampled chunk embeddings")
    parser.add_argument("--sample_n",       type=int, default=5000,
                        help="Chunk mode: number of common chunk embeddings to sample; 0 = all")
    parser.add_argument("--sample_seed",    type=int, default=42,
                        help="Chunk mode: random seed for chunk sampling")
    parser.add_argument("--abtt",           type=int, default=0, metavar="N",
                        help="Book mode only: apply ABTT, removing top-N principal directions (0 = disabled)")
    args = parser.parse_args()

    if args.embedding_level == CHUNK_LEVEL and args.abtt > 0:
        raise ValueError("ABTT is only available for book embeddings; use --abtt 0 with --embedding_level chunk.")

    family  = args.model_family
    if args.embedding_level == CHUNK_LEVEL:
        sample_suffix = "all" if args.sample_n <= 0 else str(args.sample_n)
        suffix = f"_chunks_n{sample_suffix}"
    else:
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

    selected_chunk_ids = None
    if args.embedding_level == CHUNK_LEVEL:
        selected_chunk_ids = sample_common_chunk_ids(
            family, sizes, args.sample_n, args.sample_seed
        )
        available_books = {cid.split("::chunk_", 1)[0] for cid in selected_chunk_ids}
        all_books = [book for book in all_books if book in available_books]
        print(f"[sample] {len(all_books)} books have sampled chunks")

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
        embs = load_embeddings(
            family,
            size,
            abtt_n=args.abtt,
            embedding_level=args.embedding_level,
            selected_chunk_ids=selected_chunk_ids,
        )
        print(f"  {len(embs)} embs")
        row = f"{size:>{col_w}}"
        for g in GROUPINGS:
            if args.embedding_level == CHUNK_LEVEL:
                acc = score_instances_chunk(embs, instances_map[g])
            else:
                acc = score_instances(embs, instances_map[g])
            results[g][size] = acc
            row += f"  {acc:>{col_w}.1%}" if not np.isnan(acc) else f"  {'N/A':>{col_w}}"
        print(row)

    print(sep)

    # ── Save results ─────────────────────────────────────────────────────────
    out_json = os.path.join(out_dir, "results.json")
    serial = {
        "family":         family,
        "embedding_level": args.embedding_level,
        "num_instances":  args.num_instances,
        "num_candidates": args.num_candidates,
        "seed":           args.seed,
        "sample_n":       len(selected_chunk_ids) if selected_chunk_ids is not None else None,
        "sample_seed":    args.sample_seed if args.embedding_level == CHUNK_LEVEL else None,
        "results": {
            g: {s: (v if not np.isnan(v) else None) for s, v in sv.items()}
            for g, sv in results.items()
        },
    }
    with open(out_json, "w") as fh:
        json.dump(serial, fh, indent=2)
    print(f"\n[save]  Results → {out_json}")

    plot_results(results, sizes, out_dir, family, args.num_candidates, args.embedding_level)


if __name__ == "__main__":
    main()
