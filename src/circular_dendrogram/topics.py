"""
Topic Extraction Step
=====================
Reads the cluster results produced by pipeline.py and extracts:

  1. Book-level TF-IDF keywords
     Stored at: out/Book_Topics/<pkl_stem>.json
     Shared across all model runs — only computed once (skipped if exists).

  2. Cluster-level c-TF-IDF keywords (BERTopic-style)
     Stored at: out/<model_family>_<model_size>/cluster_topics.json
     Re-computed per model run (cluster assignments differ per model).

Usage:
    python topics.py --model_family Qwen3-Embedding --model_size 0.6B
    python topics.py --model_family Pythia --model_size 70M --top_n 30
"""

import argparse
import json
import os
import pickle
import re
import string
from collections import Counter, defaultdict
from math import log

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_BOOKS = os.path.join(
    ROOT, "outputs_embeddings_all_with_chunks", "processed_books"
)
OUT_DIR = os.path.join(ROOT, "out")
BOOK_TOPICS_DIR = os.path.join(OUT_DIR, "Book_Topics")
os.makedirs(BOOK_TOPICS_DIR, exist_ok=True)

# Stop words — simple English set; avoids heavy spaCy dependency
_STOP_WORDS = set(
    "a about above after again against all am an and any are aren't as at be "
    "because been before being below between both but by can't cannot could "
    "couldn't did didn't do does doesn't doing don't down during each few for "
    "from further get got had hadn't has hasn't have haven't having he he'd "
    "he'll he's her here here's hers herself him himself his how how's i i'd "
    "i'll i'm i've if in into is isn't it it's its itself let's me more most "
    "mustn't my myself no nor not of off on once only or other ought our ours "
    "ourselves out over own same shan't she she'd she'll she's should shouldn't "
    "so some such than that that's the their theirs them themselves then there "
    "there's these they they'd they'll they're they've this those through to too "
    "under until up very was wasn't we we'd we'll we're we've were weren't what "
    "what's when when's where where's which while who who's whom why why's will "
    "with won't would wouldn't you you'd you'll you're you've your yours yourself "
    "yourselves said one two would could upon whom us thus Mr Mrs thy thee thou "
    "hath doth shall yet had did also may still even nothing ever never always "
    "every much little well good great men man old new come came went go back "
    "first last long made make now see know think way take hand say".split()
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tag(model_family: str, model_size: str) -> str:
    return f"{model_family}_{model_size}".replace("/", "_").replace(" ", "_")


def _run_dir(model_family: str, model_size: str) -> str:
    return os.path.join(OUT_DIR, model_family, model_size)


def _load_text(genre: str, pkl_stem: str) -> str:
    """Load the processed book text given its genre and pkl stem (filename without ext)."""
    path = os.path.join(PROCESSED_BOOKS, genre, pkl_stem + ".txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, alpha-only tokens, remove stop words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


# ─── Step A: Book-level TF-IDF ────────────────────────────────────────────────

def compute_book_tfidf(records: list[dict], top_n: int = 20) -> dict[str, list]:
    """
    Compute TF-IDF for every book relative to the full corpus.
    Records is a list of dicts with keys: book_title, genre, pkl_stem, text.

    Returns dict: pkl_stem → list of {word, score} dicts (top_n words).
    """
    texts = [r["text"] for r in records]
    stems = [r["pkl_stem"] for r in records]

    # sklearn TF-IDF — sublinear_tf prevents very long books from dominating
    print(f"[tfidf] Fitting TF-IDF on {len(texts)} book texts ...")
    vec = TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,          # ignore words appearing in only 1 book
        max_df=0.85,       # ignore words appearing in >85% of books
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        stop_words=list(_STOP_WORDS),
    )
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()

    result = {}
    for i, stem in enumerate(stems):
        row = X[i]
        top_indices = row.toarray()[0].argsort()[::-1][:top_n]
        keywords = [
            {"word": feature_names[j], "score": round(float(row[0, j]), 6)}
            for j in top_indices
            if row[0, j] > 0
        ]
        result[stem] = keywords

    print(f"[tfidf] Done.")
    return result


def save_book_topics(
    records: list[dict],
    book_tfidf: dict[str, list],
    force: bool = False,
) -> tuple[int, int]:
    """
    Save per-book topic JSON files to BOOK_TOPICS_DIR.
    Skips files that already exist (unless force=True).
    Returns (n_written, n_skipped).
    """
    written = skipped = 0
    for r in records:
        out_path = os.path.join(BOOK_TOPICS_DIR, r["pkl_stem"] + ".json")
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue
        payload = {
            "book_title": r["book_title"],
            "genre": r["genre"],
            "pkl_stem": r["pkl_stem"],
            "top_keywords": book_tfidf.get(r["pkl_stem"], []),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        written += 1

    print(f"[book_topics] Wrote {written} new files, skipped {skipped} existing "
          f"(use --force to overwrite).")
    return written, skipped


# ─── Step B: Cluster-level c-TF-IDF ──────────────────────────────────────────

def compute_ctfidf(
    cluster_texts: dict[int, str],
    top_n: int = 20,
) -> dict[int, list]:
    """
    BERTopic-style c-TF-IDF:
      score(t, c) = tf(t, c) * log(1 + A / f_t)
    where:
      tf(t, c)  = count of t in class c / total words in class c
      A         = average number of words per class
      f_t       = total count of t across all classes

    cluster_texts: {cluster_id: aggregated_text_string}
    Returns: {cluster_id: [{word, score}, ...]}
    """
    print(f"[ctfidf] Computing c-TF-IDF for {len(cluster_texts)} clusters ...")

    # Build per-cluster word counts
    cluster_counts: dict[int, Counter] = {}
    cluster_totals: dict[int, int] = {}
    for cid, text in cluster_texts.items():
        tokens = _tokenize(text)
        cluster_counts[cid] = Counter(tokens)
        cluster_totals[cid] = len(tokens)

    # Global word frequency across all clusters
    global_counts: Counter = Counter()
    for cnt in cluster_counts.values():
        global_counts.update(cnt)

    # Average words per cluster
    A = np.mean(list(cluster_totals.values()))

    result = {}
    for cid, cnt in cluster_counts.items():
        total = cluster_totals[cid]
        if total == 0:
            result[cid] = []
            continue

        scores = {}
        for word, count in cnt.items():
            tf = count / total
            idf = log(1 + A / (global_counts[word] + 1e-9))
            scores[word] = tf * idf

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result[cid] = [{"word": w, "score": round(s, 6)} for w, s in top]

    print(f"[ctfidf] Done.")
    return result


def save_cluster_topics(
    cluster_ctfidf: dict[int, list],
    cluster_meta: dict[int, dict],
    run_dir: str,
) -> str:
    """
    Save cluster topics JSON to <run_dir>/cluster_topics.json.
    cluster_meta: {cluster_id: {book_count, books: [title, ...]}}
    """
    out_path = os.path.join(run_dir, "cluster_topics.json")
    payload = {}
    for cid in sorted(cluster_ctfidf.keys()):
        label = f"cluster_{cid}" if cid >= 0 else "noise"
        payload[label] = {
            "cluster_id": cid,
            "book_count": cluster_meta[cid]["book_count"],
            "books": cluster_meta[cid]["books"],
            "top_keywords": cluster_ctfidf[cid],
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[cluster_topics] Saved to {out_path}")
    return out_path


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_topics(model_family: str, model_size: str, top_n: int, force: bool):
    tag = _tag(model_family, model_size)
    run_dir = _run_dir(model_family, model_size)
    clusters_pkl = os.path.join(run_dir, "clusters.pkl")

    if not os.path.exists(clusters_pkl):
        raise FileNotFoundError(
            f"clusters.pkl not found at {clusters_pkl}. "
            f"Run pipeline.py first."
        )

    # Load cluster results
    print(f"[load] Reading {clusters_pkl} ...")
    with open(clusters_pkl, "rb") as f:
        data = pickle.load(f)

    df = data["result_df"]
    labels = data["labels"]

    # Build record list with text
    print(f"[load] Loading book texts from processed_books/ ...")
    records = []
    missing = 0
    for _, row in df.iterrows():
        pkl_stem = os.path.splitext(os.path.basename(row["pkl_path"]))[0]
        text = _load_text(row["genre"], pkl_stem)
        if not text:
            missing += 1
        records.append(
            {
                "book_title": row["book_title"],
                "genre": row["genre"],
                "pkl_stem": pkl_stem,
                "cluster": row["cluster"],
                "text": text,
            }
        )

    print(f"[load] {len(records)} books loaded, {missing} texts missing.")

    # ── Book-level TF-IDF ─────────────────────────────────────────────────────
    # Determine which books still need computing (not yet in Book_Topics/)
    needs_tfidf = [
        r for r in records
        if not os.path.exists(os.path.join(BOOK_TOPICS_DIR, r["pkl_stem"] + ".json"))
        or force
    ]
    already_done = len(records) - len(needs_tfidf)

    if already_done > 0:
        print(f"[book_topics] {already_done}/{len(records)} books already have "
              f"topics in {BOOK_TOPICS_DIR} — skipping those.")

    if needs_tfidf:
        # We still need to fit TF-IDF on the full corpus for correct IDF,
        # even if we only save the new ones.
        book_tfidf = compute_book_tfidf(records, top_n=top_n)
        save_book_topics(records, book_tfidf, force=force)
    else:
        print(f"[book_topics] All {len(records)} books already processed. "
              f"Use --force to recompute.")

    # ── Cluster-level c-TF-IDF ────────────────────────────────────────────────
    # Build per-cluster aggregated text and metadata
    cluster_texts: dict[int, str] = defaultdict(str)
    cluster_meta: dict[int, dict] = defaultdict(lambda: {"book_count": 0, "books": []})

    for r in records:
        cid = r["cluster"]
        cluster_texts[cid] += " " + r["text"]
        cluster_meta[cid]["book_count"] += 1
        cluster_meta[cid]["books"].append(r["book_title"])

    # Exclude noise cluster (-1) from c-TF-IDF computation
    non_noise = {k: v for k, v in cluster_texts.items() if k >= 0}
    cluster_ctfidf = compute_ctfidf(non_noise, top_n=top_n)

    # Include noise in output (empty keywords)
    if -1 in cluster_texts:
        cluster_ctfidf[-1] = []

    save_cluster_topics(cluster_ctfidf, cluster_meta, run_dir)

    # ── Quick Summary ─────────────────────────────────────────────────────────
    print(f"\n[summary] Top keywords per cluster:")
    for cid in sorted(cluster_ctfidf.keys()):
        if cid < 0:
            continue
        kws = [x["word"] for x in cluster_ctfidf[cid][:8]]
        books_preview = cluster_meta[cid]["books"][:2]
        print(f"  Cluster {cid:3d} ({cluster_meta[cid]['book_count']:3d} books) "
              f"| {', '.join(kws)}")
        print(f"            → e.g. {books_preview[0][:60]}")

    print(f"\n[done] Topic extraction complete.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emergence Topic Extraction")
    parser.add_argument("--model_family", type=str, default="Qwen3-Embedding")
    parser.add_argument("--model_size", type=str, default="0.6B")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top keywords to extract per book/cluster")
    parser.add_argument("--force", action="store_true",
                        help="Recompute book topics even if already cached")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Topic Extraction: {args.model_family} / {args.model_size}")
    print(f"{'='*60}\n")

    run_topics(args.model_family, args.model_size, args.top_n, args.force)


if __name__ == "__main__":
    main()
