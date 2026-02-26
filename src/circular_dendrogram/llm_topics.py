"""
llm_topics.py — Add LLM-generated labels to cluster_topics.json

Usage:
    python src/llm_topics.py --model_family Qwen3-Embedding --model_size 0.6B
    python src/llm_topics.py --model_family Qwen3-Embedding --model_size 4B --force
    python src/llm_topics.py --model_family Qwen3-Embedding --model_size 8B --model claude-sonnet-4-5-20250929

Reads:
    out/<tag>/cluster_topics.json

Writes:
    out/<tag>/cluster_topics.json  (adds "llm_label" field per cluster, in-place)
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path so `utils` is importable ──────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import call_claude  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
PROMPT_FILE = SCRIPT_DIR / "llm_prompt.txt"
SLEEP_BETWEEN_CALLS = 1.0  # seconds — be polite to rate limits


def build_prompt(template: str, cluster_id: str, book_count: int,
                 keywords: list[dict], representative_books: list[str]) -> str:
    """Fill the prompt template with cluster data."""
    # Format book titles as a numbered list
    book_lines = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(representative_books))

    # Format keywords as ranked list with scores
    kw_lines = "\n".join(
        f"  {i+1}. {kw['word']} (score: {kw['score']:.4f})"
        for i, kw in enumerate(keywords[:20])  # top 20
    )

    return template.format(
        n_books=book_count,
        book_titles=book_lines,
        keywords=kw_lines,
    )


def parse_label(response: str) -> str | None:
    """
    Extract the label string from Claude's JSON response.

    Returns the label str on success, None on failure.
    """
    text = response.strip()

    # Try strict JSON parse first
    try:
        obj = json.loads(text)
        return str(obj["label"]).strip()
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: regex for "label": "..."
    m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()

    return None


def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    # ── Resolve paths ─────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "out" / args.model_family / args.model_size
    topics_path = out_dir / "cluster_topics.json"

    if not topics_path.exists():
        logger.error(f"cluster_topics.json not found at {topics_path}")
        sys.exit(1)

    # ── Load prompt template ──────────────────────────────────────────────────
    if not PROMPT_FILE.exists():
        logger.error(f"Prompt file not found: {PROMPT_FILE}")
        sys.exit(1)
    template = PROMPT_FILE.read_text(encoding="utf-8")

    # ── Load cluster data ─────────────────────────────────────────────────────
    with open(topics_path, "r", encoding="utf-8") as f:
        cluster_topics: dict = json.load(f)

    total = len(cluster_topics)
    logger.info(f"Loaded {total} clusters from {topics_path}")

    # ── Process each cluster ──────────────────────────────────────────────────
    updated = 0
    skipped = 0
    failed = 0

    for cluster_key, cluster_data in cluster_topics.items():
        already_labeled = "llm_label" in cluster_data and cluster_data["llm_label"]

        if already_labeled and not args.force:
            logger.debug(f"  {cluster_key}: already labeled → skipping")
            skipped += 1
            continue

        book_count = cluster_data.get("book_count", len(cluster_data.get("representative_books", [])))
        keywords = cluster_data.get("top_keywords", [])
        rep_books = cluster_data.get("books", [])

        if not keywords or not rep_books:
            logger.warning(f"  {cluster_key}: missing keywords or books — skipping")
            skipped += 1
            continue

        prompt = build_prompt(template, cluster_key, book_count, keywords, rep_books)

        logger.info(f"  {cluster_key} ({book_count} books) — querying Claude...")

        response = call_claude(
            prompt=prompt,
            model=args.model,
            max_tokens=128,
            temperature=0.2,
            logger=logger,
        )

        if response is None:
            logger.error(f"  {cluster_key}: API call returned None")
            failed += 1
        else:
            label = parse_label(response)
            if label:
                cluster_data["llm_label"] = label
                logger.info(f"  {cluster_key}: \"{label}\"")
                updated += 1
            else:
                # Store raw response as fallback so we don't lose it
                fallback = response.strip()[:200]
                cluster_data["llm_label"] = fallback
                logger.warning(f"  {cluster_key}: JSON parse failed — stored raw: {fallback!r}")
                failed += 1

        # Write incrementally after every cluster so partial results are saved
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump(cluster_topics, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"Done.  updated={updated}  skipped={skipped}  failed={failed}")
    logger.info("═" * 60)
    logger.info("")
    logger.info("Cluster labels:")
    for cluster_key, cluster_data in cluster_topics.items():
        lbl = cluster_data.get("llm_label", "<none>")
        n = cluster_data.get("book_count", "?")
        logger.info(f"  {cluster_key:12s} ({n:4} books)  {lbl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM cluster labels and add them to cluster_topics.json"
    )
    parser.add_argument("--model_family", required=True,
                        help="e.g. Qwen3-Embedding")
    parser.add_argument("--model_size", required=True,
                        help="e.g. 0.6B, 4B, 8B")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-label clusters that already have llm_label")
    parser.add_argument("--verbose", action="store_true",
                        help="Show DEBUG-level log messages")
    args = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    logger = logging.getLogger("llm_topics")

    run(args, logger)


if __name__ == "__main__":
    main()
