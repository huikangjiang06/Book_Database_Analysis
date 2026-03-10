"""
Count books per genre from processed_books/<genre>/*.txt directory structure.

Output:
  out/gutenberg_meta/genre_counts.json  — {genre: count} sorted descending
"""

import json
from pathlib import Path

PROC_DIR = Path(__file__).parents[2] / "outputs_embeddings_all_with_chunks" / "processed_books"
OUT_FILE = Path(__file__).parents[2] / "out" / "gutenberg_meta" / "genre_counts.json"


def main():
    counts = {}
    for genre_dir in sorted(PROC_DIR.iterdir()):
        if genre_dir.is_dir():
            counts[genre_dir.name] = sum(1 for f in genre_dir.glob("*.txt"))

    counts_sorted = dict(sorted(counts.items(), key=lambda x: -x[1]))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(counts_sorted, indent=2, ensure_ascii=False) + "\n")

    print(f"[done] {len(counts_sorted)} genres → {OUT_FILE}\n")
    for genre, count in counts_sorted.items():
        print(f"  {count:>4}  {genre}")


if __name__ == "__main__":
    main()
