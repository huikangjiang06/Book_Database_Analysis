"""
Aggregate subject and bookshelf counts across all books in meta.jsonl.

Outputs:
  out/gutenberg_meta/subject_counts.json    — {subject: count} sorted descending
  out/gutenberg_meta/bookshelf_counts.json  — {bookshelf: count} sorted descending
"""

import json
from collections import Counter
from pathlib import Path

META_FILE = Path(__file__).parents[2] / "out" / "gutenberg_meta" / "meta.jsonl"
OUT_DIR   = META_FILE.parent


def main():
    subjects   = Counter()
    bookshelves = Counter()

    with META_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for s in rec.get("subjects", []):
                subjects[s] += 1
            for b in rec.get("bookshelves", []):
                bookshelves[b] += 1

    subject_out    = OUT_DIR / "subject_counts.json"
    bookshelf_out  = OUT_DIR / "bookshelf_counts.json"

    subject_sorted   = dict(sorted(subjects.items(),    key=lambda x: -x[1]))
    bookshelf_sorted = dict(sorted(bookshelves.items(), key=lambda x: -x[1]))

    subject_out.write_text(
        json.dumps(subject_sorted,   indent=2, ensure_ascii=False) + "\n"
    )
    bookshelf_out.write_text(
        json.dumps(bookshelf_sorted, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"[done] {len(subjects)} unique subjects    → {subject_out}")
    print(f"[done] {len(bookshelves)} unique bookshelves → {bookshelf_out}")

    print("\nTop 10 subjects:")
    for s, c in list(subject_sorted.items())[:10]:
        print(f"  {c:>4}  {s}")

    print("\nTop 10 bookshelves:")
    for b, c in list(bookshelf_sorted.items())[:10]:
        print(f"  {c:>4}  {b}")


if __name__ == "__main__":
    main()
