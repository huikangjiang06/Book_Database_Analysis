"""
Extract Gutenberg metadata for every book in processed_books via the Gutendex API.

For each book:
  1. Look up its URL from any manifest (manifest_*.jsonl)
  2. Parse the Gutenberg book ID from the URL
  3. Query https://gutendex.com/books/{id}
  4. Write one JSON line to out/gutenberg_meta/meta.jsonl

Supports resuming: already-fetched books (by book_name) are skipped.
"""

import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT         = Path(__file__).parent
MANIFEST_DIR = ROOT / "outputs_embeddings_all_with_chunks"
OUT_FILE     = ROOT / "out" / "gutenberg_meta" / "meta.jsonl"

GUTENDEX_BASE  = "https://gutendex.com/books/{id}"
SLEEP          = 0.5    # seconds between successful API calls
RETRY_WAITS    = [10, 30, 60, 120, 300]  # back-off ladder for 429 (seconds)


# ── helpers ─────────────────────────────────────────────────────────────────

def load_manifest_urls() -> dict[tuple[str, str], tuple[str, str]]:
    """Return {(nfc_genre, nfc_book_name): (url, book_title)} from all manifests."""
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for mf in sorted(MANIFEST_DIR.glob("manifest_*.jsonl")):
        with mf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    unicodedata.normalize("NFC", rec["genre"]),
                    unicodedata.normalize("NFC", rec["book_name"]),
                )
                if key not in seen:
                    seen[key] = (rec["url"], rec.get("book_title", ""))
    return seen


def extract_gutenberg_id(url: str) -> int | None:
    """Parse numeric Gutenberg ID from any known URL pattern."""
    # https://www.gutenberg.org/ebooks/4928.txt.utf-8
    m = re.search(r"/ebooks/(\d+)", url)
    if m:
        return int(m.group(1))
    # https://www.gutenberg.org/cache/epub/754/pg754.txt
    m = re.search(r"/epub/(\d+)/", url)
    if m:
        return int(m.group(1))
    # https://www.gutenberg.org/files/1234/1234.txt
    m = re.search(r"/files/(\d+)/", url)
    if m:
        return int(m.group(1))
    return None


def fetch_gutendex(gid: int) -> dict:
    """Fetch one book from Gutendex, retrying on HTTP 429 with progressive waits."""
    url = GUTENDEX_BASE.format(id=gid)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt, wait in enumerate(RETRY_WAITS + [None], start=1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait is None:
                raise
            # honour Retry-After header if present, else use our ladder
            retry_after = e.headers.get("Retry-After")
            pause = int(retry_after) if retry_after and retry_after.isdigit() else wait
            print(f"    [429] rate-limited — waiting {pause}s (attempt {attempt}/{len(RETRY_WAITS)})")
            time.sleep(pause)
    raise RuntimeError(f"All retries exhausted for id={gid}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    manifest_urls = load_manifest_urls()
    print(f"[info] Loaded URLs for {len(manifest_urls)} unique books from manifests")

    proc_dir = MANIFEST_DIR / "processed_books"
    books: list[tuple[str, str]] = []
    for txt in sorted(proc_dir.glob("*/*.txt")):
        genre     = unicodedata.normalize("NFC", txt.parent.name)
        book_name = unicodedata.normalize("NFC", txt.stem)
        books.append((genre, book_name))
    print(f"[info] Found {len(books)} books in processed_books/")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip books already written
    already_done: set[str] = set()
    if OUT_FILE.exists():
        with OUT_FILE.open() as fh:
            for line in fh:
                try:
                    already_done.add(json.loads(line.strip())["book_name"])
                except Exception:
                    pass
        if already_done:
            print(f"[info] Resuming — {len(already_done)} books already done")

    remaining = [(g, n) for g, n in books if n not in already_done]
    print(f"[info] {len(remaining)} books to fetch\n")

    with OUT_FILE.open("a") as out_fh:
        for i, (genre, book_name) in enumerate(remaining, 1):
            key = (genre, book_name)
            if key not in manifest_urls:
                print(f"  [{i:>3}/{len(remaining)}] SKIP  {book_name}  — no manifest entry")
                continue

            url, book_title = manifest_urls[key]
            gid = extract_gutenberg_id(url)
            if gid is None:
                print(f"  [{i:>3}/{len(remaining)}] SKIP  {book_name}  — can't parse ID from {url}")
                continue

            try:
                meta = fetch_gutendex(gid)
                record = {
                    "book_name":     book_name,
                    "book_title":    book_title,
                    "genre":         genre,
                    "gutenberg_id":  gid,
                    "gutenberg_url": url,
                    **meta,
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                authors = ", ".join(a["name"] for a in meta.get("authors", []))
                print(f"  [{i:>3}/{len(remaining)}] OK    {book_name}  (id={gid}, {authors})")
                time.sleep(SLEEP)
            except urllib.error.HTTPError as e:
                print(f"  [{i:>3}/{len(remaining)}] HTTP{e.code} {book_name} — skipping")
            except Exception as e:
                print(f"  [{i:>3}/{len(remaining)}] ERR   {book_name}: {e}")

    total = sum(1 for _ in OUT_FILE.open())
    print(f"\n[done] {total} records → {OUT_FILE}")


if __name__ == "__main__":
    main()
