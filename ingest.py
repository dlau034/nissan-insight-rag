#!/usr/bin/env python3
"""
ingest.py — Extract, chunk, embed, and upsert Nissan insight PDFs into Supabase pgvector.

Usage:
  python ingest.py                         # ingest all new PDFs (skips already ingested)
  python ingest.py --force-source 4860     # re-ingest a specific media ID
  python ingest.py --force-all             # re-ingest everything
"""

import os, sys, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import tiktoken
import voyageai
from supabase import create_client
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

VOYAGE_KEY   = os.environ["VOYAGE_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
PDF_DIR      = Path("data/pdfs")
INDEX_FILE   = Path("data/pdf_index.json")

CHUNK_TOKENS   = 800
OVERLAP_TOKENS = 100
MIN_PAGE_CHARS = 150
EMBED_BATCH    = 64    # Voyage supports up to 128; 64 is safe
BATCH_PAUSE    = 0.5   # brief pause between batches

voyage   = voyageai.Client(api_key=VOYAGE_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
enc      = tiktoken.get_encoding("cl100k_base")


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages with extractable text."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """
    Merge short pages into adjacent ones, then split long pages into
    token-aware chunks with overlap.
    Returns [(page_num, chunk_index, text), ...]
    """
    merged: list[tuple[int, str]] = []
    buf_page, buf_text = None, ""
    for page_num, text in pages:
        if buf_page is None:
            buf_page, buf_text = page_num, text
        elif len(buf_text) < MIN_PAGE_CHARS:
            buf_text += " " + text
        else:
            merged.append((buf_page, buf_text))
            buf_page, buf_text = page_num, text
    if buf_page is not None:
        merged.append((buf_page, buf_text))

    chunks: list[tuple[int, int, str]] = []
    for page_num, text in merged:
        tokens = enc.encode(text)
        if len(tokens) <= CHUNK_TOKENS:
            chunks.append((page_num, 0, text))
        else:
            idx, start = 0, 0
            while start < len(tokens):
                end = min(start + CHUNK_TOKENS, len(tokens))
                chunks.append((page_num, idx, enc.decode(tokens[start:end])))
                idx += 1
                start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


# ── Embeddings (batched) ───────────────────────────────────────────────────────

@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6))
def embed_batch(texts: list[str], input_type: str = "document") -> list[list[float]]:
    result = voyage.embed(texts, model="voyage-3", input_type=input_type)
    return result.embeddings


def embed_all(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed any number of texts, batching and pausing to respect free-tier limits."""
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(embed_batch(texts[i : i + EMBED_BATCH], input_type))
        time.sleep(BATCH_PAUSE)
    return embeddings


# ── Supabase helpers ───────────────────────────────────────────────────────────

def is_ingested(source_id: str) -> bool:
    resp = (
        supabase.table("insights")
        .select("id", count="exact")
        .eq("source_id", source_id)
        .limit(1)
        .execute()
    )
    return (resp.count or 0) > 0


def delete_source(source_id: str) -> None:
    supabase.table("insights").delete().eq("source_id", source_id).execute()


def upsert_chunks(records: list[dict]) -> None:
    for i in range(0, len(records), 50):
        supabase.table("insights").upsert(records[i : i + 50]).execute()


# ── Per-PDF ingestion ──────────────────────────────────────────────────────────

def ingest_one(row: dict, force: bool = False) -> int:
    source_id    = str(row["id"])
    title        = row["title"]
    source_url   = row["source_url"]
    date_str     = row.get("date", "")
    published_at = date_str + "T00:00:00Z" if date_str and "T" not in date_str else date_str

    pdf_files = list(PDF_DIR.glob(f"{source_id}__*.pdf"))
    if not pdf_files:
        print("SKIP (no local file)")
        return 0

    if not force and is_ingested(source_id):
        print("SKIP (already ingested)")
        return 0

    if force:
        delete_source(source_id)

    pages = extract_pages(pdf_files[0])
    if not pages:
        print("SKIP (no extractable text)")
        return 0

    chunks = chunk_pages(pages)
    texts  = [text for _, _, text in chunks]
    vecs   = embed_all(texts)

    records = []
    for (page_num, chunk_idx, text), vec in zip(chunks, vecs):
        records.append({
            "source_id":    source_id,
            "source_url":   source_url,
            "title":        title,
            "published_at": published_at or None,
            "page":         page_num,
            "chunk_index":  chunk_idx,
            "content":      text,
            "embedding":    vec,
        })

    upsert_chunks(records)
    return len(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-source", metavar="ID", help="Re-ingest one PDF by media ID")
    parser.add_argument("--force-all", action="store_true", help="Re-ingest all PDFs")
    args = parser.parse_args()

    with open(INDEX_FILE, encoding="utf-8") as f:
        index = json.load(f)

    if args.force_source:
        index = [r for r in index if str(r["id"]) == args.force_source]
        if not index:
            sys.exit(f"No entry found for source ID {args.force_source}")

    total_chunks, t0 = 0, time.time()
    for i, row in enumerate(index, 1):
        label = row["title"][:55]
        print(f"[{i:>3}/{len(index)}] {label:<55}", end=" ", flush=True)
        n = ingest_one(row, force=args.force_all or bool(args.force_source))
        total_chunks += n
        if n:
            print(f"{n} chunks")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s — {total_chunks} chunks upserted to Supabase")


if __name__ == "__main__":
    main()
