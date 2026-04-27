# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

A private RAG (retrieval-augmented generation) interface over 121 Nissan customer
experience research PDFs published by `gabriel-edytor` on the WordPress site
`drivece.kinsta.cloud` (which redirects to `drivetheexperience.com`). Built for
single-user / small-team use; password-gated.

## Architecture

```
WordPress wp-json media API  ──┐
                                ├──► ingest.py (extract → chunk → embed → upsert)
local extra_pdfs/ folder     ──┘
                                            │
                                            ▼
                                  Supabase pgvector (insights table)
                                            ▲
                                            │
        ┌── voyage-3 query embedding ───────┤
        │                                   │
   app.py ◄── Gemini 3 Flash (with fallback to 3.1 Flash Lite) on retrieved chunks
```

- **Embeddings:** Voyage AI `voyage-3` (1024 dims). Same model used at ingest and query time.
- **LLM:** Gemini 3 Flash Preview (primary) → Gemini 3.1 Flash Lite Preview (fallback). Daily-quota auto-fallback.
- **Vector store:** Supabase pgvector (project `mukczuwgyebdimvtcjuq`, eu-west-1).
- **Frontend:** Streamlit chat with citation cards linking back to original PDFs.

## Repo layout

```
app.py                      # Streamlit chat UI + RAG logic
ingest.py                   # PDF discovery, extraction, chunking, embedding, upsert
requirements.txt            # Python deps (pin for Streamlit Cloud builds)
.env.example                # template — real values go in .env (gitignored)
.streamlit/config.toml      # Streamlit config (headless, light theme)

data/                       # NOT committed
  pdf_index.json            # 121 PDF metadata records (id, title, url, date, size)
  pdf_urls.csv              # human-readable index for review
  pdf_urls.txt              # plain URL list
  pdfs/                     # downloaded PDFs (~223 MB, named {media_id}__{filename}.pdf)
  usage.json                # daily Gemini quota counter (auto-managed)

Document Source/            # NOT committed (sample PDF reference)
.claude/                    # NOT committed
```

## How the pieces connect

- **Discovery:** `ingest.py` fetches `/wp-json/wp/v2/media?author=14&mime_type=application/pdf`
  paginated, builds `data/pdf_index.json`. Filtered to author `gabriel-edytor` (id 14).
  121 PDFs total, all publicly fetchable from `drivece.kinsta.cloud/wp-content/uploads/...`.

- **Ingestion (`ingest.py`):**
  - Extracts text per page via `pypdf`. PDFs are decks; per-page text is short and
    self-contained. **No OCR needed.**
  - Pages with <150 chars get merged forward (title slides etc.).
  - Long pages split into 800-token chunks with 100-token overlap (tiktoken `cl100k_base`).
  - Embeds 64 chunks per Voyage API call.
  - Upserts to Supabase keyed on `(source_id, page, chunk_index)` — re-runs are idempotent.
  - Currently 3,298 chunks total in Supabase.

- **Query (`app.py`):**
  - Embed question with Voyage (`input_type="query"`).
  - `match_insights(query_embedding, match_count=12)` RPC in Supabase returns top-12
    by cosine similarity.
  - Build prompt: system instructions + numbered excerpts + history + question.
  - Call primary Gemini model. On 429, automatically retry on fallback model.
  - Render answer with numbered citation cards linking to `{source_url}#page={N}`.

## Supabase

- Project ID: `mukczuwgyebdimvtcjuq`
- Schema: single `insights` table with `embedding vector(1024)` column.
- Indexes: ivfflat cosine similarity, plus btree on `source_id` and `published_at`.
- RPC: `match_insights(query_embedding, match_count, filter_source)`.
- pgvector caps ivfflat/hnsw indexes at 2000 dims — that's why we use voyage-3 (1024)
  rather than gemini-embedding-001 (3072 default).

## Rate-limit context

- **Voyage:** card-on-file unlocks standard limits. 200M free tokens covers many
  full re-ingests. Without card: 3 RPM / 10K TPM (too tight for bulk ingest).
- **Gemini Embeddings (`gemini-embedding-001`):** 1000 requests/day per Google account
  (NOT per project — rotating projects on the same account doesn't help). This is why
  we use Voyage for embeddings.
- **Gemini Chat (`gemini-3-flash-preview`):** 20 RPD primary, falls back to
  `gemini-3.1-flash-lite-preview` at 500 RPD. Daily counter at `data/usage.json`,
  resets at UTC midnight.

## Common tasks

### Re-ingest after new reports are added on the WordPress site

```bash
python ingest.py            # idempotent — skips already-ingested by source_id
python ingest.py --force-source 4860   # re-ingest one PDF (e.g. updated content)
python ingest.py --force-all          # nuclear option — re-embed everything
```

If new author IDs need to be included, edit the discovery query in `ingest.py` (currently
hard-coded to `author=14`).

### Run locally

```bash
pip install -r requirements.txt
# fill out .env from .env.example
streamlit run app.py
```

### Deploy to Streamlit Community Cloud

1. Sign in at share.streamlit.io with the GitHub account that owns the repo.
2. New app → repo `dlau034/nissan-insight-rag`, branch `master`, main file `app.py`.
3. Advanced settings → paste secrets in TOML format (see `.env.example` for the keys).

## Conventions

- **Don't commit `.env`, `data/`, `Document Source/`, or `.claude/`.** All contain
  secrets, large binaries, or personal context.
- **Don't run `--force-all` casually** — costs Voyage tokens and ~7 minutes of runtime.
- **Don't change the embedding model without re-embedding all chunks** — vectors in
  Supabase are model-specific. Mismatched dims will simply error; mismatched models at
  same dim will silently produce garbage retrievals.
- **PDF source URLs use `drivece.kinsta.cloud`** even though the main site redirects
  to `drivetheexperience.com`. The `kinsta.cloud` host is what's stored in WordPress
  and what's directly fetchable without auth.

## Things to watch

- The Gemini chat models are *preview* IDs. Google may rename or retire them —
  if you see a 404 on the primary, check `gemini.models.list()` for the current
  flagship Flash variant and update the constants in `app.py`.
- ivfflat index quality degrades when row counts grow >>10× — at current 3.3K rows
  it's fine, but if the corpus 10×s consider switching to hnsw.
- Streamlit Community Cloud has an ephemeral filesystem — `data/usage.json` will
  reset on app restart. The daily counter is therefore approximate when deployed.
  The actual quota is enforced server-side by Google regardless.
