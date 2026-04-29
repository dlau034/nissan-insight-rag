#!/usr/bin/env python3
"""
app.py — Streamlit chat interface for the Nissan Insight RAG.
Run with: streamlit run app.py
"""

import os, time, json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import voyageai
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from supabase import create_client

load_dotenv()

VOYAGE_KEY   = os.environ["VOYAGE_API_KEY"]
GEMINI_KEY   = os.environ["GEMINI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

MATCH_COUNT  = 30   # retrieve more candidates for reranking
RERANK_TOP_K = 12   # final top-k after Voyage rerank-2

# Primary + fallback Gemini models with their daily request limits (free tier)
PRIMARY_MODEL  = "gemini-3-flash-preview"
PRIMARY_RPD    = 20
FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"
FALLBACK_RPD   = 500

USAGE_FILE = Path("data/usage.json")


# ── Clients ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_clients():
    voyage   = voyageai.Client(api_key=VOYAGE_KEY)
    gemini   = genai.Client(api_key=GEMINI_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return voyage, gemini, supabase

voyage, gemini, supabase = get_clients()


# ── Daily usage tracking ───────────────────────────────────────────────────────

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_usage() -> dict:
    """Return today's usage dict, resetting if the date has rolled over."""
    today = _today_utc()
    if USAGE_FILE.exists():
        try:
            data = json.loads(USAGE_FILE.read_text())
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, PRIMARY_MODEL: 0, FALLBACK_MODEL: 0}


def save_usage(usage: dict) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(usage))


def record_use(model: str) -> None:
    usage = load_usage()
    usage[model] = usage.get(model, 0) + 1
    save_usage(usage)


def remaining(model: str, limit: int) -> int:
    return max(0, limit - load_usage().get(model, 0))


# ── RAG helpers ────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    return voyage.embed([text], model="voyage-3", input_type="query").embeddings[0]


def retrieve(query_vec: list[float], query_text: str) -> list[dict]:
    resp = supabase.rpc(
        "hybrid_match_insights",
        {"query_embedding": query_vec, "query_text": query_text, "match_count": MATCH_COUNT},
    ).execute()
    return resp.data or []


def rerank_chunks(question: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks
    texts = [c["content"] for c in chunks]
    result = voyage.rerank(query=question, documents=texts, model="rerank-2", top_k=RERANK_TOP_K)
    return [chunks[r.index] for r in result.results]


def build_prompt(question: str, chunks: list[dict], history: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i+1}] Report: \"{c['title']}\" | Page {c['page']}\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        )

    return f"""You are an expert analyst on Nissan customer experience research. Your role is to synthesise findings from internal CX research reports and present them clearly to a non-technical business audience.

INSTRUCTIONS:
1. Answer using ONLY the provided report excerpts. Do not invent or infer facts not present in them.
2. Cite every claim with the excerpt number in square brackets, e.g. [1] or [2][3]. Never make an uncited claim.
3. Structure your answer as:
   - **TL;DR** (2–3 sentences maximum): the single most important finding.
   - **Detail**: supporting points, grouped by theme, market, or brand where relevant.
4. When excerpts cover multiple markets or Nissan/Infiniti brands, distinguish them explicitly (e.g. "In the UK… [2]", "For Infiniti… [5]").
5. If excerpts contradict each other, flag the contradiction: "Reports differ: [3] states X while [7] states Y."
6. Preserve quantitative data exactly as stated (percentages, scores, sample sizes). Do not round or paraphrase numbers.
7. If the excerpts do not contain enough information to answer fully, state clearly what is and is not covered, and suggest what kind of report might contain the missing data.
{history_text}

Report excerpts:
{excerpts}

Question: {question}"""


def call_gemini(model: str, prompt: str) -> str:
    """Call Gemini once with retry on transient errors. Raises on quota exhaustion."""
    for attempt in range(4):
        try:
            response = gemini.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2),
            )
            return response.text
        except genai_errors.ClientError as e:
            # 429 = quota exhausted → don't retry, let caller fall back
            if getattr(e, "status_code", None) == 429:
                raise
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
        except genai_errors.ServerError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def ask(question: str, history: list[dict]) -> tuple[str, list[dict], str]:
    """Returns (answer, chunks, model_used)."""
    query_vec = embed_query(question)
    chunks    = retrieve(query_vec, question)
    chunks    = rerank_chunks(question, chunks)
    prompt    = build_prompt(question, chunks, history)

    # Try primary, fall back to lite on quota exhaustion
    try:
        answer = call_gemini(PRIMARY_MODEL, prompt)
        record_use(PRIMARY_MODEL)
        return answer, chunks, PRIMARY_MODEL
    except genai_errors.ClientError as e:
        if getattr(e, "status_code", None) != 429:
            raise
        # Primary quota exhausted — try fallback
        answer = call_gemini(FALLBACK_MODEL, prompt)
        record_use(FALLBACK_MODEL)
        return answer, chunks, FALLBACK_MODEL


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nissan Insight RAG",
    page_icon="🚗",
    layout="wide",
)

# Password gate
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Nissan Insight RAG")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("Nissan Insight RAG")
st.caption("Ask questions across 121 Nissan customer experience research reports.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Reports:** 121 PDFs (June 2023 – Nov 2025)  \n"
        "**Chunks:** 3,298  \n"
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_chunks = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about the Nissan research reports…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching reports…"):
            try:
                answer, chunks, model_used = ask(question, st.session_state.messages[:-1])
                st.session_state.last_chunks = chunks
            except genai_errors.ClientError as e:
                if getattr(e, "status_code", None) == 429:
                    st.error("Both daily quotas are exhausted. Try again after midnight UTC.")
                    st.stop()
                raise

        st.markdown(answer)
        st.caption(f"_Answered with **{model_used}**_")

        if chunks:
            st.divider()
            st.caption("**Sources retrieved**")
            cols = st.columns(3)
            for i, chunk in enumerate(chunks):
                with cols[i % 3]:
                    page_url = f"{chunk['source_url']}#page={chunk['page']}"
                    st.markdown(
                        f"**[{i+1}]** [{chunk['title'][:45]}]({page_url})  \n"
                        f"Page {chunk['page']} · similarity {chunk['similarity']:.2f}"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
