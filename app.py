#!/usr/bin/env python3
"""
app.py — Streamlit chat interface for the Nissan Insight RAG.
Run with: streamlit run app.py
"""

import os, time
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

MATCH_COUNT  = 12   # chunks retrieved per query
GEMINI_MODEL = "gemini-2.5-flash"

# ── Clients (cached so they're reused across reruns) ───────────────────────────

@st.cache_resource
def get_clients():
    voyage   = voyageai.Client(api_key=VOYAGE_KEY)
    gemini   = genai.Client(api_key=GEMINI_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return voyage, gemini, supabase

voyage, gemini, supabase = get_clients()


# ── RAG helpers ────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    result = voyage.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]


def retrieve(query_vec: list[float]) -> list[dict]:
    resp = supabase.rpc(
        "match_insights",
        {"query_embedding": query_vec, "match_count": MATCH_COUNT},
    ).execute()
    return resp.data or []


def build_prompt(question: str, chunks: list[dict], history: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i+1}] Report: \"{c['title']}\" | Page {c['page']}\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]  # last 3 turns
        )

    return f"""You are an expert analyst on Nissan customer experience research.
Answer the question below using ONLY the provided report excerpts.
Cite every claim with the excerpt number in square brackets e.g. [1] or [2][3].
If the excerpts do not contain enough information to answer, say so clearly.
Do not invent facts not present in the excerpts.
{history_text}

Report excerpts:
{excerpts}

Question: {question}"""


def ask(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    query_vec = embed_query(question)
    chunks    = retrieve(query_vec)
    prompt    = build_prompt(question, chunks, history)

    for attempt in range(4):
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2),
            )
            break
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            if attempt == 3:
                raise
            wait = 5 * (attempt + 1)
            time.sleep(wait)
    answer = response.text
    return answer, chunks


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nissan Insight RAG",
    page_icon="🚗",
    layout="wide",
)

# ── Password gate ──────────────────────────────────────────────────────────────
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
        "**Embeddings:** Voyage AI voyage-3  \n"
        "**LLM:** Gemini 2.5 Flash  \n"
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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching reports…"):
            answer, chunks = ask(question, st.session_state.messages[:-1])
            st.session_state.last_chunks = chunks

        st.markdown(answer)

        # Citation cards
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
