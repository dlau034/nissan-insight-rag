#!/usr/bin/env python3
"""
app.py — Streamlit chat interface for the Nissan Insight RAG.
Run with: streamlit run app.py
"""

import os, time, json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import streamlit as st
import voyageai
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from supabase import create_client

try:
    from tavily import TavilyClient as _TavilyClient
except ImportError:
    _TavilyClient = None

load_dotenv()

def _setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return os.environ.get(name, value)


VOYAGE_KEY   = os.environ["VOYAGE_API_KEY"]
GEMINI_KEY   = os.environ["GEMINI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
TAVILY_KEY   = _setting("TAVILY_API_KEY")

MATCH_COUNT          = 30   # hybrid retrieve candidates
RERANK_TOP_K         = 12   # internal top-k after Voyage rerank-2
WEB_K                = 8    # Tavily results per call
WEB_RERANK_TOP_K     = 8    # web pool rerank top-k
RESEARCH_POLL_SECONDS = 2
RESEARCH_TIMEOUT_SECONDS = 120
RESEARCH_MODEL = "mini"

# Primary + fallback Gemini models
PRIMARY_MODEL  = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"

USAGE_FILE = Path("data/usage.json")
REPORTS_MODE = "CE Reports"
WEB_MODE = "Web Search"
DEEP_RESEARCH_MODE = "Deep Research"
APP_VERSION = "v1.1.0"


# ── Clients ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_clients():
    v = voyageai.Client(api_key=VOYAGE_KEY)
    g = genai.Client(api_key=GEMINI_KEY)
    s = create_client(SUPABASE_URL, SUPABASE_KEY)
    t = _TavilyClient(api_key=TAVILY_KEY) if (_TavilyClient and TAVILY_KEY) else None
    return v, g, s, t

voyage, gemini, supabase, tavily = get_clients()
TAVILY_ENABLED = tavily is not None


# ── Daily usage tracking ─────────────────────────────────────────────────────────

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_usage() -> dict:
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


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url[:40]


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        tag = " (web search)" if m.get("kind") == "user_web" else \
              " (web answer)" if m.get("kind") == "assistant_web" else \
              " (deep research)" if m.get("kind") == "user_research" else \
              " (deep research answer)" if m.get("kind") == "assistant_research" else ""
        lines.append(f"{role}{tag}: {m['content']}")
    return "\n\nConversation so far:\n" + "\n".join(lines)


def _clean_followup_text(text: str) -> str:
    """Remove assistant-style labels before pre-filling the chat box."""
    cleaned = text.strip().strip('"\'').lstrip("- ").strip()
    prefix = "Follow-up question:"
    if cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):].strip()
    return cleaned


def _prefill_chat(text: str, mode: str) -> None:
    """Queue text for the composer and force Streamlit to rebuild its widgets."""
    st.session_state.pending_input = _clean_followup_text(text)
    st.session_state.mode = mode
    st.session_state.form_counter = st.session_state.get("form_counter", 0) + 1


# ── RAG core ─────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    return voyage.embed([text], model="voyage-3", input_type="query").embeddings[0]


def retrieve(query_vec: list[float], query_text: str) -> list[dict]:
    resp = supabase.rpc(
        "hybrid_match_insights",
        {"query_embedding": query_vec, "query_text": query_text, "match_count": MATCH_COUNT},
    ).execute()
    return resp.data or []


def rerank_chunks(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return chunks
    texts = [c["content"] for c in chunks]
    result = voyage.rerank(query=question, documents=texts, model="rerank-2", top_k=top_k)
    return [chunks[r.index] for r in result.results]


def call_gemini(model: str, prompt: str) -> str:
    """Call Gemini with retry on transient errors. Raises on 429."""
    for attempt in range(4):
        try:
            response = gemini.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2),
            )
            return response.text
        except genai_errors.ClientError as e:
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


def _call_with_fallback(prompt: str) -> tuple[str, str]:
    """Try primary model, fall back to lite on 429. Returns (answer, model_used)."""
    try:
        answer = call_gemini(PRIMARY_MODEL, prompt)
        record_use(PRIMARY_MODEL)
        return answer, PRIMARY_MODEL
    except genai_errors.ClientError as e:
        if getattr(e, "status_code", None) != 429:
            raise
        answer = call_gemini(FALLBACK_MODEL, prompt)
        record_use(FALLBACK_MODEL)
        return answer, FALLBACK_MODEL


def build_internal_prompt(question: str, chunks: list[dict], history: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i+1}] Report: \"{c['title']}\" | Page {c['page']}\n{c['content']}"
        for i, c in enumerate(chunks)
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
{_format_history(history)}

Report excerpts:
{excerpts}

Question: {question}"""


def build_web_prompt(question: str, chunks: list[dict], history: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[w{i+1}] {_domain(c['source_url'])} | \"{c['title']}\"\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    return f"""You are answering a question using only public web sources retrieved for this query.

INSTRUCTIONS:
1. Answer using ONLY the provided web excerpts. Do not invent or infer facts not present in them.
2. Cite every claim with the source number in square brackets, e.g. [w1] or [w2][w3]. Never make an uncited claim.
3. Structure your answer as:
   - **TL;DR** (2–3 sentences maximum): the single most important finding.
   - **Detail**: supporting points grouped by theme or source.
4. Preserve quantitative data exactly as stated. Do not round or paraphrase numbers.
5. If sources contradict each other, flag it: "Sources differ: [w1] states X while [w3] states Y."
6. If the web excerpts do not contain enough information to answer, say so clearly.
{_format_history(history)}

Web excerpts:
{excerpts}

Question: {question}"""


# ── Ask functions ────────────────────────────────────────────────────────────────

def ask_internal(question: str, history: list[dict]) -> tuple[str, list[dict], str, str | None]:
    """Returns (answer, chunks, model_used, suggested_followup)."""
    query_vec = embed_query(question)
    chunks    = retrieve(query_vec, question)
    chunks    = rerank_chunks(question, chunks, top_k=RERANK_TOP_K)
    prompt    = build_internal_prompt(question, chunks, history)
    answer, model_used = _call_with_fallback(prompt)
    suggestion = suggest_followup(question, answer, history) if TAVILY_ENABLED else None
    return answer, chunks, model_used, suggestion


def suggest_followup(question: str, internal_answer: str, history: list[dict]) -> str | None:
    """Generate a web follow-up suggestion using Gemini Flash Lite. Returns None if not useful."""
    prompt = f"""You are helping a Nissan CX researcher decide whether the open web could extend an internal-research answer.

Original question: {question}
Internal answer (summary): {internal_answer[:600]}
{_format_history(history[-4:])}

Write ONE follow-up question (max 20 words) the user could ask the open web to fill a gap or add external industry context, while staying anchored on their original intent. Phrase it as a natural question.
If the internal answer is already comprehensive and no useful web extension exists, return exactly: NONE

Follow-up question:"""
    try:
        text = _clean_followup_text(call_gemini(FALLBACK_MODEL, prompt))
        record_use(FALLBACK_MODEL)
        return None if text.upper().startswith("NONE") else text or None
    except Exception:
        return None


def ask_web(query: str, history: list[dict]) -> tuple[str, list[dict], str]:
    """Search Tavily, rerank, answer via Gemini. Returns (answer, web_chunks, model_used)."""
    if not TAVILY_ENABLED:
        return "Web search is not configured. Add TAVILY_API_KEY to .env to enable it.", [], FALLBACK_MODEL

    try:
        resp = tavily.search(query, search_depth="basic", max_results=WEB_K, include_answer=False)
        results = resp.get("results", [])
    except Exception as e:
        return f"Web search failed: {e}", [], FALLBACK_MODEL

    if not results:
        return "The web search returned no results for that query.", [], FALLBACK_MODEL

    web_chunks = [
        {
            "source_id":   f"web:{r['url']}",
            "source_url":  r["url"],
            "title":       r.get("title", _domain(r["url"])),
            "page":        None,
            "chunk_index": 0,
            "content":     r.get("content", ""),
            "similarity":  None,
            "is_external": True,
        }
        for r in results
    ]
    web_chunks = rerank_chunks(query, web_chunks, top_k=WEB_RERANK_TOP_K)
    prompt = build_web_prompt(query, web_chunks, history)
    answer, model_used = _call_with_fallback(prompt)
    return answer, web_chunks, model_used


def _research_status(result: dict) -> str:
    return str(result.get("status") or result.get("state") or "").lower()


def _research_content(result: dict) -> str:
    return result.get("content") or result.get("report") or result.get("answer") or ""


def _research_sources(result: dict) -> list:
    return result.get("sources") or result.get("results") or []


def _normalize_research_sources(sources: list) -> list[dict]:
    normalized = []
    for i, source in enumerate(sources):
        if isinstance(source, str):
            url = source
            title = _domain(source) or f"Source {i + 1}"
            content = ""
        elif isinstance(source, dict):
            url = source.get("url") or source.get("source_url") or source.get("link") or ""
            title = source.get("title") or source.get("name") or _domain(url) or f"Source {i + 1}"
            content = source.get("content") or source.get("snippet") or source.get("summary") or ""
        else:
            continue
        normalized.append(
            {
                "source_id": f"research:{url or i}",
                "source_url": url,
                "title": title,
                "page": None,
                "chunk_index": 0,
                "content": content,
                "similarity": None,
                "is_external": True,
                "is_research": True,
            }
        )
    return normalized


def ask_deep_research(query: str, status_box=None) -> tuple[str, list[dict], str, str | None]:
    """Run Tavily Deep Research and return (report, sources, model_label, request_id)."""
    model_label = "tavily-research-mini"
    if not TAVILY_ENABLED:
        return "Deep Research is not configured. Add TAVILY_API_KEY to enable it.", [], model_label, None

    def _status(message: str) -> None:
        if status_box is not None:
            status_box.info(message)

    try:
        _status("Creating research task…")
        task = tavily.research(
            input=query,
            model=RESEARCH_MODEL,
            citation_format="numbered",
        )
        request_id = task.get("request_id") or task.get("id")

        if not request_id:
            content = _research_content(task)
            if content:
                _status("Finalizing report…")
                return content, _normalize_research_sources(_research_sources(task)), model_label, None
            return "Deep Research did not return a request ID or report content.", [], model_label, None

        _status(f"Research task created… Request ID: {request_id}")
        deadline = time.time() + RESEARCH_TIMEOUT_SECONDS
        finalizing_statuses = {"finalizing", "synthesizing", "writing", "generating"}
        complete_statuses = {"completed", "complete", "done", "success"}
        failed_statuses = {"failed", "error", "cancelled", "canceled"}

        while time.time() < deadline:
            result = tavily.get_research(request_id)
            status = _research_status(result)

            if status in complete_statuses:
                _status("Finalizing report…")
                content = _research_content(result)
                sources = _normalize_research_sources(_research_sources(result))
                if not content:
                    content = "Deep Research completed but returned no report content."
                return content, sources, model_label, request_id

            if status in failed_statuses:
                message = result.get("error") or result.get("message") or "No error detail was returned."
                return f"Deep Research failed for request `{request_id}`: {message}", [], model_label, request_id

            if status in finalizing_statuses:
                _status("Finalizing report…")
            else:
                _status("Research in progress…")
            time.sleep(RESEARCH_POLL_SECONDS)

        return (
            f"Deep Research is still running after {RESEARCH_TIMEOUT_SECONDS} seconds. "
            f"Request ID: `{request_id}`. Please try again shortly.",
            [],
            model_label,
            request_id,
        )
    except Exception as e:
        return f"Deep Research failed: {e}", [], model_label, None


# ── UI helpers ───────────────────────────────────────────────────────────────────

def _render_source_cards(chunks: list[dict], is_web: bool = False) -> None:
    if not chunks:
        return
    st.divider()
    st.caption("**Sources retrieved**")
    cols = st.columns(3)
    for i, chunk in enumerate(chunks):
        with cols[i % 3]:
            if is_web:
                url = chunk.get("source_url", "")
                domain = _domain(url) if url else "Source"
                source_ref = f"[{i+1}]" if chunk.get("is_research") else f"[w{i+1}]"
                source_link = f"[{domain}]({url})" if url else domain
                st.markdown(
                    f"**{source_ref}** {source_link}  \n"
                    f"_{chunk['title'][:55]}_"
                )
            else:
                page_url = f"{chunk['source_url']}#page={chunk['page']}"
                sim = chunk.get("similarity")
                sim_str = f" · {sim:.2f}" if sim is not None else ""
                st.markdown(
                    f"**[{i+1}]** [{chunk['title'][:45]}]({page_url})  \n"
                    f"Page {chunk['page']}{sim_str}"
                )


def _render_origin_label(source_kind: str) -> None:
    if source_kind == "research":
        label = "Generated from Tavily Deep Research"
    elif source_kind == "web":
        label = "Generated from Web search"
    else:
        label = "Generated from CE reports"
    bg = "#111827" if source_kind in {"web", "research"} else "#F3F4F6"
    fg = "#FFFFFF" if source_kind in {"web", "research"} else "#111827"
    st.markdown(
        f"""
        <div style="margin: 0.15rem 0 0.65rem;">
            <span style="
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                background: {bg};
                color: {fg};
                font-size: 0.8rem;
                font-weight: 700;
                line-height: 1;
                padding: 0.4rem 0.65rem;
            ">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_followup_block(suggestion: str | None, btn_key: str) -> None:
    """Render the suggested web follow-up below an internal answer."""
    if not TAVILY_ENABLED:
        return
    st.divider()
    if suggestion:
        st.markdown(
            "**Would you like to extend your question with a web search based on this recommended follow up question?**"
        )
        st.markdown(
            f"""
            <div style="
                margin: 0.35rem 0 0.65rem;
                color: #111827;
                font-size: 1rem;
                font-weight: 650;
                line-height: 1.45;
            ">{escape(suggestion)}</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Copy to chat", key=btn_key):
            _prefill_chat(suggestion, WEB_MODE)
            st.rerun()
    else:
        st.caption("**Extend with a web search** — switch to Web Search mode below and ask a follow-up.")


# ── Page config & auth ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nissan Insight RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
st.session_state.setdefault("authenticated", False)

if not st.session_state.authenticated:
    st.title("Nissan Insight RAG")
    st.caption(APP_VERSION)
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("Nissan Insight RAG")
st.caption(f"{APP_VERSION} · Ask questions across 121 Nissan customer experience research reports.")

# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.caption(APP_VERSION)
    st.markdown(
        "**Reports:** 121 PDFs (June 2023 – Nov 2025)  \n"
        "**Chunks:** 3,298  \n"
    )
    if not TAVILY_ENABLED:
        st.caption("Web search is disabled until `TAVILY_API_KEY` is added to Streamlit secrets.")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages      = []
        st.session_state.mode          = REPORTS_MODE
        st.session_state.pending_input = ""
        st.rerun()

# ── Session state defaults ───────────────────────────────────────────────────────

st.session_state.setdefault("messages",       [])
st.session_state.setdefault("mode",           REPORTS_MODE)
st.session_state.setdefault("pending_input",  "")
st.session_state.setdefault("form_counter",   0)


# ── Render chat history ──────────────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    kind = msg.get("kind", "user_internal" if msg["role"] == "user" else "assistant_internal")

    if msg["role"] == "user":
        with st.chat_message("user"):
            prefix = "🌐 " if kind == "user_web" else "Deep Research: " if kind == "user_research" else ""
            st.markdown(f"{prefix}{msg['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            source_kind = "research" if kind == "assistant_research" else "web" if kind == "assistant_web" else "reports"
            _render_origin_label(source_kind)
            model_label = msg.get("model", "")
            source_label = "  · Web sources" if kind == "assistant_web" else \
                           "  · Deep Research sources" if kind == "assistant_research" else ""
            request_id = msg.get("request_id")
            request_label = f"  · Request `{request_id}`" if request_id else ""
            if model_label:
                st.caption(f"_Answered with **{model_label}**{request_label}{source_label}_")
            _render_source_cards(msg.get("chunks", []), is_web=(kind in {"assistant_web", "assistant_research"}))
            if kind == "assistant_internal":
                _render_followup_block(msg.get("suggestion"), btn_key=f"cp_{idx}")


# ── Bottom bar ───────────────────────────────────────────────────────────────────

# Keep the chat composer pinned while explicitly constraining its height. Without
# the height resets, Streamlit's internal form wrapper can cover the main page.
st.markdown("""
<style>
:root {
    --sidebar-width: 200px;
    --composer-height: 88px;
    --streamlit-cloud-cta-space: 10rem;
}
section[data-testid="stSidebar"] {
    width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
    max-width: var(--sidebar-width) !important;
}
section[data-testid="stSidebar"] > div {
    width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
    max-width: var(--sidebar-width) !important;
}
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
div[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] {
    display: none !important;
}
.main .block-container {
    padding-bottom: 104px !important;
}
div[data-testid="stForm"] {
    position: fixed !important;
    right: 0 !important;
    bottom: 0 !important;
    left: var(--sidebar-width) !important;
    width: auto !important;
    max-width: none !important;
    min-width: 0 !important;
    z-index: 999 !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: 88px !important;
    overflow: visible !important;
    box-sizing: border-box !important;
    background-color: white;
    padding: 0.55rem calc(1rem + var(--streamlit-cloud-cta-space)) 0.65rem 1rem;
    border-top: 1px solid rgba(49, 51, 63, 0.2);
    box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
}
div[data-testid="stForm"] > div:first-child {
    height: auto !important;
    min-height: 0 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
    height: auto !important;
    min-height: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
    background-color: #111827 !important;
    border-color: #111827 !important;
    color: white !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #000000 !important;
    border-color: #000000 !important;
    color: white !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button p {
    color: inherit !important;
}
@media (max-width: 900px) {
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    .main .block-container {
        padding-bottom: 220px !important;
    }
    div[data-testid="stForm"] {
        left: 0 !important;
        width: auto !important;
        max-width: none !important;
        max-height: 210px !important;
        padding-left: 1rem;
        padding-right: calc(1rem + var(--streamlit-cloud-cta-space));
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.55rem !important;
    }
    div[data-testid="stForm"] div[data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
}
</style>
""", unsafe_allow_html=True)

mode_options = [REPORTS_MODE, WEB_MODE, DEEP_RESEARCH_MODE] if TAVILY_ENABLED else [REPORTS_MODE]
pending      = st.session_state.get("pending_input", "")
current_mode = st.session_state.get("mode", REPORTS_MODE)
if current_mode == "Reports":
    current_mode = REPORTS_MODE
    st.session_state.mode = REPORTS_MODE
elif current_mode == "Web":
    current_mode = WEB_MODE
    st.session_state.mode = WEB_MODE
mode_idx     = mode_options.index(current_mode) if current_mode in mode_options else 0

with st.form(f"chat_form_{st.session_state.form_counter}", clear_on_submit=False):
    form_id = st.session_state.form_counter
    col_input, col_mode, col_send = st.columns([4.8, 1.6, 0.8])
    with col_input:
        chat_text = st.text_input(
            "message",
            value=pending,
            key=f"chat_text_{form_id}",
            label_visibility="collapsed",
            placeholder="Ask a question about the CE reports…",
        )
    with col_mode:
        mode = st.selectbox(
            "Mode",
            mode_options,
            index=mode_idx,
            key=f"chat_mode_{form_id}",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.form_submit_button("↵", use_container_width=True)

# Consume the pending input now that the form has rendered it
st.session_state.pending_input = ""

# ── Handle send ──────────────────────────────────────────────────────────────────

if send and chat_text and chat_text.strip():
    question = chat_text.strip()
    st.session_state.mode = mode
    st.session_state.form_counter += 1  # cycle form key → fresh empty form on next render

    if mode == DEEP_RESEARCH_MODE:
        st.session_state.messages.append({"role": "user", "kind": "user_research", "content": question})
        with st.chat_message("user"):
            st.markdown(f"Deep Research: {question}")

        with st.chat_message("assistant"):
            status_box = st.empty()
            answer, research_chunks, model_used, request_id = ask_deep_research(
                question, status_box=status_box
            )
            status_box.empty()
            st.markdown(answer)
            _render_origin_label("research")
            request_label = f"  · Request `{request_id}`" if request_id else ""
            st.caption(f"_Answered with **{model_used}**{request_label}  · Deep Research sources_")
            _render_source_cards(research_chunks, is_web=True)

        st.session_state.messages.append({
            "role": "assistant", "kind": "assistant_research",
            "content": answer, "model": model_used,
            "chunks": research_chunks, "suggestion": None,
            "request_id": request_id,
        })
        st.session_state.mode = REPORTS_MODE   # auto-revert

    elif mode == WEB_MODE:
        st.session_state.messages.append({"role": "user", "kind": "user_web", "content": question})
        with st.chat_message("user"):
            st.markdown(f"🌐 {question}")

        with st.chat_message("assistant"):
            with st.spinner("Searching the web…"):
                try:
                    answer, web_chunks, model_used = ask_web(
                        question, st.session_state.messages[:-1]
                    )
                except genai_errors.ClientError as e:
                    if getattr(e, "status_code", None) == 429:
                        st.error("Both daily quotas are exhausted. Try again after midnight UTC.")
                        st.stop()
                    raise
            st.markdown(answer)
            _render_origin_label("web")
            st.caption(f"_Answered with **{model_used}**  · Web sources_")
            _render_source_cards(web_chunks, is_web=True)

        st.session_state.messages.append({
            "role": "assistant", "kind": "assistant_web",
            "content": answer, "model": model_used,
            "chunks": web_chunks, "suggestion": None,
        })
        st.session_state.mode = REPORTS_MODE   # auto-revert

    else:
        st.session_state.messages.append({"role": "user", "kind": "user_internal", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching reports…"):
                try:
                    answer, chunks, model_used, suggestion = ask_internal(
                        question, st.session_state.messages[:-1]
                    )
                except genai_errors.ClientError as e:
                    if getattr(e, "status_code", None) == 429:
                        st.error("Both daily quotas are exhausted. Try again after midnight UTC.")
                        st.stop()
                    raise
            st.markdown(answer)
            _render_origin_label("reports")
            st.caption(f"_Answered with **{model_used}**_")
            _render_source_cards(chunks, is_web=False)
            _render_followup_block(suggestion, btn_key=f"cp_new_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant", "kind": "assistant_internal",
            "content": answer, "model": model_used,
            "chunks": chunks, "suggestion": suggestion,
        })

    st.rerun()
