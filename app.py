"""
app.py
──────
Streamlit frontend for the Multilingual RAG Pipeline.

Features
────────
• Chat-style Q&A interface with conversation history
• Language auto-detection badge on each query
• Source citations with page numbers, language tags, and similarity scores
• Sidebar: index stats, top-k slider, model selector, ingest trigger
• Handles errors gracefully with user-friendly messages
• Session state preserves conversation across reruns

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multilingual PDF Q&A",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (after page config) ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import GROQ_MODEL, TOP_K, PDF_FOLDER, CHROMA_DB_PATH, COLLECTION_NAME
from utils.chunker import detect_language

logger = logging.getLogger(__name__)

# ── Language display helpers ───────────────────────────────────────────────────

_LANG_LABELS: dict[str, str] = {
    "en": "🇬🇧 English",
    "fr": "🇫🇷 French",
    "hi": "🇮🇳 Hindi",
    "kn": "🇮🇳 Kannada",
    "ta": "🇮🇳 Tamil",
    "te": "🇮🇳 Telugu",
    "de": "🇩🇪 German",
    "es": "🇪🇸 Spanish",
    "pt": "🇧🇷 Portuguese",
    "it": "🇮🇹 Italian",
    "ar": "🇸🇦 Arabic",
    "zh": "🇨🇳 Chinese",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "ru": "🇷🇺 Russian",
}

def _lang_label(code: str) -> str:
    return _LANG_LABELS.get(code, f"🌐 {code.upper()}")


def _distance_to_pct(dist: float) -> str:
    """Convert cosine distance (0–2) to a relevance percentage string."""
    pct = max(0.0, min(100.0, (1.0 - dist / 2.0) * 100))
    return f"{pct:.0f}%"


# ── Cached retriever ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval engine…")
def _load_retriever(top_k: int):
    """Load retriever once; cache across sessions."""
    from retriever import MultilingualRetriever
    return MultilingualRetriever(top_k=top_k)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of {role, content, sources?, lang?}
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K
    if "retriever_error" not in st.session_state:
        st.session_state.retriever_error = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Index stats ───────────────────────────────────────────────────────
        st.subheader("📚 Knowledge Base")
        if st.button("🔄 Refresh Stats"):
            st.cache_resource.clear()

        try:
            retriever = _load_retriever(st.session_state.top_k)
            stats = retriever.collection_stats()
            if "error" in stats:
                st.error(f"Vector store not ready: {stats['error']}")
                st.info("Run `python ingest.py` to index your PDFs first.")
            else:
                st.metric("Indexed chunks", stats.get("total_chunks", 0))
                st.caption(f"Collection: `{stats.get('collection', COLLECTION_NAME)}`")
                st.caption(f"DB path: `{stats.get('db_path', CHROMA_DB_PATH)}`")
        except Exception as exc:
            st.warning(f"Could not connect to vector store: {exc}")

        st.divider()

        # ── Retrieval settings ────────────────────────────────────────────────
        st.subheader("🔍 Retrieval")
        new_top_k = st.slider(
            "Number of chunks to retrieve (Top-K)",
            min_value=1, max_value=15,
            value=st.session_state.top_k,
            help="Higher values provide more context but may include less relevant chunks.",
        )
        if new_top_k != st.session_state.top_k:
            st.session_state.top_k = new_top_k
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        # ── Ingest trigger ────────────────────────────────────────────────────
        st.subheader("📥 Ingest PDFs")
        pdf_folder_input = st.text_input(
            "PDF Folder",
            value=str(PDF_FOLDER),
            help="Absolute or relative path to the folder containing PDF files.",
        )
        reset_flag = st.checkbox(
            "Reset index (re-ingest everything)",
            value=False,
            help="⚠️ This will delete the existing vector store and start fresh.",
        )
        if st.button("▶️ Run Ingestion", type="primary"):
            _run_ingestion(pdf_folder_input, reset_flag)

        st.divider()

        # ── Conversation management ───────────────────────────────────────────
        st.subheader("💬 Conversation")
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # ── Info ──────────────────────────────────────────────────────────────
        st.subheader("ℹ️ System Info")
        st.caption(f"**LLM model:** `{GROQ_MODEL}`")
        st.caption("**Embedding model:**")
        st.caption("`paraphrase-multilingual-mpnet-base-v2`")
        st.caption("**Supported languages:** EN, FR, HI, KN + 46 others")


def _run_ingestion(folder: str, reset: bool) -> None:
    """Trigger ingest.py as a subprocess and stream stdout to the UI."""
    cmd = [sys.executable, "ingest.py", "--pdf-folder", folder]
    if reset:
        cmd.append("--reset")

    with st.spinner("Ingesting PDFs… (this may take several minutes)"):
        output_placeholder = st.empty()
        log_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent),
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                log_lines.append(line.rstrip())
                output_placeholder.code("\n".join(log_lines[-30:]), language="")
            proc.wait()

            if proc.returncode == 0:
                st.success("✅ Ingestion complete! Refresh stats to see updated chunk count.")
                st.cache_resource.clear()
            else:
                st.error(f"❌ Ingestion failed with exit code {proc.returncode}.")
        except FileNotFoundError:
            st.error("Could not find `ingest.py`. Make sure you are running from the project root.")
        except Exception as exc:
            st.error(f"Ingestion error: {exc}")


# ── Chat interface ────────────────────────────────────────────────────────────

def _render_chat() -> None:
    st.title("🌍 Multilingual PDF Question Answering")
    st.caption(
        "Ask questions about your PDF library in **any language** — "
        "Kannada, Hindi, English, French, and 46+ more."
    )

    # ── Render conversation history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                lang_code = msg.get("lang", "en")
                st.markdown(msg["content"])
                st.caption(f"Detected language: {_lang_label(lang_code)}")
            else:
                st.markdown(msg["content"])
                _render_sources(msg.get("sources", []))

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask a question in any language…",
        key="chat_input",
    )

    if not user_input:
        return

    # Detect query language
    query_lang = detect_language(user_input)

    # Add user message to history
    st.session_state.messages.append({
        "role"    : "user",
        "content" : user_input,
        "lang"    : query_lang,
    })

    # Display user turn immediately
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"Detected language: {_lang_label(query_lang)}")

    # ── Generate answer ───────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                retriever = _load_retriever(st.session_state.top_k)
                result    = retriever.query(user_input)
            except Exception as exc:
                st.error(f"⚠️ Error: {exc}")
                st.info(
                    "Make sure you have:\n"
                    "1. Set `GROQ_API_KEY` in your `.env` file.\n"
                    "2. Run `python ingest.py` to build the index."
                )
                return

        # Display answer
        if result.error and not result.answer:
            st.error(result.answer)
        else:
            st.markdown(result.answer)

        _render_sources(result.sources)

    # Save assistant turn to history
    st.session_state.messages.append({
        "role"    : "assistant",
        "content" : result.answer,
        "sources" : result.sources,
    })


def _render_sources(sources) -> None:
    """Render collapsible source citations below an answer."""
    if not sources:
        return

    with st.expander(f"📄 Sources ({len(sources)} chunk(s))", expanded=False):
        for idx, chunk in enumerate(sources, start=1):
            relevance = _distance_to_pct(chunk.distance)
            lang      = _lang_label(chunk.language)
            tags = []
            if chunk.has_tables:
                tags.append("📊 table")
            if chunk.has_images:
                tags.append("🖼️ image")
            tag_str = "  " + "  ".join(tags) if tags else ""

            st.markdown(
                f"**[Source {idx}]** `{chunk.source}` — "
                f"page **{chunk.page_num}** | {lang} | "
                f"relevance **{relevance}**{tag_str}"
            )
            with st.container():
                st.text_area(
                    label     = f"Excerpt from {chunk.source} p.{chunk.page_num}",
                    value     = chunk.text[:800] + ("…" if len(chunk.text) > 800 else ""),
                    height    = 120,
                    disabled  = True,
                    key       = f"src_{idx}_{id(chunk)}_{int(time.time()*1000) % 100000}",
                    label_visibility="collapsed",
                )
            st.divider()


# ── Welcome / onboarding screen ───────────────────────────────────────────────

def _render_welcome() -> None:
    """Show onboarding instructions when the knowledge base is empty."""
    st.info(
        "👋 **Welcome!** Your knowledge base appears to be empty.\n\n"
        "**To get started:**\n"
        "1. Place your PDF files in the `pdfs/` folder (or configure a custom path in the sidebar).\n"
        "2. Click **▶️ Run Ingestion** in the sidebar to index your documents.\n"
        "3. Once ingestion is complete, ask your first question below!\n\n"
        "Supported languages include English, French, Hindi, Kannada, and 46+ others."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    _render_sidebar()

    # Check if knowledge base is ready
    kb_ready = False
    try:
        retriever = _load_retriever(st.session_state.top_k)
        stats = retriever.collection_stats()
        kb_ready = stats.get("total_chunks", 0) > 0
    except Exception:
        kb_ready = False

    if not kb_ready and not st.session_state.messages:
        _render_welcome()

    _render_chat()


if __name__ == "__main__":
    main()
