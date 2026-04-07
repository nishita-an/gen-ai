# """
# app.py — Multilingual PDF Q&A (Streamlit UI)
# Updated: Direct PDF upload instead of folder path
# """

# from __future__ import annotations

# import logging
# import subprocess
# import sys
# import time
# from pathlib import Path

# import streamlit as st

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Multilingual PDF Q&A",
#     page_icon="🌍",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# sys.path.insert(0, str(Path(__file__).parent))

# from config import GROQ_MODEL, TOP_K, COLLECTION_NAME
# from utils.chunker import detect_language

# logger = logging.getLogger(__name__)

# # ── Language labels ───────────────────────────────────────────────────────────
# _LANG_LABELS = {
#     "en": "🇬🇧 English",
#     "fr": "🇫🇷 French",
#     "hi": "🇮🇳 Hindi",
#     "kn": "🇮🇳 Kannada",
#     "ta": "🇮🇳 Tamil",
#     "te": "🇮🇳 Telugu",
#     "de": "🇩🇪 German",
#     "es": "🇪🇸 Spanish",
#     "pt": "🇧🇷 Portuguese",
#     "it": "🇮🇹 Italian",
#     "ar": "🇸🇦 Arabic",
#     "zh": "🇨🇳 Chinese",
#     "ja": "🇯🇵 Japanese",
#     "ko": "🇰🇷 Korean",
#     "ru": "🇷🇺 Russian",
# }

# def _lang_label(code: str):
#     return _LANG_LABELS.get(code, f"🌐 {code}")

# # ── Cached retriever ──────────────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading retrieval engine...")
# def _load_retriever(top_k: int):
#     from retriever import MultilingualRetriever
#     return MultilingualRetriever(top_k=top_k)

# # ── Session state init ────────────────────────────────────────────────────────
# def _init_session():
#     defaults = {
#         "messages": [],
#         "top_k": TOP_K,
#         "uploaded_files": [],
#     }

#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# # ── Run ingestion ─────────────────────────────────────────────────────────────
# def _run_ingestion(folder: str, reset=True):

#     cmd = [
#         sys.executable,
#         "ingest.py",
#         "--pdf-folder",
#         folder
#     ]

#     if reset:
#         cmd.append("--reset")

#     log_box = st.empty()
#     logs = []

#     try:
#         proc = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             cwd=str(Path(__file__).parent)
#         )

#         for line in proc.stdout:
#             logs.append(line)
#             log_box.code("".join(logs[-15:]))

#         proc.wait()

#         if proc.returncode == 0:
#             st.success("Ingestion complete")
#             st.cache_resource.clear()
#             time.sleep(1)
#             st.rerun()

#         else:
#             st.error("Ingestion failed")

#     except Exception as e:
#         st.error(str(e))

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# def _render_sidebar():

#     with st.sidebar:

#         st.title("Settings")

#         # Upload PDFs
#         st.subheader("Upload PDFs")

#         uploaded_files = st.file_uploader(
#             "Upload one or more PDFs",
#             type="pdf",
#             accept_multiple_files=True
#         )

#         if uploaded_files:

#             save_dir = Path("data/uploaded_pdfs")
#             save_dir.mkdir(parents=True, exist_ok=True)

#             for f in uploaded_files:

#                 file_path = save_dir / f.name

#                 with open(file_path, "wb") as out:
#                     out.write(f.getbuffer())

#             st.success(f"{len(uploaded_files)} file(s) uploaded")

#             if st.button("Process PDFs"):

#                 _run_ingestion(
#                     folder=str(save_dir),
#                     reset=True
#                 )

#         st.divider()

#         # Retrieval settings
#         st.subheader("Retrieval")

#         st.slider(
#             "Top K chunks",
#             1,
#             15,
#             key="top_k"
#         )

#         st.divider()

#         # System info
#         st.subheader("System")

#         st.caption(f"LLM: {GROQ_MODEL}")
#         st.caption("Embeddings: multilingual-mpnet")

#         if st.button("Clear chat"):
#             st.session_state.messages = []
#             st.rerun()

#         # DB stats
#         try:

#             r = _load_retriever(st.session_state.top_k)

#             stats = r.collection_stats()

#             if stats.get("total_chunks"):

#                 st.metric(
#                     "Indexed chunks",
#                     stats["total_chunks"]
#                 )

#                 st.caption(
#                     f"Collection: {stats.get('collection', COLLECTION_NAME)}"
#                 )

#         except:
#             pass

# # ── Chat UI ───────────────────────────────────────────────────────────────────
# def _render_chat():

#     st.title("Multilingual PDF Question Answering")

#     st.caption(
#         "Ask questions from PDFs in any language"
#     )

#     # show history
#     for msg in st.session_state.messages:

#         with st.chat_message(msg["role"]):

#             st.markdown(msg["content"])

#             if msg["role"] == "assistant":

#                 _render_sources(msg.get("sources", []))

#     # input box
#     question = st.chat_input(
#         "Ask a question"
#     )

#     if question:

#         lang = detect_language(question)

#         st.session_state.messages.append(
#             {
#                 "role": "user",
#                 "content": question,
#                 "lang": lang
#             }
#         )

#         with st.chat_message("user"):
#             st.markdown(question)
#             st.caption(_lang_label(lang))

#         with st.chat_message("assistant"):

#             with st.spinner("Searching..."):

#                 try:

#                     retriever = _load_retriever(
#                         st.session_state.top_k
#                     )

#                     result = retriever.query(question)

#                     st.markdown(result.answer)

#                     _render_sources(result.sources)

#                     st.session_state.messages.append(
#                         {
#                             "role": "assistant",
#                             "content": result.answer,
#                             "sources": result.sources
#                         }
#                     )

#                 except Exception as e:

#                     st.error(str(e))

# # ── show sources ──────────────────────────────────────────────────────────────
# def _render_sources(sources):

#     if not sources:
#         return

#     with st.expander(f"Sources ({len(sources)})"):

#         for i, s in enumerate(sources):

#             st.markdown(
#                 f"Source {i+1}: {s.source} (page {s.page_num})"
#             )

#             st.text_area(
#                 label=f"src_{i}",
#                 value=s.text[:500],
#                 height=120,
#                 disabled=True,
#                 label_visibility="collapsed"
#             )

# # ── main ──────────────────────────────────────────────────────────────────────
# def main():

#     _init_session()

#     _render_sidebar()

#     _render_chat()

# if __name__ == "__main__":

#     main()






"""
app.py — Multilingual PDF Q&A (Streamlit UI)
Updated: Direct PDF upload instead of folder path
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual PDF Q&A",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from config import GROQ_MODEL, TOP_K, COLLECTION_NAME
from utils.chunker import detect_language

logger = logging.getLogger(__name__)

# ── Language labels ───────────────────────────────────────────────────────────
_LANG_LABELS = {
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

def _lang_label(code: str):
    return _LANG_LABELS.get(code, f"🌐 {code}")

# ── Cached retriever ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval engine...")
def _load_retriever(top_k: int):
    from retriever import MultilingualRetriever
    return MultilingualRetriever(top_k=top_k)

# ── Session state init ────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "messages": [],
        "top_k": TOP_K,
        "uploaded_files": [],
        "input_key": 0,           # <-- used to reset the text input
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Run ingestion ─────────────────────────────────────────────────────────────
def _run_ingestion(folder: str, reset=True):
    cmd = [
        sys.executable,
        "ingest.py",
        "--pdf-folder",
        folder,
    ]
    if reset:
        cmd.append("--reset")

    log_box = st.empty()
    logs = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).parent),
        )

        for line in proc.stdout:
            logs.append(line)
            log_box.code("".join(logs[-15:]))

        proc.wait()

        if proc.returncode == 0:
            st.success("✅ Ingestion complete")
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Ingestion failed")

    except Exception as e:
        st.error(str(e))

# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")

        # Upload PDFs
        st.subheader("📄 Upload PDFs")

        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type="pdf",
            accept_multiple_files=True,
        )

        if uploaded_files:
            save_dir = Path("data/uploaded_pdfs")
            save_dir.mkdir(parents=True, exist_ok=True)

            for f in uploaded_files:
                file_path = save_dir / f.name
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())

            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

            if st.button("⚙️ Process PDFs", use_container_width=True):
                _run_ingestion(folder=str(save_dir), reset=True)

        st.divider()

        # Retrieval settings
        st.subheader("🔍 Retrieval")
        st.slider("Top K chunks", 1, 15, key="top_k")

        st.divider()

        # System info
        st.subheader("🖥️ System")
        st.caption(f"LLM: {GROQ_MODEL}")
        st.caption("Embeddings: multilingual-mpnet")

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # DB stats
        try:
            r = _load_retriever(st.session_state.top_k)
            stats = r.collection_stats()
            if stats.get("total_chunks"):
                st.metric("Indexed chunks", stats["total_chunks"])
                st.caption(f"Collection: {stats.get('collection', COLLECTION_NAME)}")
        except Exception:
            pass

# ── show sources ──────────────────────────────────────────────────────────────
def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources):
            st.markdown(f"**Source {i+1}:** `{s.source}` — page {s.page_num}")
            st.text_area(
                label=f"src_{i}",
                value=s.text[:500],
                height=120,
                disabled=True,
                label_visibility="collapsed",
            )

# ── Handle question submission ────────────────────────────────────────────────
def _handle_question(question: str):
    """Process a question and append result to session messages."""
    question = question.strip()
    if not question:
        return

    lang = detect_language(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "lang": lang}
    )

    try:
        retriever = _load_retriever(st.session_state.top_k)
        result = retriever.query(question)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )
    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ Error: {e}", "sources": []}
        )

    # Bump key to reset the text input widget
    st.session_state.input_key += 1

# ── Chat UI ───────────────────────────────────────────────────────────────────
def _render_chat():
    st.title("🌍 Multilingual PDF Question Answering")
    st.caption("Ask questions from your uploaded PDFs in any language.")

    # ── Message history ───────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "user":
                    st.caption(_lang_label(msg.get("lang", "en")))
                if msg["role"] == "assistant":
                    _render_sources(msg.get("sources", []))

    # ── Input row (always visible at bottom) ─────────────────────────────────
    st.divider()

    col1, col2 = st.columns([9, 1])

    with col1:
        question = st.text_input(
            label="Your question",
            placeholder="💬 Ask a question about your PDFs…",
            label_visibility="collapsed",
            key=f"user_input_{st.session_state.input_key}",
        )

    with col2:
        send_clicked = st.button("Send ➤", use_container_width=True)

    # Trigger on button click OR Enter key (non-empty input)
    if (send_clicked or question) and question.strip():
        with st.spinner("🔍 Searching…"):
            _handle_question(question)
        st.rerun()

    # Show a hint if no PDFs are indexed yet
    try:
        r = _load_retriever(st.session_state.top_k)
        stats = r.collection_stats()
        if not stats.get("total_chunks"):
            st.info("⬅️ Upload and process PDFs from the sidebar to get started.")
    except Exception:
        st.info("⬅️ Upload and process PDFs from the sidebar to get started.")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    _init_session()
    _render_sidebar()
    _render_chat()

if __name__ == "__main__":
    main()