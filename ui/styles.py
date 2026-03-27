"""
ui/styles.py
────────────
Injects the global CSS for the dark-lab aesthetic.
Call `inject_styles()` once at the top of app.py after `set_page_config`.
"""

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ── App shell ─────────────────────────────────────────────────────────── */
.stApp          { background: #0d0f14; color: #e8eaf0; }
.block-container { padding-top: 2rem; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161920; border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid #2a2d38;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 600;
    font-size: 0.85rem; letter-spacing: 0.03em;
    padding: 8px 18px; transition: all 0.2s;
}
.stTabs [aria-selected="true"] { background: #1e8f5e !important; color: #fff !important; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #10131a; border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #1a1d27; border: 1px solid #2a2d3e; color: #e8eaf0;
    border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
}

/* ── Inputs ────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: #161920 !important; border: 1px solid #2a2d38 !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #1e8f5e !important;
    box-shadow: 0 0 0 2px rgba(30,143,94,0.2) !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e8f5e, #15705f) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.9rem !important; letter-spacing: 0.04em !important;
    padding: 10px 24px !important; transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(30,143,94,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(30,143,94,0.4) !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #161920 !important; border: 1px solid #2a2d38 !important;
    border-radius: 10px !important; color: #9ca3af !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-card {
    background: #161920; border: 1px solid #2a2d38;
    border-radius: 12px; padding: 18px 22px; text-align: center;
}
.metric-card .label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 8px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #1e8f5e;
}

/* ── Strategy badges ───────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-zs  { background: #1e2a3a; color: #60a5fa; border: 1px solid #2a3d5a; }
.badge-fs  { background: #1e2e24; color: #4ade80; border: 1px solid #2a4034; }
.badge-cot { background: #2a2620; color: #fb923c; border: 1px solid #3d3020; }
.badge-rp  { background: #2a1e30; color: #c084fc; border: 1px solid #3d2a45; }
.badge-bad { background: #2a1e1e; color: #f87171; border: 1px solid #3d2a2a; }

/* ── Prompt code blocks ────────────────────────────────────────────────── */
.prompt-block {
    background: #0a0c11; border: 1px solid #1e8f5e33;
    border-left: 3px solid #1e8f5e; border-radius: 8px; padding: 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: #a5f3c8; white-space: pre-wrap; line-height: 1.7; margin-top: 8px;
}

/* ── Consistency score bar ─────────────────────────────────────────────── */
.score-bar-wrap {
    background: #1e2130; border-radius: 6px; height: 8px; margin: 6px 0; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }

/* ── Section headers ───────────────────────────────────────────────────── */
.section-header {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4b5563; margin-bottom: 14px;
    padding-bottom: 8px; border-bottom: 1px solid #1e2130;
}

/* ── Alert boxes ───────────────────────────────────────────────────────── */
.lab-alert {
    background: #1a2a1a; border: 1px solid #1e8f5e44; border-radius: 10px;
    padding: 14px 18px; color: #86efac; font-size: 0.85rem; margin: 10px 0;
}
.lab-danger {
    background: #1a1010; border: 1px solid #ef444444; border-radius: 10px;
    padding: 14px 18px; color: #fca5a5; font-size: 0.85rem; margin: 10px 0;
}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { background: #161920; border-radius: 12px; overflow: hidden; }
</style>
"""


def inject_styles() -> None:
    """Inject the global CSS into the Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)
