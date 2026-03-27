"""
app.py — Prompt Testing Lab
════════════════════════════
Slim entry point. All business logic lives in:
  config.py        — constants
  prompts.py       — prompt builders
  api_client.py    — Anthropic API calls
  analytics.py     — scoring & ranking
  ui/styles.py     — CSS injection
  ui/sidebar.py    — sidebar component
  ui/tab_run.py    — Run Lab tab
  ui/tab_results.py    — Results tab
  ui/tab_analytics.py  — Analytics tab
  ui/tab_inspector.py  — Prompt Inspector tab

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Prompt Testing Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles & Sidebar ──────────────────────────────────────────────────────────
from ui.styles  import inject_styles
from ui.sidebar import render_sidebar

inject_styles()
api_key = render_sidebar()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='margin-bottom:32px;'>
        <h1 style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                   color:#e8eaf0;margin:0;letter-spacing:-0.03em;'>
            Prompt Testing Lab
        </h1>
        <p style='color:#4b5563;font-size:0.95rem;margin:6px 0 0;'>
            Compare Zero-Shot · Few-Shot · Chain-of-Thought · Role Prompting — side by side.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_analytics, tab_inspector = st.tabs(
    ["🚀  Run Lab", "📊  Results", "📈  Analytics", "🔍  Prompt Inspector"]
)

from ui.tab_run       import render_tab_run
from ui.tab_results   import render_tab_results
from ui.tab_analytics import render_tab_analytics
from ui.tab_inspector import render_tab_inspector

with tab_run:
    render_tab_run(api_key)

with tab_results:
    render_tab_results()

with tab_analytics:
    render_tab_analytics()

with tab_inspector:
    render_tab_inspector()
