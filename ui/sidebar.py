"""
ui/sidebar.py
─────────────
Renders the sidebar and returns the API key entered by the user.
Kept in its own module so app.py stays declarative.
"""

import streamlit as st
from config import MODEL, MAX_TOKENS, STRATEGIES


def render_sidebar() -> str:
    """
    Render the sidebar UI and return the API key string.
    (Empty string if the user hasn't typed one yet.)
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style='margin-bottom:24px;'>
                <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                            color:#e8eaf0;letter-spacing:-0.02em;'>🧪 Prompt Testing Lab</div>
                <div style='font-size:0.75rem;color:#4b5563;margin-top:4px;
                            letter-spacing:0.06em;text-transform:uppercase;font-weight:600;'>
                    Strategy Comparison Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── API Key ───────────────────────────────────────────────────────────
        api_key: str = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your key is used only in this session and never stored.",
        )


        # ── Model info ────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Model Info</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                        color:#6b7280;background:#0d0f14;border:1px solid #1e2130;
                        border-radius:8px;padding:10px 12px;'>
                <span style='color:#1e8f5e;'>model:</span> {MODEL}<br>
                <span style='color:#1e8f5e;'>max_tokens:</span> {MAX_TOKENS}<br>
                <span style='color:#1e8f5e;'>strategies:</span> {len(STRATEGIES)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Explainer ─────────────────────────────────────────────────────────
        with st.expander("ℹ️ What is this?"):
            st.markdown(
                """
                **Prompt Testing Lab** lets you systematically compare 4 core prompting strategies:

                - 🔵 **Zero-Shot** — bare instruction, no examples
                - 🟢 **Few-Shot** — 3 auto-generated labeled examples
                - 🟠 **Chain-of-Thought** — step-by-step reasoning first
                - 🟣 **Role Prompting** — expert persona framing

                Enter a task + test inputs, click **Run Comparison**, and instantly see
                how each strategy performs across consistency, length, and robustness.
                """
            )

    return api_key
