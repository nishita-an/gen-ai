"""
ui/tab_inspector.py
───────────────────
Renders the "Prompt Inspector" tab: exact prompts sent per strategy,
few-shot examples used, and the bad-prompt anatomy section.
"""

import streamlit as st

from config import STRATEGIES, STRATEGY_BADGE
from prompts import build_bad_prompt


def _escape(text: str) -> str:
    """Escape HTML special characters for safe inline rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_tab_inspector() -> None:
    """Render the full Prompt Inspector tab content."""

    if "results" not in st.session_state:
        st.markdown(
            "<div class='lab-alert'>"
            "Run the comparison first to inspect the exact prompts that were sent."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    r = st.session_state["results"]

    st.markdown(
        "<div class='lab-alert'>"
        "📚 Below are the <b>exact prompts</b> sent to the model for each strategy "
        "and each input. Use these to understand what each technique looks like under "
        "the hood — and to write better prompts yourself."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Few-shot examples used ────────────────────────────────────────────────
    if r.get("few_shot_examples"):
        with st.expander("🔬 Auto-Generated Few-Shot Examples (shared across all inputs)"):
            for i, ex in enumerate(r["few_shot_examples"], 1):
                st.markdown(
                    f"<div style='margin-bottom:10px;'>"
                    f"<span style='color:#4ade80;font-weight:700;"
                    f"font-family:JetBrains Mono,monospace;font-size:0.8rem;'>Example {i}</span><br>"
                    f"<div class='prompt-block'>"
                    f"Input:  {_escape(ex['input'])}\n"
                    f"Output: {_escape(ex['output'])}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input selector ────────────────────────────────────────────────────────
    selected_idx: int = st.selectbox(
        "View prompts for input:",
        options=list(range(len(r["inputs"]))),
        format_func=lambda i: (
            f"Input {i + 1}: {r['inputs'][i][:60]}…"
            if len(r["inputs"][i]) > 60
            else f"Input {i + 1}: {r['inputs'][i]}"
        ),
    )

    # ── Per-strategy prompt + response ────────────────────────────────────────
    for strat in STRATEGIES:
        badge = STRATEGY_BADGE[strat]
        prompt_text = r["prompts"][strat][selected_idx]
        resp_text   = r["responses"][strat][selected_idx]

        with st.expander(strat):
            st.markdown(f"<span class='badge {badge}'>{strat}</span>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='prompt-block'>{_escape(prompt_text)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:14px;'>"
                f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:#4b5563;margin-bottom:6px;'>Model Response</div>"
                f"<div style='background:#0d0f14;border:1px solid #2a2d38;border-radius:8px;"
                f"padding:12px 14px;font-size:0.82rem;color:#d1d5db;"
                f"font-family:JetBrains Mono,monospace;line-height:1.6;'>"
                f"{_escape(resp_text)}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Bad Prompt Anatomy ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-header'>Bad Prompt Anatomy</div>",
        unsafe_allow_html=True,
    )

    bad_prompt_text = build_bad_prompt(r["task"], r["inputs"][0])
    st.markdown(
        f"<span class='badge badge-bad'>💀 Bad Prompt</span>"
        f"<div class='prompt-block'>{_escape(bad_prompt_text)}</div>"
        f"<div class='lab-danger' style='margin-top:10px;'>"
        f"<b>Why this fails:</b> No task context, no output format, no examples, no persona. "
        f"The model has no idea what \"the thing\" is, what format to use, or what success looks like. "
        f"Compare this to any of the 4 strategies above and the difference is immediately clear."
        f"</div>",
        unsafe_allow_html=True,
    )
