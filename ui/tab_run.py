"""
ui/tab_run.py
─────────────
Renders the "Run Lab" tab: task input, test inputs, strategy overview,
Run Comparison button, and Bad Prompt demo button.
"""

import streamlit as st
import anthropic

from config import STRATEGIES, STRATEGY_COLORS
from api_client import run_comparison, run_bad_prompt
from prompts import build_bad_prompt


def render_tab_run(api_key: str) -> None:
    """Render the full Run Lab tab content."""

    # ── Input row ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Task Definition</div>", unsafe_allow_html=True)
        task_input: str = st.text_area(
            "Describe the task",
            height=120,
            placeholder="e.g. classify customer sentiment as positive, negative, or neutral",
            help="The same description is used verbatim in all 4 prompt strategies.",
            key="task_input",
        )

    with col_right:
        st.markdown("<div class='section-header'>Test Inputs</div>", unsafe_allow_html=True)
        raw_inputs: str = st.text_area(
            "Enter test inputs (one per line)",
            height=120,
            placeholder="e.g. line 1\nline 2\nline 3",
            key="test_inputs",
        )

        st.markdown("---")

    # ── Strategy overview ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Strategies to Compare</div>", unsafe_allow_html=True)

    strategy_meta = [
        ("Zero-Shot",        "#60a5fa", "Direct instruction. No examples or scaffolding."),
        ("Few-Shot",         "#4ade80", "3 auto-generated labeled examples before the task."),
        ("Chain-of-Thought", "#fb923c", "Forces step-by-step reasoning before answering."),
        ("Role Prompting",   "#c084fc", "Expert persona with 20 years of domain experience."),
    ]

    cols = st.columns(4)
    for col, (strat, color, desc) in zip(cols, strategy_meta):
        col.markdown(
            f"""
            <div class='metric-card' style='border-top:3px solid {color};text-align:left;'>
                <div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;
                            text-transform:uppercase;color:{color};margin-bottom:8px;'>
                    {strat}
                </div>
                <div style='font-size:0.82rem;color:#9ca3af;line-height:1.5;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_col1, btn_col2, _ = st.columns([2, 2, 3])
    run_clicked = btn_col1.button("⚡ Run Comparison", use_container_width=True)
    bad_clicked = btn_col2.button("💀 Try a Bad Prompt", use_container_width=True)

    # ── Run Comparison ────────────────────────────────────────────────────────
    if run_clicked:
        if not api_key:
            st.error("🔑 Please enter your  API key in the sidebar.")
            return
        if not task_input.strip():
            st.error("📝 Please enter a task description.")
            return

        # Normalize input lines and guard against NBSP / invisible whitespace.
        normalized = []
        for line in raw_inputs.splitlines():
            clean = line.replace("\xa0", " ").strip()
            if clean:
                normalized.append(clean)

        st.write("DEBUG: raw_inputs:", repr(raw_inputs))
        st.write("DEBUG: normalized test inputs:", normalized)

        test_inputs = normalized
        if not test_inputs:
            st.error("📝 Please enter at least one test input.")
            return

        progress_bar = st.progress(0, text="Preparing…")

        def _progress(step: int, total: int, msg: str) -> None:
            progress_bar.progress(step / total, text=msg)

        try:
            with st.spinner("🧪 Running all strategies — this may take a moment…"):
                st.session_state["results"] = run_comparison(
                    api_key,
                    task_input.strip(),
                    test_inputs,
                    progress_callback=_progress,
                )
            progress_bar.empty()
            st.success("✅ Comparison complete! Head to the **Results** and **Analytics** tabs.")
        except anthropic.AuthenticationError:
            progress_bar.empty()
            st.error("🔑 Invalid API key — please check the key in the sidebar.")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"❌ Unexpected error: {exc}")

    # ── Bad Prompt Demo ───────────────────────────────────────────────────────
    if bad_clicked:
        if not api_key:
            st.error("🔑 Please enter your  API key in the sidebar.")
            return
        if not task_input.strip():
            st.error("📝 Please enter a task description so we know what the bad prompt is failing at.")
            return

        test_inputs = [l.strip() for l in raw_inputs.strip().splitlines() if l.strip()]
        if not test_inputs:
            st.error("📝 Please enter at least one test input.")
            return

        with st.spinner("💀 Running deliberately vague prompt…"):
            bad_responses = run_bad_prompt(api_key, task_input.strip(), test_inputs)
            st.session_state["bad_responses"] = bad_responses
            st.session_state["bad_task"]      = task_input.strip()
            st.session_state["bad_inputs"]    = test_inputs

    # ── Inline bad-prompt results ─────────────────────────────────────────────
    if "bad_responses" in st.session_state:
        st.markdown("---")
        st.markdown(
            """
            <div class='lab-danger'>
                <b>💀 Bad Prompt Demo</b> — Here's what a vague prompt like
                <code>do the thing with {input}</code> produces. Notice how the model
                can't infer intent, gives generic or off-topic responses, and lacks the
                structure to reliably complete your task. This illustrates why prompt
                engineering matters.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show the raw bad prompt
        bad_prompt_text = build_bad_prompt(
            st.session_state["bad_task"],
            st.session_state["bad_inputs"][0],
        )
        st.markdown(
            f"<div class='prompt-block'>{bad_prompt_text}</div>",
            unsafe_allow_html=True,
        )

        for inp, resp in zip(
            st.session_state["bad_inputs"],
            st.session_state["bad_responses"],
        ):
            label = f"Input: {inp[:60]}…" if len(inp) > 60 else f"Input: {inp}"
            with st.expander(label):
                st.markdown(
                    f"<div style='color:#fca5a5;font-family:JetBrains Mono,monospace;"
                    f"font-size:0.82rem;'>{resp}</div>",
                    unsafe_allow_html=True,
                )
