"""
ui/tab_results.py
─────────────────
Renders the "Results" tab: summary metrics, comparison dataframe,
and per-row expandable full-response panels.
"""

import streamlit as st
import pandas as pd

from config import STRATEGIES, STRATEGY_BADGE, CELL_TRUNCATE
from analytics import score_to_color


def _truncate(text: str, limit: int = CELL_TRUNCATE) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def render_tab_results() -> None:
    """Render the full Results tab content."""

    if "results" not in st.session_state:
        st.markdown(
            "<div class='lab-alert'>"
            "No results yet. Head to the <b>Run Lab</b> tab, enter your task and inputs, "
            "then click <b>Run Comparison</b>."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    r = st.session_state["results"]

    # ── Summary metrics ────────────────────────────────────────────────────────
    avg_score  = sum(r["scores"]) / max(len(r["scores"]), 1)
    n_outliers = sum(1 for o in r["outliers"] if o is not None)

    m_cols = st.columns(4)
    for col, label, val in zip(
        m_cols,
        ["Task Inputs", "Avg Consistency", "Outlier Rows", "Strategies Run"],
        [str(len(r["inputs"])), f"{avg_score:.0%}", str(n_outliers), "4"],
    ):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison dataframe ───────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Response Comparison Table</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for idx, inp in enumerate(r["inputs"]):
        rows.append(
            {
                "Test Input":         inp,
                "Zero-Shot":          _truncate(r["responses"]["Zero-Shot"][idx]),
                "Few-Shot":           _truncate(r["responses"]["Few-Shot"][idx]),
                "CoT":                _truncate(r["responses"]["Chain-of-Thought"][idx]),
                "Role Prompt":        _truncate(r["responses"]["Role Prompting"][idx]),
                "Consistency Score":  f"{r['scores'][idx]:.0%}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per-row detail expanders ───────────────────────────────────────────────
    st.markdown("<br><div class='section-header'>Full Responses by Input</div>", unsafe_allow_html=True)

    for idx, inp in enumerate(r["inputs"]):
        score   = r["scores"][idx]
        outlier = r["outliers"][idx]
        color   = score_to_color(score)

        label = f"Input {idx + 1}: {inp[:70]}{'…' if len(inp) > 70 else ''}"
        with st.expander(label):
            # Score bar + outlier flag
            outlier_html = (
                f"<span style='color:#f87171;font-size:0.8rem;font-weight:700;'>"
                f"⚠ Outlier: {outlier}</span>"
                if outlier else ""
            )
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px;'>
                    <div>
                        <div style='font-size:0.68rem;color:#6b7280;text-transform:uppercase;
                                    font-weight:700;letter-spacing:0.08em;'>Consistency</div>
                        <div style='font-family:JetBrains Mono,monospace;font-weight:700;
                                    color:{color};font-size:1.2rem;'>{score:.0%}</div>
                    </div>
                    <div class='score-bar-wrap' style='flex:1;'>
                        <div class='score-bar-fill'
                             style='width:{score * 100:.0f}%;background:{color};'></div>
                    </div>
                    {outlier_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Four strategy columns
            resp_cols = st.columns(4)
            for col, strat, badge in zip(resp_cols, STRATEGIES, STRATEGY_BADGE.values()):
                resp = r["responses"][strat][idx]
                safe = resp.replace("<", "&lt;").replace(">", "&gt;")
                col.markdown(
                    f"<span class='badge {badge}'>{strat}</span>"
                    f"<div style='margin-top:8px;font-size:0.8rem;color:#d1d5db;"
                    f"font-family:JetBrains Mono,monospace;line-height:1.6;"
                    f"background:#0d0f14;padding:10px;border-radius:8px;"
                    f"border:1px solid #1e2130;min-height:80px;'>{safe}</div>",
                    unsafe_allow_html=True,
                )
