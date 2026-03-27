"""
ui/tab_analytics.py
───────────────────
Renders the "Analytics" tab: response-length bar chart, per-row
consistency line chart, robustness flags, and best/worst strategy summary.
"""

import streamlit as st
import pandas as pd
import altair as alt

from config import STRATEGIES, STRATEGY_COLORS
from analytics import rank_strategies


def render_tab_analytics() -> None:
    """Render the full Analytics tab content."""

    if "results" not in st.session_state:
        st.markdown(
            "<div class='lab-alert'>Run the comparison first to see analytics.</div>",
            unsafe_allow_html=True,
        )
        return

    r = st.session_state["results"]

    # ── 1. Average response-length bar chart ───────────────────────────────────
    st.markdown(
        "<div class='section-header'>Average Response Length (words) by Strategy</div>",
        unsafe_allow_html=True,
    )

    length_df = pd.DataFrame(
        [
            {"Strategy": s, "Avg Words": r["lengths"][s], "Color": STRATEGY_COLORS[s]}
            for s in STRATEGIES
        ]
    )

    bar_chart = (
        alt.Chart(length_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Strategy:N", axis=alt.Axis(labelColor="#9ca3af", titleColor="#4b5563")),
            y=alt.Y("Avg Words:Q", axis=alt.Axis(labelColor="#9ca3af", titleColor="#4b5563")),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Strategy", alt.Tooltip("Avg Words:Q", format=".1f")],
        )
        .properties(height=260, background="#161920")
        .configure_view(stroke=None)
        .configure_axis(gridColor="#1e2130", domainColor="#2a2d38")
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # ── 2. Row-by-row consistency line chart ───────────────────────────────────
    st.markdown(
        "<div class='section-header'>Row-by-Row Consistency Scores</div>",
        unsafe_allow_html=True,
    )

    score_df = pd.DataFrame(
        [{"Input": f"Input {i + 1}", "Score": r["scores"][i]} for i in range(len(r["scores"]))]
    )

    line_chart = (
        alt.Chart(score_df)
        .mark_line(point=True, color="#1e8f5e", strokeWidth=2)
        .encode(
            x=alt.X("Input:N", axis=alt.Axis(labelColor="#9ca3af", titleColor="#4b5563")),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelColor="#9ca3af", titleColor="#4b5563", format="%"),
            ),
            tooltip=["Input", alt.Tooltip("Score:Q", format=".0%")],
        )
        .properties(height=220, background="#161920")
        .configure_view(stroke=None)
        .configure_axis(gridColor="#1e2130", domainColor="#2a2d38")
        .configure_point(color="#1e8f5e", size=80)
    )
    st.altair_chart(line_chart, use_container_width=True)

    # ── 3. Robustness flags ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Robustness Flags</div>", unsafe_allow_html=True)

    any_outlier = False
    for idx, outlier in enumerate(r["outliers"]):
        if outlier:
            any_outlier = True
            inp = r["inputs"][idx]
            st.markdown(
                f"<div class='lab-danger'>"
                f"⚠️ <b>Input {idx + 1}</b> (\"{inp[:50]}…\") — "
                f"<b>{outlier}</b> gave a significantly different response than the others. "
                f"This may indicate the strategy is less robust for this type of input."
                f"</div>",
                unsafe_allow_html=True,
            )

    if not any_outlier:
        st.markdown(
            "<div class='lab-alert'>"
            "✅ No significant outliers detected. All strategies responded with reasonable consistency."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 4. Best / worst strategy summary ──────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Strategy Performance Summary</div>",
        unsafe_allow_html=True,
    )

    best, worst = rank_strategies(r)

    s1, s2 = st.columns(2)
    s1.markdown(
        f"<div class='metric-card' style='border:1px solid #1e8f5e;'>"
        f"<div class='label' style='color:#1e8f5e;'>🏆 Best Performing</div>"
        f"<div style='font-size:1.3rem;font-weight:800;color:#e8eaf0;"
        f"font-family:Syne,sans-serif;'>{best}</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;margin-top:6px;'>"
        f"Highest detail with fewest outlier flags</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    s2.markdown(
        f"<div class='metric-card' style='border:1px solid #ef4444;'>"
        f"<div class='label' style='color:#ef4444;'>⚠️ Most Variable</div>"
        f"<div style='font-size:1.3rem;font-weight:800;color:#e8eaf0;"
        f"font-family:Syne,sans-serif;'>{worst}</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;margin-top:6px;'>"
        f"More likely to produce inconsistent results</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
