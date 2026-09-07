"""Two complementary views of recent traffic: a donut chart of the overall
tier mix, and a scatter of individual recent requests (threat index over
time, colored by tier, watermark triggers marked distinctly) -- the donut
answers "what's the mix right now", the scatter answers "what happened, in
what order, and which requests got watermarked"."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

TIER_COLORS = {"normal": "#22c55e", "elevated": "#f59e0b", "critical": "#ef4444"}


def render_donut(tier_counts: dict[str, int]) -> None:
    st.markdown("#### 🍩 Response Tier Mix")
    total = sum(tier_counts.values())
    if total == 0:
        st.info("No requests yet.")
        return

    labels = list(tier_counts.keys())
    values = [tier_counts[label] for label in labels]
    colors = [TIER_COLORS.get(label, "#64748b") for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} requests (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[dict(text=f"{total}<br>total", x=0.5, y=0.5, font_size=18, showarrow=False, font=dict(color="#94a3b8"))],
        font=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_scatter(recent_events: list[dict]) -> None:
    st.markdown("#### 🔎 Recent Requests")
    if not recent_events:
        st.info("No requests yet.")
        return

    events = list(reversed(recent_events))  # oldest first for a left-to-right timeline
    x = list(range(len(events)))
    y = [e["threat_index"] for e in events]
    tiers = [e["tier"] for e in events]
    colors = [TIER_COLORS.get(t, "#64748b") for t in tiers]
    symbols = ["star" if e.get("watermarked") else "circle" for e in events]
    sizes = [14 if e.get("watermarked") else 8 for e in events]
    hover = [
        f"client: {e['client_id']}<br>tier: {e['tier']}<br>threat: {e['threat_index']:.1f}"
        + ("<br>⭐ watermarked" if e.get("watermarked") else "")
        for e in events
    ]

    fig = go.Figure(data=[go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color="rgba(255,255,255,0.3)")),
        hovertext=hover,
        hoverinfo="text",
    )])
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], title="Threat Index", gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(title="Most recent requests (left to right)", gridcolor="rgba(128,128,128,0.1)"),
        font=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("⭐ star marker = watermark triggered on that request")
