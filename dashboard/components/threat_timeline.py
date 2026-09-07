"""Interactive threat-index timeline: a filled area chart with the Layer 3
tier boundaries drawn in as colored bands, so a viewer can see at a glance
how much recent traffic sits in each response tier and how it's trending."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

NORMAL_COLOR = "#22c55e"
ELEVATED_COLOR = "#f59e0b"
CRITICAL_COLOR = "#ef4444"
LINE_COLOR = "#60a5fa"


def render(recent_threat_indices: list[float], tier_normal_max: int, tier_elevated_max: int) -> None:
    st.markdown("#### 📈 Threat Index Timeline")
    if not recent_threat_indices:
        st.info("No requests yet -- send traffic to see live scoring.")
        return

    x = list(range(len(recent_threat_indices)))
    y = recent_threat_indices

    fig = go.Figure()

    # Tier bands as background shading.
    fig.add_hrect(y0=0, y1=tier_normal_max, fillcolor=NORMAL_COLOR, opacity=0.08, line_width=0)
    fig.add_hrect(y0=tier_normal_max, y1=tier_elevated_max, fillcolor=ELEVATED_COLOR, opacity=0.08, line_width=0)
    fig.add_hrect(y0=tier_elevated_max, y1=100, fillcolor=CRITICAL_COLOR, opacity=0.10, line_width=0)

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=LINE_COLOR, width=2, shape="spline", smoothing=0.3),
        fill="tozeroy",
        fillcolor="rgba(96, 165, 250, 0.12)",
        hovertemplate="Request #%{x}<br>Threat Index: %{y:.1f}<extra></extra>",
        name="Threat Index",
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], title="Threat Index", gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(title="Request #", gridcolor="rgba(128,128,128,0.1)"),
        showlegend=False,
        hovermode="x unified",
        font=dict(color="#94a3b8"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"🟢 normal 0-{tier_normal_max}  ·  🟠 elevated {tier_normal_max + 1}-{tier_elevated_max}  ·  "
        f"🔴 critical {tier_elevated_max + 1}-100"
    )
