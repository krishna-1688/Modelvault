"""A prominent speedometer-style gauge showing the current threat level --
the single most recognizable "security console" visual, meant to be the
first thing a viewer's eye lands on. Shows the average of the last few
requests (smoother than the single latest value, still responsive)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render(recent_threat_indices: list[float], tier_normal_max: int, tier_elevated_max: int) -> None:
    if not recent_threat_indices:
        current = 0.0
    else:
        window = recent_threat_indices[-10:]
        current = float(np.mean(window))

    if current <= tier_normal_max:
        bar_color = "#22c55e"
    elif current <= tier_elevated_max:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current,
        number={"suffix": "", "font": {"size": 42, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#64748b"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, tier_normal_max], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [tier_normal_max, tier_elevated_max], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [tier_elevated_max, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": current,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
