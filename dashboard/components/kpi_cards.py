"""Custom-styled KPI cards. Streamlit's built-in st.metric is serviceable but
generic -- these render as HTML/CSS "glass" cards with an icon, a big number,
and a colored accent bar, which reads as a purpose-built security console
rather than a default widget grid."""
from __future__ import annotations

import streamlit as st

CARD_TEMPLATE = """
<div class="mv-kpi" style="--accent: {accent};">
    <div class="mv-kpi-icon">{icon}</div>
    <div class="mv-kpi-value">{value}</div>
    <div class="mv-kpi-label">{label}</div>
</div>
"""


def render(cards: list[dict]) -> None:
    """cards: list of {icon, value, label, accent} dicts."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                CARD_TEMPLATE.format(
                    icon=card["icon"],
                    value=card["value"],
                    label=card["label"],
                    accent=card.get("accent", "#6366f1"),
                ),
                unsafe_allow_html=True,
            )
