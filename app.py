"""Transfer Edge -- La Liga player valuation with explainable predictions.

Two tabs: look a player up and see why the model prices him where it does,
or browse the most undervalued players by position.

All the logic lives in src/tooling_for_data_science/ and is unit-tested;
this file is presentation only.
"""

from __future__ import annotations

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tooling_for_data_science.features import compute_undervalue_score
from tooling_for_data_science.leaderboard import top_undervalued
from tooling_for_data_science.loading import load_players
from tooling_for_data_science.model import explain, load_model, predict

# Test-set MAPE of the trained model. Used as an approximate band around the
# point estimate, and as the cutoff for the over-/undervalued verdict below --
# NOT a statistically derived confidence interval.
MAPE = 0.279

TOP_N_PER_SIDE = 3

# Diverging pair (blue <-> red) plus one categorical slot for the market-value
# reference. Blue/red is colourblind-safe; green/red is not.
COLOR_UP = "#2a78d6"
COLOR_DOWN = "#e34948"
COLOR_MARKET = "#eb6834"
COLOR_MUTED = "#52514e"

POSITION_LABELS = {
    "GK": "Goalkeeper",
    "DEF": "Defender",
    "DM": "Defensive midfield",
    "MID": "Midfield",
    "ATT": "Attack",
}

# Plain-language definitions for engineered feature names that show up in the
# "Why this valuation" breakdown but aren't self-explanatory on their own.
FEATURE_GLOSSARY = {
    "value_growth": (
        "Year-over-year change in market value (log-space), vs. the player's "
        "value the previous season."
    ),
    "years_to_peak": (
        "Years until the player reaches the typical peak age for his position "
        "(e.g. 26 for attackers, 30 for goalkeepers). Negative means past peak."
    ),
    "club_med_value": (
        "Median market value of all players at the player's club that season -- "
        "a proxy for the club's prestige/visibility."
    ),
    "log_value_lag1": (
        "The player's market value from the previous season (log-space), "
        "when available -- often the single strongest predictor."
    ),
    "minutes_pct": (
        "Percentile rank of minutes played, compared to other players in the "
        "same position and season (0 = fewest, 1 = most)."
    ),
    "natgrp_Spain": (
        "Whether the player's nationality group is Spain (1) or not (0), one "
        "of the model's nationality indicator features."
    ),
}

FAVICON = Path(__file__).parent / "assets" / "favicon.png"

st.set_page_config(page_title="Transfer Edge", page_icon=str(FAVICON), layout="wide")


@st.cache_resource
def get_model() -> dict:
    return load_model()


@st.cache_data
def get_players() -> pd.DataFrame:
    df = load_players().reset_index(drop=True)
    df["undervalue_score"] = compute_undervalue_score(df)
    return df


def format_eur(value: float | None) -> str:
    """Compact euro formatting: 45_000_000 -> 'EUR 45.0M'."""
    if value is None or pd.isna(value):
        return "unknown"
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:.0f}"


def format_height(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "height unknown"
    return f"{value:.2f} m" if value < 3 else f"{value:.0f} cm"


def valuation_range_chart(predicted: float, market: float) -> alt.LayerChart:
    """Point estimate with its MAPE band, against the listed market value."""
    low, high = predicted * (1 - MAPE), predicted * (1 + MAPE)
    band = pd.DataFrame([{"low": low, "high": high, "predicted": predicted, "market": market}])

    lo, hi = min(low, market), max(high, market)
    pad = max((hi - lo) * 0.22, hi * 0.05)
    # clamp at zero -- a negative euro axis reads as a bug
    scale = alt.Scale(domain=[max(0.0, lo - pad), hi + pad])
    axis = alt.Axis(format="~s", title="Value (EUR)", grid=False, tickCount=5)

    whisker = (
        alt.Chart(band)
        .mark_rule(size=10, color=COLOR_UP, opacity=0.25)
        .encode(
            x=alt.X("low:Q", scale=scale, axis=axis),
            x2="high:Q",
        )
    )
    point = (
        alt.Chart(band)
        .mark_point(size=260, filled=True, color=COLOR_UP)
        .encode(
            x=alt.X("predicted:Q", scale=scale, axis=axis),
            tooltip=[alt.Tooltip("predicted:Q", title="Predicted", format=",.0f")],
        )
    )
    market_rule = (
        alt.Chart(band)
        .mark_rule(size=3, color=COLOR_MARKET, strokeDash=[6, 4])
        .encode(
            x=alt.X("market:Q", scale=scale, axis=axis),
            tooltip=[alt.Tooltip("market:Q", title="Market value", format=",.0f")],
        )
    )
    predicted_label = (
        alt.Chart(band)
        .mark_text(dy=-28, color=COLOR_UP, fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("predicted:Q", scale=scale, axis=axis),
            text=alt.value(f"Model: {format_eur(predicted)}"),
        )
    )
    market_label = (
        alt.Chart(band)
        .mark_text(dy=28, color=COLOR_MARKET, fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("market:Q", scale=scale, axis=axis),
            text=alt.value(f"Listed: {format_eur(market)}"),
        )
    )

    return (whisker + market_rule + point + predicted_label + market_label).properties(
        height=170,
        padding={"top": 30, "bottom": 10, "left": 10, "right": 10},
    )


with st.spinner("The data is being loaded, please wait a couple seconds"):
    players = get_players()
    bundle = get_model()

st.title("La Liga players valuation")
st.caption(
    "Check La Liga players' market value and compare with what we believe should "
    "be their actual value. Explore the features that push or drag them. Jump to "
    "the leaderboard to see some market gems."
)

search_tab, leaderboard_tab = st.tabs(["Player search", "Leaderboard"])

with search_tab:
    club_series = players["club"].fillna("Unknown club")
    club_col, player_col = st.columns(2)
    selected_club = club_col.selectbox("Club", sorted(club_series.unique()), index=0)

    club_players = players[club_series == selected_club].sort_values("player_name")
    player_names = club_players["player_name"].fillna("Unknown").tolist()
    selected_player = player_col.selectbox("Player", player_names, index=0)

    row = club_players[club_players["player_name"].fillna("Unknown") == selected_player].iloc[[0]]
    player = row.iloc[0]

    predicted = float(predict(bundle, row)[0])
    market = float(player["market_value"])
    score = player["undervalue_score"]

    meta_parts = [
        str(player.get("club") or ""),
        str(player.get("position") or ""),
        f"{player['age']:.0f} years old" if pd.notna(player["age"]) else "",
        format_height(player.get("height")),
        str(player.get("nationality") or ""),
    ]
    meta = " &nbsp;·&nbsp; ".join(html.escape(part) for part in meta_parts if part)

    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.6rem 0 0.4rem 0;">
            <div style="font-size:2.9rem; font-weight:700; line-height:1.15;">
                {html.escape(str(player["player_name"]))}
            </div>
            <div style="font-size:1.1rem; color:{COLOR_MUTED}; padding-top:0.6rem;">
                {meta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if pd.isna(score):
        st.info("No market value on record, so there is nothing to compare against.")
    elif score > MAPE:
        st.success(
            f"**Undervalued.** The model prices him {score * 100:.0f}% above his "
            "listed market value."
        )
    elif score < -MAPE:
        st.warning(
            f"**Overvalued.** The model prices him {abs(score) * 100:.0f}% below his "
            "listed market value."
        )
    else:
        st.info(
            f"**Fairly valued.** The model lands within its {MAPE:.1%} MAPE of the listed value."
        )

    left, middle, right = st.columns(3)
    left.metric("Model valuation", format_eur(predicted))
    middle.metric("Listed market value", format_eur(market))
    right.metric(
        "Undervalue score",
        "n/a" if pd.isna(score) else f"{score * 100:+.0f}%",
    )

    st.altair_chart(valuation_range_chart(predicted, market), width="stretch")

    st.caption(
        f"The blue band is the model's valuation give or take {MAPE:.1%} -- its "
        f"**MAPE** (mean absolute percentage error), meaning that on players it had "
        f"never seen during training, its estimate was off by {MAPE:.1%} on average. "
        "It is a rough sense of typical error, not a fitted confidence interval. "
        "The orange dashed line is the value actually listed on Transfermarkt: when "
        "it falls outside the band, the model and the market genuinely disagree."
    )

    st.subheader("Why this valuation")

    result = explain(bundle, row)
    shap_df = pd.DataFrame(
        {
            "feature": list(result["shap_values"].keys()),
            "impact": list(result["shap_values"].values()),
        }
    )
    positive = (
        shap_df[shap_df["impact"] > 0].sort_values("impact", ascending=False).head(TOP_N_PER_SIDE)
    )
    negative = shap_df[shap_df["impact"] < 0].sort_values("impact").head(TOP_N_PER_SIDE)

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Positive attributes** _(top 3)_")
        if positive.empty:
            st.caption("None among the top features.")
        else:
            for _, feat in positive.iterrows():
                st.markdown(f"{feat['feature']} ({feat['impact']:+.3f})")
    with neg_col:
        st.markdown("**Negative attributes** _(top 3)_")
        if negative.empty:
            st.caption("None among the top features.")
        else:
            for _, feat in negative.iterrows():
                st.markdown(f"{feat['feature']} ({feat['impact']:+.3f})")

    st.caption(
        "Values are in log-space, where they sum exactly to the prediction -- "
        "individual numbers are not euro amounts."
    )

    with st.expander("What do these features mean?"):
        for feature, definition in FEATURE_GLOSSARY.items():
            st.markdown(f"**{feature}** -- {definition}")

with leaderboard_tab:
    st.subheader("Most undervalued players")

    controls_left, controls_right = st.columns([2, 3])
    position = controls_left.selectbox(
        "Position",
        ["All positions", *POSITION_LABELS],
        format_func=lambda key: POSITION_LABELS.get(key, key),
    )
    count = controls_right.slider("How many to show", 5, 50, 20)

    table = top_undervalued(
        players,
        position_group=None if position == "All positions" else position,
        limit=count,
    )
    table = table.assign(undervalue_pct=table["undervalue_score"] * 100)

    st.dataframe(
        table[
            [
                "player_name",
                "club",
                "age",
                "pos_group",
                "market_value",
                "pred_value",
                "undervalue_pct",
            ]
        ],
        column_config={
            "player_name": "Player",
            "club": "Club",
            "age": st.column_config.NumberColumn("Age", format="%.0f"),
            "pos_group": "Position",
            "market_value": st.column_config.NumberColumn("Market value", format="EUR %.0f"),
            "pred_value": st.column_config.NumberColumn("Predicted", format="EUR %.0f"),
            "undervalue_pct": st.column_config.NumberColumn("Undervalued by", format="%+.0f%%"),
        },
        hide_index=True,
        width="stretch",
    )

    st.caption(
        f"Ranked by (predicted - market) / market across {len(players)} players "
        "from the 2025 season."
    )
