"""Pure filtering functions over the player sample.

Each function takes a DataFrame and returns a new one -- the input is
never mutated -- so they are directly unit-testable with a small
in-memory DataFrame and no fixture files.
"""

from __future__ import annotations

import pandas as pd


def filter_by_position(df: pd.DataFrame, position_group: str) -> pd.DataFrame:
    """Rows matching a position group (GK, DEF, DM, MID, ATT)."""
    return df[df["pos_group"] == position_group].reset_index(drop=True)


def top_undervalued(
    df: pd.DataFrame,
    position_group: str | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """The most undervalued players, optionally restricted to one position.

    ``df`` must already have an ``undervalue_score`` column (see
    :func:`tooling_for_data_science.features.compute_undervalue_score`).
    Rows with a NaN score (e.g. a zero market value) are dropped before
    ranking, since they cannot be meaningfully compared.
    """
    result = df if position_group is None else filter_by_position(df, position_group)
    result = result.dropna(subset=["undervalue_score"])
    return (
        result.sort_values("undervalue_score", ascending=False).head(limit).reset_index(drop=True)
    )
