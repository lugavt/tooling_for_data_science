"""Feature constants and pure feature computations.

PEAK_AGES is ported from the original Transfer Edge model notebook.
FEATURE_COLS documents the feature set for reference; model.py reads
the authoritative list from the model bundle itself at predict time,
so this constant is never used to select columns for the real model.
"""

from __future__ import annotations

import pandas as pd

PEAK_AGES = {"GK": 30, "DEF": 28, "DM": 28, "MID": 27, "ATT": 26}

FEATURE_COLS = [
    # Personal
    "age",
    "age2",
    "height",
    "foot_enc",
    # Per-90 stats
    "minutes",
    "games",
    "goals_p90",
    "xG_p90",
    "assists_p90",
    "xA_p90",
    "shots_p90",
    "key_passes_p90",
    "npxG_p90",
    "xGChain_p90",
    "xGBuildup_p90",
    # Engineered
    "goals_minus_xG",
    "assists_minus_xA",
    "xG_p90_pct",
    "xA_p90_pct",
    "npxG_p90_pct",
    "minutes_pct",
    "years_to_peak",
    "prime_window",
    "club_med_value",
    "club_tier",
    # Lag features (NaN when year_diff != 1)
    "log_value_lag1",
    "xG_p90_lag1",
    "goals_p90_lag1",
    "assists_p90_lag1",
    "minutes_lag1",
    "value_growth",
    # Position one-hot (ATT dropped as reference)
    "posgrp_DEF",
    "posgrp_DM",
    "posgrp_GK",
    "posgrp_MID",
    # Nationality one-hot (kept as trained -- not actually dropped)
    "natgrp_Brazil",
    "natgrp_Colombia",
    "natgrp_France",
    "natgrp_Morocco",
    "natgrp_Other",
    "natgrp_Portugal",
    "natgrp_Senegal",
    "natgrp_Serbia",
    "natgrp_Spain",
    "natgrp_Uruguay",
]


def compute_undervalue_score(df: pd.DataFrame) -> pd.Series:
    """(predicted - market) / market. Positive means undervalued.

    A zero or missing market value makes the ratio meaningless rather
    than infinite, so those rows come back as NaN instead of +/-inf.
    """
    market = df["market_value"].replace(0, pd.NA)
    return (df["pred_value"] - market) / market
