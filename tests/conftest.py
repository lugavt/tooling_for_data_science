"""Shared fixtures.

sample_population is a small, hand-built DataFrame for fast unit tests
on filtering/scoring logic. Tests that need the real committed sample
data or the real model (test_loading's happy path, test_model) load
data/sample_players.parquet and models/xgb_model.pkl directly -- both
are tiny, committed files, so those tests are just as deterministic and
offline as the ones using sample_population.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tooling_for_data_science.loading import load_players as _load_players
from tooling_for_data_science.model import load_model as _load_model


@pytest.fixture
def sample_population() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_name": "A",
                "year": 2025,
                "club": "X",
                "age": 24,
                "pos_group": "ATT",
                "market_value": 40_000_000,
                "pred_value": 60_000_000,
            },
            {
                "player_id": "2",
                "player_name": "B",
                "year": 2025,
                "club": "X",
                "age": 25,
                "pos_group": "ATT",
                "market_value": 60_000_000,
                "pred_value": 60_000_000,
            },
            {
                "player_id": "3",
                "player_name": "C",
                "year": 2025,
                "club": "Y",
                "age": 26,
                "pos_group": "DEF",
                "market_value": 20_000_000,
                "pred_value": 10_000_000,
            },
            {
                "player_id": "4",
                "player_name": "D",
                "year": 2025,
                "club": "Y",
                "age": 30,
                "pos_group": "ATT",
                "market_value": 0,
                "pred_value": 5_000_000,
            },
        ]
    )


@pytest.fixture(scope="session")
def bundle() -> dict:
    """The real, committed model bundle -- loaded once per test session."""
    return _load_model()


@pytest.fixture(scope="session")
def players() -> pd.DataFrame:
    """The real, committed player sample -- loaded once per test session."""
    return _load_players()
