"""Loading the player sample data.

This is the "data importing" seam the brief asks to have tests on: one
function that turns the committed parquet file into a validated
DataFrame, failing loudly on a missing file or a malformed schema
instead of letting a bad row surface as a confusing error three layers
up in the Streamlit app or the model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "sample_players.parquet"

REQUIRED_COLUMNS = (
    "player_id",
    "player_name",
    "year",
    "club",
    "age",
    "pos_group",
    "market_value",
    "pred_value",
)


def load_players(path: str | Path = DEFAULT_SAMPLE_PATH) -> pd.DataFrame:
    """Load the player sample and validate its schema.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is missing a column the rest of the app relies on.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no player data file at {path}")

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}")

    return df
