from __future__ import annotations

import pandas as pd
import pytest

from tooling_for_data_science.loading import DEFAULT_SAMPLE_PATH, load_players


class TestLoadPlayers:
    def test_happy_path_loads_committed_sample(self):
        df = load_players()
        assert len(df) > 0
        assert "player_name" in df.columns
        assert "pred_value" in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_players(tmp_path / "does_not_exist.parquet")

    def test_missing_required_column_raises(self, tmp_path):
        bad_path = tmp_path / "bad.parquet"
        pd.DataFrame({"player_id": ["1"], "player_name": ["A"]}).to_parquet(bad_path)
        with pytest.raises(ValueError, match="missing required column"):
            load_players(bad_path)

    def test_default_path_points_at_committed_file(self):
        assert DEFAULT_SAMPLE_PATH.name == "sample_players.parquet"
        assert DEFAULT_SAMPLE_PATH.exists()
