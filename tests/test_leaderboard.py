from __future__ import annotations

from tooling_for_data_science.features import compute_undervalue_score
from tooling_for_data_science.leaderboard import filter_by_position, top_undervalued


class TestFilterByPosition:
    def test_happy_path(self, sample_population):
        result = filter_by_position(sample_population, "ATT")
        assert set(result["player_name"]) == {"A", "B", "D"}

    def test_unknown_position_returns_empty(self, sample_population):
        result = filter_by_position(sample_population, "GK")
        assert result.empty

    def test_does_not_mutate_input(self, sample_population):
        before = sample_population.copy()
        filter_by_position(sample_population, "ATT")
        pd_testing_equal = sample_population.equals(before)
        assert pd_testing_equal


class TestTopUndervalued:
    def _scored(self, df):
        df = df.copy()
        df["undervalue_score"] = compute_undervalue_score(df)
        return df

    def test_sorted_descending(self, sample_population):
        scored = self._scored(sample_population)
        result = top_undervalued(scored, limit=10)
        scores = result["undervalue_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_respects_limit(self, sample_population):
        scored = self._scored(sample_population)
        result = top_undervalued(scored, limit=2)
        assert len(result) == 2

    def test_respects_position_filter(self, sample_population):
        scored = self._scored(sample_population)
        result = top_undervalued(scored, position_group="DEF", limit=10)
        assert set(result["player_name"]) == {"C"}

    def test_drops_nan_scores(self, sample_population):
        # player D has market_value == 0 -> NaN undervalue_score -> excluded
        scored = self._scored(sample_population)
        result = top_undervalued(scored, limit=10)
        assert "D" not in result["player_name"].tolist()
