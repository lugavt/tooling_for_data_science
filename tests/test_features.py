from __future__ import annotations

import pandas as pd
import pytest

from tooling_for_data_science.features import compute_undervalue_score


class TestComputeUndervalueScore:
    def test_known_ratio(self):
        df = pd.DataFrame({"market_value": [50.0], "pred_value": [75.0]})
        score = compute_undervalue_score(df)
        assert score.iloc[0] == pytest.approx(0.5)

    def test_overvalued_is_negative(self):
        df = pd.DataFrame({"market_value": [100.0], "pred_value": [80.0]})
        score = compute_undervalue_score(df)
        assert score.iloc[0] < 0

    def test_zero_market_value_is_nan_not_inf(self):
        df = pd.DataFrame({"market_value": [0.0], "pred_value": [10.0]})
        score = compute_undervalue_score(df)
        assert pd.isna(score.iloc[0])

    def test_preserves_index(self, sample_population):
        score = compute_undervalue_score(sample_population)
        assert list(score.index) == list(sample_population.index)
