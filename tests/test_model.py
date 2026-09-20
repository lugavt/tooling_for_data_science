from __future__ import annotations

import pytest

from tooling_for_data_science.model import DEFAULT_MODEL_PATH, explain, load_model, predict


class TestLoadModel:
    def test_happy_path(self):
        bundle = load_model()
        assert "model" in bundle
        assert "features" in bundle
        assert len(bundle["features"]) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "does_not_exist.pkl")

    def test_malformed_pickle_raises(self, tmp_path):
        import pickle

        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as f:
            pickle.dump({"not_a_model": 1}, f)
        with pytest.raises(ValueError, match="model"):
            load_model(bad_path)

    def test_default_path_points_at_committed_file(self):
        assert DEFAULT_MODEL_PATH.name == "xgb_model.pkl"
        assert DEFAULT_MODEL_PATH.exists()


class TestPredictAndExplain:
    def test_predict_matches_stored_predictions(self, bundle, players):
        preds = predict(bundle, players)
        assert len(preds) == len(players)
        stored = players["pred_value"].to_numpy()
        assert preds == pytest.approx(stored, rel=1e-4)

    def test_explain_satisfies_shap_guarantee(self, bundle, players):
        row = players.iloc[[0]]
        result = explain(bundle, row)
        baseline_plus_shap = result["baseline_log"] + sum(result["shap_values"].values())
        assert baseline_plus_shap == pytest.approx(result["pred_log"], abs=1e-6)

    def test_explain_matches_predict(self, bundle, players):
        row = players.iloc[[0]]
        result = explain(bundle, row)
        direct_pred = predict(bundle, row)[0]
        assert result["pred_eur"] == pytest.approx(direct_pred, rel=1e-3)
