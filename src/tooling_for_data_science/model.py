"""Loading the trained model bundle and getting predictions / explanations.

The pickle is a bundle -- {"model": XGBRegressor, "features": [...]} --
not a bare model. The feature list is read from the bundle itself, never
hardcoded, so a retrained model with a different feature set can never
silently go out of sync with the code that calls it.

The model was trained on log(market_value), so every prediction is
back-transformed with np.exp().
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load the trained model bundle: {"model": ..., "features": [...]}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model file at {path}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError(f"{path.name}: expected a dict with 'model' and 'features' keys")

    return bundle


def predict(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    """Predicted market values in EUR (back-transformed from log-space)."""
    return np.exp(bundle["model"].predict(x[bundle["features"]]))


def explain(bundle: dict, x: pd.DataFrame) -> dict:
    """SHAP explanation for a single player row (``x`` has exactly one row).

    All SHAP values are in log-space. baseline_log + sum(shap_values)
    equals pred_log exactly; only the final sum is back-transformed to
    EUR, never an individual feature's SHAP value.
    """
    features = bundle["features"]
    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer.shap_values(x[features])
    return {
        "baseline_log": float(explainer.expected_value),
        "shap_values": dict(zip(features, sv[0], strict=True)),
        "pred_log": float(explainer.expected_value + sv[0].sum()),
        "pred_eur": float(np.exp(explainer.expected_value + sv[0].sum())),
    }
