"""
Load trained models and run inference.

The module caches the model and feature list in memory after first load.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from model.features import TARGETS, build_single_row

ARTIFACTS = Path(__file__).parent / "artifacts"
DATA_PATH = Path(__file__).parent.parent / "Textile_Dyeing_Dataset_Causal.csv"


@lru_cache(maxsize=1)
def _load_model():
    model = joblib.load(ARTIFACTS / "xgb_model.pkl")
    with open(ARTIFACTS / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


@lru_cache(maxsize=1)
def _load_recent_data() -> pd.DataFrame:
    """Load historical dataset for lag/rolling proxies."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").tail(90).reset_index(drop=True)
    return df


def predict_single(date_str: str, factory_values: dict) -> dict:
    """
    Predict waste for one day given factory operational parameters.

    Parameters
    ----------
    date_str : 'YYYY-MM-DD'
    factory_values : raw process variables (see API schema)

    Returns
    -------
    dict with predicted kg/L for each waste stream
    """
    model, feature_cols = _load_model()
    recent = _load_recent_data()
    X = build_single_row(date_str, factory_values, recent, feature_cols)
    preds = model.predict(X)[0]
    return {t: round(float(v), 2) for t, v in zip(TARGETS, preds)}


def predict_batch(rows: list[dict]) -> list[dict]:
    """
    Batch prediction: list of {date, ...factory_values} dicts.
    Returns list of prediction dicts in the same order.
    """
    return [predict_single(r.pop("date"), r) for r in rows]

