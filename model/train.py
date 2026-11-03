"""
Train the XGBoost waste-prediction models.

Usage:
    python -m model.train
    python -m model.train --data path/to/dataset.xlsx

Outputs (written to model/artifacts/):
    xgb_model.pkl            — trained PerTargetXGB
    feature_cols.json         — ordered feature list
    best_params.json          — per-target hyperparameters
    metrics.json              — test-set metrics for all 4 targets
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from model.features import TARGETS, engineer

warnings.filterwarnings("ignore")

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

DEFAULT_DATA = Path(__file__).parent.parent / "Textile_Dyeing_Dataset_Causal.xlsx"

PARAM_GRID = {
    "max_depth":        [2, 3, 4],
    "learning_rate":    [0.03, 0.05, 0.1],
    "n_estimators":     [400, 700],
    "min_child_weight": [3, 5, 10],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda":       [1.0, 5.0],
}


class PerTargetXGB:
    """One XGBRegressor per target, each with its own best hyperparameters."""

    def __init__(self, params_per_target: dict, targets: list[str]):
        self.params_per_target = params_per_target
        self.targets = targets
        self.estimators_: list[XGBRegressor] = []

    def fit(self, X, Y):
        self.estimators_ = []
        for t in self.targets:
            est = XGBRegressor(
                **self.params_per_target[t],
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            est.fit(X, Y[t])
            self.estimators_.append(est)
        return self

    def predict(self, X) -> np.ndarray:
        return np.column_stack([est.predict(X) for est in self.estimators_])


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def tune_target(X_train, y_series, tscv) -> dict:
    gs = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_series)
    return gs.best_params_


def evaluate(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4), "mape_pct": round(mape, 2)}


def train(data_path: Path = DEFAULT_DATA):
    print(f"Loading data from {data_path}")
    df_raw = load_data(data_path)

    print("Engineering features...")
    df, feature_cols = engineer(df_raw)

    df_train = df[df["Date"].dt.year == 2023].copy()
    df_test  = df[df["Date"].dt.year == 2024].copy()

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train[TARGETS].reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)
    y_test  = df_test[TARGETS].reset_index(drop=True)

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    tscv = TimeSeriesSplit(n_splits=3, gap=7)

    print("Tuning hyperparameters per target (this takes a few minutes)...")
    best_params = {}
    for target in TARGETS:
        print(f"  -> {target}")
        best_params[target] = tune_target(X_train, y_train[target], tscv)
        print(f"     {best_params[target]}")

    print("Training final model...")
    model = PerTargetXGB(best_params, TARGETS)
    model.fit(X_train, y_train)

    y_pred = pd.DataFrame(model.predict(X_test), columns=TARGETS)

    metrics = {}
    print("\nTest-set performance (2024):")
    for t in TARGETS:
        m = evaluate(y_test[t], y_pred[t])
        metrics[t] = m
        print(f"  {t:<25}  R²={m['r2']:.4f}  MAE={m['mae']:.1f}  MAPE={m['mape_pct']:.1f}%")

    # Save artifacts
    model_path = ARTIFACTS / "xgb_model.pkl"
    joblib.dump(model, model_path)

    fc_path = ARTIFACTS / "feature_cols.json"
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    bp_path = ARTIFACTS / "best_params.json"
    with open(bp_path, "w") as f:
        json.dump(best_params, f, indent=2)

    mt_path = ARTIFACTS / "metrics.json"
    with open(mt_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS}/")
    print(f"  {model_path.name}")
    print(f"  {fc_path.name}")
    print(f"  {bp_path.name}")
    print(f"  {mt_path.name}")

    return model, feature_cols, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    train(args.data)

