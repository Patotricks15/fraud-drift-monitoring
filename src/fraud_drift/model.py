"""Logistic regression reference model with standardized inputs and balanced class weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import AMOUNT, PCA_FEATURES, TARGET

RANDOM_STATE = 42
C_REFERENCE = 1.0
MODEL_VERSION = "logistic-C1.0-balanced"
FEATURES = [*PCA_FEATURES, "log_amount"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out[AMOUNT])
    return out


def build_logistic(C: float = C_REFERENCE) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_reference(df: pd.DataFrame, train_mask, C: float = C_REFERENCE) -> Pipeline:
    """Fit the reference model on the rows selected by `train_mask` only."""
    train = df.loc[np.asarray(train_mask, dtype=bool)]
    if train.empty:
        raise ValueError("Reference training window is empty")
    if train[TARGET].nunique() < 2:
        raise ValueError("Reference training window needs both classes")
    return build_logistic(C).fit(train[FEATURES], train[TARGET])


def score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predicted fraud probability for every row of `df`."""
    return model.predict_proba(df[FEATURES])[:, 1]
