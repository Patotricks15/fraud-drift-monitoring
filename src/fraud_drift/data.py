"""Load the card fraud dataset and split it by time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KAGGLE_INPUT_PATH = Path("/kaggle/input/creditcardfraud/creditcard.csv")
DATA_PATH_ENV = "FRAUD_DATA_PATH"
TARGET = "Class"
TIME = "Time"
AMOUNT = "Amount"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = {TIME, AMOUNT, TARGET, *PCA_FEATURES}


def resolve_data_path() -> Path:
    """Return the first existing dataset path: env override, then Kaggle input."""
    candidates = []
    env_path = os.environ.get(DATA_PATH_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(KAGGLE_INPUT_PATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "creditcard.csv not found. Attach mlg-ulb/creditcardfraud in a Kaggle "
        f"Notebook or set {DATA_PATH_ENV}."
    )


def load_creditcard(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset and sort it by transaction time."""
    data_path = Path(path) if path is not None else resolve_data_path()
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df.sort_values(TIME, kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class TimeSplit:
    """Boolean masks for a time-ordered train, validation and test split."""

    train: pd.Series
    val: pd.Series
    test: pd.Series
    train_end: float
    val_end: float


def time_split(
    df: pd.DataFrame,
    time_col: str = TIME,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TimeSplit:
    """Split by time quantiles: earliest 70% of time, next 15%, last 15%."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum below 1")
    times = df[time_col]
    train_end = float(times.quantile(train_frac))
    val_end = float(times.quantile(train_frac + val_frac))
    train = times <= train_end
    val = (times > train_end) & (times <= val_end)
    test = times > val_end
    for name, mask in (("train", train), ("val", val), ("test", test)):
        if not mask.any():
            raise ValueError(f"The {name} split is empty; check the time column")
    return TimeSplit(train=train, val=val, test=test, train_end=train_end, val_end=val_end)
