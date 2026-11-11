"""Fixed-width time windows and the per-window PSI and PR-AUC tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import TIME
from .metrics import pr_auc
from .psi import bin_shares, psi_from_shares, reference_edges
from .trigger import MIN_FRAUDS, classify_psi

DEFAULT_WINDOW_SECONDS = 21600
SECONDS_PER_HOUR = 3600
TARGET = "Class"
SCORE = "score"


def window_hours(window_seconds: float) -> float:
    return float(window_seconds) / SECONDS_PER_HOUR


def assign_window(times, window_seconds: float = DEFAULT_WINDOW_SECONDS, origin: float = 0.0) -> np.ndarray:
    """Start time (seconds) of the fixed-width window containing each timestamp."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    t = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("times contain NaN or infinite values")
    return origin + np.floor((t - origin) / window_seconds) * window_seconds


def psi_table(
    reference: pd.DataFrame,
    monitored: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 10,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    time_col: str = TIME,
) -> pd.DataFrame:
    """PSI per window and feature against the reference. Bin edges come from the reference only.

    Returns rows of window_start (hours since the first transaction), feature, psi, status.
    """
    missing = [c for c in [time_col, *feature_cols] if c not in monitored.columns]
    missing += [c for c in feature_cols if c not in reference.columns]
    if missing:
        raise ValueError(f"Missing columns for PSI: {sorted(set(missing))}")

    edges = {c: reference_edges(reference[c], n_bins) for c in feature_cols}
    expected = {c: bin_shares(reference[c], edges[c]) for c in feature_cols}
    starts = assign_window(monitored[time_col], window_seconds)

    rows = []
    for start in np.unique(starts):
        window = monitored.loc[starts == start]
        for col in feature_cols:
            value = psi_from_shares(expected[col], bin_shares(window[col], edges[col]))
            rows.append(
                {
                    "window_start": start / SECONDS_PER_HOUR,
                    "feature": col,
                    "psi": value,
                    "status": classify_psi(value),
                }
            )
    return pd.DataFrame(rows, columns=["window_start", "feature", "psi", "status"])


def rolling_pr_auc(
    reference_pr_auc: float,
    monitored: pd.DataFrame,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    time_col: str = TIME,
    target: str = TARGET,
    score_col: str = SCORE,
    min_frauds: int = MIN_FRAUDS,
) -> pd.DataFrame:
    """PR-AUC per window. Windows with fewer than `min_frauds` frauds get NaN PR-AUC."""
    if not math.isfinite(reference_pr_auc) or reference_pr_auc <= 0:
        raise ValueError("reference_pr_auc must be a finite positive number")
    starts = assign_window(monitored[time_col], window_seconds)
    rows = []
    for start in np.unique(starts):
        window = monitored.loc[starts == start]
        n_frauds = int(window[target].sum())
        if n_frauds >= min_frauds:
            value = pr_auc(window[target], window[score_col])
            drop = (reference_pr_auc - value) / reference_pr_auc
        else:
            value = math.nan
            drop = math.nan
        rows.append(
            {
                "window_start": start / SECONDS_PER_HOUR,
                "n_rows": len(window),
                "n_frauds": n_frauds,
                "pr_auc": value,
                "reference_pr_auc": float(reference_pr_auc),
                "relative_drop": drop,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["window_start", "n_rows", "n_frauds", "pr_auc", "reference_pr_auc", "relative_drop"],
    )
