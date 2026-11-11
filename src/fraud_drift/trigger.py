"""Alert rules and the retraining trigger, both pure functions of per-window tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

PSI_WARNING = 0.10
PSI_ALERT = 0.25
PR_AUC_DROP = 0.30
CONSECUTIVE_WINDOWS = 2
MIN_FRAUDS = 5
SCORE_FEATURE = "score"
STABLE = "stable"
WARNING = "warning"
ALERT = "alert"
UNDEFINED = "undefined"


def classify_psi(value: float) -> str:
    """Stable up to 0.10, warning up to 0.25, alert above 0.25."""
    if value is None or not math.isfinite(value):
        return UNDEFINED
    if value > PSI_ALERT:
        return ALERT
    if value > PSI_WARNING:
        return WARNING
    return STABLE


def _consecutive_fire(window_starts, flags, width: float, required: int) -> float | None:
    """Return the start of the window that completes the first run of `required` flagged windows."""
    run = 0
    prev = None
    for start, flag in zip(window_starts, flags):
        if not flag:
            run = 0
        elif run > 0 and prev is not None and math.isclose(start - prev, width, abs_tol=1e-9):
            run += 1
        else:
            run = 1
        prev = start
        if run >= required:
            return float(start)
    return None


def retraining_trigger(
    psi_table: pd.DataFrame,
    rolling_table: pd.DataFrame,
    reference_pr_auc: float,
    window_hours: float,
    psi_threshold: float = PSI_ALERT,
    pr_auc_drop: float = PR_AUC_DROP,
    consecutive: int = CONSECUTIVE_WINDOWS,
    min_frauds: int = MIN_FRAUDS,
) -> dict:
    """Retrain when score PSI or PR-AUC drop fires over consecutive windows.

    Score PSI fires after `consecutive` adjacent windows with PSI above `psi_threshold`.
    PR-AUC fires after `consecutive` adjacent windows whose PR-AUC is at least
    `pr_auc_drop` below the reference. Windows with fewer than `min_frauds` frauds
    (or undefined PR-AUC) are not counted and break a run.
    """
    if not math.isfinite(reference_pr_auc) or reference_pr_auc <= 0:
        raise ValueError("reference_pr_auc must be a finite positive number")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    scores = psi_table.loc[psi_table["feature"] == SCORE_FEATURE].sort_values("window_start")
    score_starts = scores["window_start"].tolist()
    score_flags = [
        bool(math.isfinite(v) and v > psi_threshold) for v in scores["psi"].astype(float)
    ]
    score_fired = _consecutive_fire(score_starts, score_flags, window_hours, consecutive)

    rolling = rolling_table.sort_values("window_start")
    pr_starts = rolling["window_start"].tolist()
    pr_flags = []
    for n_frauds, pr in zip(rolling["n_frauds"], rolling["pr_auc"].astype(float)):
        eligible = int(n_frauds) >= min_frauds and math.isfinite(pr)
        drop = (reference_pr_auc - pr) / reference_pr_auc if eligible else math.nan
        pr_flags.append(bool(eligible and drop > pr_auc_drop))
    pr_fired = _consecutive_fire(pr_starts, pr_flags, window_hours, consecutive)

    fired_at = [w for w in (score_fired, pr_fired) if w is not None]
    return {
        "rule": (
            f"retrain if score PSI > {psi_threshold} in {consecutive} consecutive windows, "
            f"or PR-AUC drops more than {pr_auc_drop:.0%} vs reference in {consecutive} "
            f"consecutive windows with at least {min_frauds} frauds"
        ),
        "window_hours": float(window_hours),
        "psi_threshold": float(psi_threshold),
        "pr_auc_drop_threshold": float(pr_auc_drop),
        "consecutive_windows": int(consecutive),
        "min_frauds": int(min_frauds),
        "reference_pr_auc": float(reference_pr_auc),
        "n_monitored_windows": len(pr_starts),
        "score_psi_fired_at_window_start": score_fired,
        "pr_auc_fired_at_window_start": pr_fired,
        "retrain": bool(fired_at),
        "first_fired_at_window_start": min(fired_at) if fired_at else None,
    }


def write_decision_record(record: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out
