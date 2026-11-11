"""Evaluation metrics for rare-event fraud classification."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def pr_auc(y_true, scores) -> float:
    """Average precision (PR-AUC), the primary metric for rare fraud."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def recall_at_alert_rate(y_true, scores, alert_rate: float) -> float:
    """Share of frauds caught when the top `alert_rate` fraction of scores is alerted."""
    if not 0 < alert_rate <= 1:
        raise ValueError("alert_rate must be in (0, 1]")
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n_alerts = max(1, math.ceil(alert_rate * len(y)))
    top = np.argsort(-s, kind="mergesort")[:n_alerts]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    return float(y[top].sum() / n_pos)


def confusion_at_threshold(y_true, scores, threshold: float) -> dict[str, int]:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def brier(y_true, probs) -> float:
    return float(brier_score_loss(y_true, probs))


def metric_summary(y_true, scores, alert_rate: float = 0.01) -> dict[str, float]:
    return {
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        f"recall_at_{alert_rate:.2%}_alerts": recall_at_alert_rate(y_true, scores, alert_rate),
    }
