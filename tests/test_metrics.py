import math

import numpy as np
import pytest

from fraud_drift.metrics import (
    brier,
    confusion_at_threshold,
    pr_auc,
    recall_at_alert_rate,
)


def test_pr_auc_is_one_for_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pr_auc(y, scores) == pytest.approx(1.0)


def test_recall_at_alert_rate_counts_top_alerts():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.05, 0.3, 0.04, 0.03, 0.02])
    # Top 30% of 10 rows = 3 alerts: scores 0.9, 0.8, 0.7 catch 2 of 3 frauds.
    assert recall_at_alert_rate(y, scores, 0.3) == pytest.approx(2 / 3)


def test_recall_at_alert_rate_validates_rate():
    with pytest.raises(ValueError):
        recall_at_alert_rate([0, 1], [0.1, 0.9], 0.0)


def test_confusion_at_threshold_counts_cells():
    counts = confusion_at_threshold([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_brier_score_of_certain_correct_predictions_is_zero():
    assert brier([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert not math.isnan(brier([0, 1], [0.5, 0.5]))
