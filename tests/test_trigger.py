import math

import pandas as pd
import pytest

from fraud_drift.trigger import classify_psi, retraining_trigger

WIDTH_HOURS = 6.0


def _psi_rows(score_values, feature_value=0.0):
    rows = []
    for i, value in enumerate(score_values):
        start = i * WIDTH_HOURS
        rows.append({"window_start": start, "feature": "score", "psi": value})
        rows.append({"window_start": start, "feature": "V1", "psi": feature_value})
    return pd.DataFrame(rows)


def _rolling(starts, n_frauds, pr_auc):
    return pd.DataFrame(
        {
            "window_start": starts,
            "n_frauds": n_frauds,
            "pr_auc": pr_auc,
        }
    )


def _run(psi_table, rolling_table, reference=0.8):
    return retraining_trigger(psi_table, rolling_table, reference, WIDTH_HOURS)


def test_score_psi_fires_after_two_consecutive_windows():
    psi_table = _psi_rows([0.05, 0.30, 0.30, 0.05])
    rolling = _rolling([0.0, 6.0, 12.0, 18.0], [20, 20, 20, 20], [0.8, 0.8, 0.8, 0.8])
    record = _run(psi_table, rolling)
    assert record["score_psi_fired_at_window_start"] == pytest.approx(12.0)
    assert record["pr_auc_fired_at_window_start"] is None
    assert record["retrain"] is True
    assert record["first_fired_at_window_start"] == pytest.approx(12.0)


def test_single_high_score_psi_window_does_not_trigger():
    psi_table = _psi_rows([0.05, 0.30, 0.05, 0.30, 0.05])
    rolling = _rolling([0.0, 6.0, 12.0, 18.0, 24.0], [20] * 5, [0.8] * 5)
    record = _run(psi_table, rolling)
    assert record["score_psi_fired_at_window_start"] is None
    assert record["retrain"] is False


def test_feature_psi_alone_does_not_trigger():
    psi_table = _psi_rows([0.05, 0.05], feature_value=0.9)
    rolling = _rolling([0.0, 6.0], [20, 20], [0.8, 0.8])
    assert _run(psi_table, rolling)["retrain"] is False


def test_non_adjacent_windows_do_not_count_as_consecutive():
    psi_table = pd.DataFrame(
        {
            "window_start": [0.0, 12.0],
            "feature": ["score", "score"],
            "psi": [0.30, 0.30],
        }
    )
    rolling = _rolling([0.0, 12.0], [20, 20], [0.8, 0.8])
    assert _run(psi_table, rolling)["score_psi_fired_at_window_start"] is None


def test_pr_auc_trigger_ignores_windows_with_fewer_than_five_frauds():
    psi_table = _psi_rows([0.05, 0.05])
    # Both windows drop far below 70% of the reference, but have only 4 frauds.
    rolling = _rolling([0.0, 6.0], [4, 4], [math.nan, math.nan])
    record = _run(psi_table, rolling)
    assert record["pr_auc_fired_at_window_start"] is None
    assert record["retrain"] is False


def test_pr_auc_trigger_counts_windows_with_five_frauds():
    psi_table = _psi_rows([0.05, 0.05])
    rolling = _rolling([0.0, 6.0], [5, 5], [0.5, 0.5])
    record = _run(psi_table, rolling)
    assert record["pr_auc_fired_at_window_start"] == pytest.approx(6.0)
    assert record["retrain"] is True


def test_pr_auc_trigger_needs_drop_over_thirty_percent():
    psi_table = _psi_rows([0.05, 0.05])
    # 0.7 of 0.8 is a 12.5% drop, below the 30% threshold.
    rolling = _rolling([0.0, 6.0], [20, 20], [0.7, 0.7])
    assert _run(psi_table, rolling)["pr_auc_fired_at_window_start"] is None


def test_pr_auc_run_broken_by_ineligible_window():
    psi_table = _psi_rows([0.05, 0.05, 0.05])
    rolling = _rolling([0.0, 6.0, 12.0], [20, 4, 20], [0.4, math.nan, 0.4])
    assert _run(psi_table, rolling)["pr_auc_fired_at_window_start"] is None


def test_retraining_record_contains_thresholds_and_reference():
    record = _run(_psi_rows([0.05]), _rolling([0.0], [20], [0.8]))
    assert record["psi_threshold"] == 0.25
    assert record["pr_auc_drop_threshold"] == 0.30
    assert record["consecutive_windows"] == 2
    assert record["min_frauds"] == 5
    assert record["reference_pr_auc"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.0, "stable"), (0.10, "stable"), (0.20, "warning"), (0.25, "warning"), (0.26, "alert")],
)
def test_psi_status_thresholds(value, status):
    assert classify_psi(value) == status


def test_psi_status_is_undefined_for_nan():
    assert classify_psi(math.nan) == "undefined"
