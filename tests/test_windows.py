import numpy as np
import pandas as pd
import pytest

from fraud_drift.windows import assign_window, psi_table, rolling_pr_auc


def test_window_assignment_uses_half_open_intervals():
    starts = assign_window([0, 21599, 21600, 43199, 43200], window_seconds=21600)
    np.testing.assert_allclose(starts, [0, 0, 21600, 21600, 43200])


def test_window_assignment_rejects_non_positive_width():
    with pytest.raises(ValueError):
        assign_window([0.0], window_seconds=0)


def test_psi_table_has_one_row_per_window_and_feature():
    reference = pd.DataFrame({"V1": [0.0, 1.0, 2.0, 3.0], "score": [0.1, 0.2, 0.3, 0.4]})
    monitored = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 21600.0],
            "V1": [0.0, 1.0, 2.0],
            "score": [0.1, 0.2, 0.3],
        }
    )
    table = psi_table(reference, monitored, ["V1", "score"], n_bins=2, window_seconds=21600)
    assert list(table.columns) == ["window_start", "feature", "psi", "status"]
    assert len(table) == 2 * 2
    assert sorted(table["window_start"].unique()) == [0.0, 6.0]


def test_rolling_pr_auc_is_nan_for_windows_with_few_frauds():
    rng = np.random.default_rng(3)
    n = 40
    monitored = pd.DataFrame(
        {
            "Time": np.linspace(0, 21599, n),
            "Class": [1] * 4 + [0] * (n - 4),
            "score": rng.uniform(size=n),
        }
    )
    table = rolling_pr_auc(0.8, monitored, window_seconds=21600)
    assert len(table) == 1
    assert table.loc[0, "n_frauds"] == 4
    assert np.isnan(table.loc[0, "pr_auc"])
    assert np.isnan(table.loc[0, "relative_drop"])
