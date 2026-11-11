import numpy as np
import pandas as pd
import pytest

from fraud_drift.data import TIME, time_split


def _frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({TIME: np.sort(rng.uniform(0, 172_800, n))})


def test_split_is_time_ordered_and_disjoint():
    df = _frame()
    split = time_split(df)
    assert not (split.train & split.val).any()
    assert not (split.val & split.test).any()
    assert not (split.train & split.test).any()
    assert (split.train | split.val | split.test).all()
    assert df.loc[split.train, TIME].max() <= df.loc[split.val, TIME].min()
    assert df.loc[split.val, TIME].max() <= df.loc[split.test, TIME].min()


def test_split_fractions_follow_time_not_row_shuffle():
    df = _frame(2000)
    split = time_split(df)
    assert split.train.sum() / len(df) == pytest.approx(0.70, abs=0.01)
    assert split.val.sum() / len(df) == pytest.approx(0.15, abs=0.01)


def test_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        time_split(_frame(), train_frac=0.9, val_frac=0.2)
