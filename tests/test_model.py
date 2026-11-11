import numpy as np
import pandas as pd

from fraud_drift.model import FEATURES, add_features, build_logistic, fit_reference


def _data(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.uniform(1, 500, n)
    df["Class"] = (rng.uniform(size=n) < 0.05).astype(int)
    return add_features(df)


def test_reference_model_is_fit_on_train_rows_only():
    df = _data()
    train_mask = np.arange(len(df)) < 300
    # Shift the held-out rows far away: any leakage would move the scaler mean.
    df.loc[~train_mask, "V1"] += 1000.0
    model = fit_reference(df, train_mask)
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, df.loc[train_mask, FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, df[FEATURES].mean().to_numpy())


def test_reference_model_matches_manual_fit_on_train_rows():
    df = _data()
    train_mask = np.arange(len(df)) < 300
    model = fit_reference(df, train_mask)
    manual = build_logistic().fit(df.loc[train_mask, FEATURES], df.loc[train_mask, "Class"])
    np.testing.assert_allclose(
        model.named_steps["clf"].coef_, manual.named_steps["clf"].coef_
    )


def test_add_features_does_not_change_input():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Amount": rng.uniform(1, 500, 10)})
    out = add_features(df)
    assert "log_amount" not in df.columns
    np.testing.assert_allclose(out["log_amount"], np.log1p(df["Amount"]))
