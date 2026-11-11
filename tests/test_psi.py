import numpy as np
import pytest

from fraud_drift.psi import bin_shares, psi, psi_from_shares, reference_edges


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_hand_computed_two_bin_example():
    # Reference median is 1.5, so bins are {0, 1} and {2, 3}: expected shares [0.5, 0.5].
    # Actual values [0, 0, 0, 3] give shares [0.75, 0.25].
    # PSI = 0.25 * ln(1.5) + (-0.25) * ln(0.5)
    expected_value = 0.25 * np.log(1.5) + 0.25 * np.log(2.0)
    assert psi([0, 1, 2, 3], [0, 0, 0, 3], n_bins=2) == pytest.approx(expected_value)


def test_psi_is_larger_for_a_big_shift_than_a_mild_one():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=5000)
    mild = psi(reference, rng.normal(loc=0.1, size=5000))
    large = psi(reference, rng.normal(loc=1.0, size=5000))
    assert mild > 0
    assert large > mild


def test_bin_edges_come_from_reference_only():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=3000)
    edges = reference_edges(reference, n_bins=10)
    shifted = reference + 100.0
    # Edges fixed by the reference put every shifted value into the open last bin.
    shares = bin_shares(shifted, edges)
    assert shares[-1] == pytest.approx(1.0)
    assert psi(reference, shifted) == pytest.approx(
        psi_from_shares(bin_shares(reference, edges), shares)
    )
    np.testing.assert_array_equal(reference_edges(reference, n_bins=10), edges)


def test_duplicate_quantiles_are_collapsed_without_error():
    reference = np.array([0.0] * 90 + [1.0] * 10)
    edges = reference_edges(reference, n_bins=10)
    assert len(edges) < 9
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_zero_share_bins_are_clipped_so_psi_is_finite():
    value = psi_from_shares([1.0, 0.0], [0.0, 1.0])
    assert np.isfinite(value)
    assert value > 0


def test_psi_rejects_nan_in_actual():
    with pytest.raises(ValueError):
        psi([0.0, 1.0, 2.0], [0.0, np.nan])
