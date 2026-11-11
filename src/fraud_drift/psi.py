"""Population Stability Index with bin edges fixed by a reference sample."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _as_finite_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def reference_edges(reference, n_bins: int = 10) -> np.ndarray:
    """Interior bin edges at reference quantiles. Duplicate quantiles are dropped."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    ref = _as_finite_1d(reference, "reference")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(ref, quantiles))


def bin_shares(values, edges) -> np.ndarray:
    """Share of values in each bin. Outer bins are open, so every value is counted."""
    arr = _as_finite_1d(values, "values")
    edges = np.asarray(edges, dtype=float)
    index = np.searchsorted(edges, arr, side="right")
    counts = np.bincount(index, minlength=len(edges) + 1)
    return counts / arr.size


def psi_from_shares(expected, actual) -> float:
    """PSI = sum((a_i - e_i) * ln(a_i / e_i)), with shares clipped to EPS."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same number of bins")
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, n_bins: int = 10) -> float:
    """PSI of `actual` against `expected`, with bins taken from `expected` only."""
    edges = reference_edges(expected, n_bins)
    return psi_from_shares(bin_shares(expected, edges), bin_shares(actual, edges))
