"""Figures for the FD06 report. Each title names the dataset, split, model version and window width."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .trigger import PSI_ALERT, PSI_WARNING  # noqa: E402

STATUS_COLORS = {"stable": "tab:green", "warning": "tab:orange", "alert": "tab:red"}


def save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_psi_bars(table, title: str, path: str | Path) -> Path:
    """Horizontal PSI bars for one window, with the warning and alert thresholds."""
    ordered = table.sort_values("psi", ascending=True)
    colors = [STATUS_COLORS.get(s, "grey") for s in ordered["status"]]
    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.3 * len(ordered))))
    ax.barh(ordered["feature"], ordered["psi"], color=colors)
    ax.axvline(PSI_WARNING, color="tab:orange", linestyle="--", label=f"warning > {PSI_WARNING}")
    ax.axvline(PSI_ALERT, color="tab:red", linestyle="--", label=f"alert > {PSI_ALERT}")
    ax.set_xlabel("PSI vs training reference")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return save(fig, path)


def plot_rolling_pr_auc(
    table, reference_pr_auc: float, drop_threshold: float, title: str, path: str | Path
) -> Path:
    """PR-AUC per window against the validation reference and the drop threshold band."""
    threshold = reference_pr_auc * (1 - drop_threshold)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(0, threshold, color="tab:red", alpha=0.08, label=f"below {drop_threshold:.0%} drop")
    ax.axhline(reference_pr_auc, color="black", linestyle="-", label="reference PR-AUC (validation)")
    ax.axhline(threshold, color="tab:red", linestyle="--", label=f"trigger threshold ({drop_threshold:.0%} drop)")
    ax.plot(table["window_start"], table["pr_auc"], marker="o", color="tab:blue", label="window PR-AUC")
    ax.set_xlabel("Window start (hours since first transaction)")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return save(fig, path)
