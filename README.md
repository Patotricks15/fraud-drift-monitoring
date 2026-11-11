# fraud-drift-monitoring (FD06)

**Question:** When does the production model stop being reliable?

## Data

- Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013, 284,807 rows, 492 frauds).
- License per Kaggle API: Database Open License. Metadata checked 2026-11-08. The raw file is read in place and is not committed. See [data/README.md](data/README.md).
- `Time` covers about two days. Monitoring therefore uses 6-hour windows, not monthly windows. Monthly drift cannot be observed in this dataset.
- The dataset has no card, account or device IDs, so drift is measured at the population level only.

## Method

- Features: `V1`-`V28` (PCA components, used as provided) and `log_amount = log1p(Amount)`. `Time` and `Class` are excluded from features.
- Reference model: `StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced", random_state=42)`, version `logistic-C1.0-balanced`. It is fit on the earliest 70% of `Time` (train window) only.
- Monitored period: the next 15% (validation) and last 15% (test) of `Time`. The validation window provides the reference PR-AUC. The reference PR-AUC is measured out of sample on that window.
- Windows: fixed-width, 6 hours (`window_seconds = 21600`, configurable), labelled by their start hour counted from the first transaction. The first monitored window may be partial because the train boundary is a time quantile, not a window edge.
- PSI: `PSI = sum((a_i - e_i) * ln(a_i / e_i))`, where `e_i` and `a_i` are the reference and window shares in each bin. Bin edges are the training-window deciles, duplicates removed. Outer bins are open, so values outside the reference range are counted. Shares are clipped to `1e-6` before the logarithm. PSI is computed for the predicted score and for every feature, in every monitored window.
- Rolling PR-AUC: average precision computed per 6-hour window. Windows with fewer than 5 frauds get NaN and do not count toward the trigger.

## Alert rules and retraining trigger

| Rule | Threshold |
|---|---|
| PSI stable | PSI <= 0.10 |
| PSI warning | 0.10 < PSI <= 0.25 |
| PSI alert (major shift) | PSI > 0.25 |
| Retrain on score PSI | score PSI > 0.25 in 2 consecutive windows |
| Retrain on PR-AUC | PR-AUC more than 30% below the validation reference in 2 consecutive windows, each with at least 5 frauds |

Retraining is triggered if either rule fires. "Consecutive" means adjacent 6-hour windows. A missing window or an ineligible window breaks the run. The decision record is written to `reports/retraining_trigger.json`.

## Results

Not yet produced. The notebook needs the Kaggle dataset, which is not downloaded to this workstation. After running the notebook on Kaggle, fill in the tables below from the generated files.

| Item | Value |
|---|---|
| Reference PR-AUC (validation) | pending Kaggle run |
| Monitored windows | pending Kaggle run |
| First window with a PSI alert on score | pending Kaggle run |
| First window with PR-AUC drop > 30% | pending Kaggle run |
| Retraining trigger fired | pending Kaggle run |

## Decision

Pending the Kaggle run. The decision is whether the retraining trigger fires, and in which 6-hour window, on the real validation and test windows.

## Limitation

- The two-day window is too short for monthly or seasonal drift. Drift is shown only at a 6-hour scale.
- No entity-level drift, because the dataset has no card or device IDs.
- The features are anonymized PCA components, so feature PSI is interpretive only. A shifted `V14` cannot be linked to a business meaning.
- The score PSI uses in-sample predictions on the training window as the reference, so the reference distribution is slightly optimistic.
- The reference model and thresholds are fixed, not tuned. The alert thresholds (0.10, 0.25, 30%, 5 frauds) are conventional choices, not calibrated to this data.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
# On Kaggle: attach mlg-ulb/creditcardfraud, upload the notebook, Run All.
# Locally (development only): export FRAUD_DATA_PATH=/path/to/creditcard.csv
make notebook
```

Generated tables are written to `reports/` and are not tracked in git until a real run is reviewed:

- `reports/psi_by_window.csv`: rows = window x feature (including `score`), with `window_start` (hours), `feature`, `psi`, `status`.
- `reports/rolling_pr_auc.csv`: per window `window_start`, `n_rows`, `n_frauds`, `pr_auc`, `reference_pr_auc`, `relative_drop`.
- `reports/retraining_trigger.json`: decision record with thresholds, reference PR-AUC, and fired windows.

Figures are written to `reports/figures/`: `fd06_psi_worst_window.png` and `fd06_rolling_pr_auc.png`.

## Layout

```text
src/fraud_drift/              data loading and time split, metrics, PSI, windows, reference model, trigger, plots
notebooks/                    jupytext source (.py) and paired notebook (.ipynb)
tests/                        PSI, windows, trigger, model leakage, metrics and split tests
reports/                      generated tables (ignored) and figures (ignored)
```
