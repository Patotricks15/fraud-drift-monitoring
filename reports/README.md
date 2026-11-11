# Reports

Generated files are written here by `notebooks/fd06_drift_monitoring.ipynb`, which needs the real Kaggle dataset:

- `psi_by_window.csv`: PSI per 6-hour window and feature, against the training reference.
- `rolling_pr_auc.csv`: PR-AUC per 6-hour window, against the validation reference.
- `retraining_trigger.json`: retraining decision record.
- `figures/`: PNG figures.

The CSV and JSON files are git-ignored until a real run has been reviewed. No results are committed yet.
