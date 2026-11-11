# %% [markdown]
# # FD06: Drift monitoring for the card fraud model
# Question: when does the production model stop being reliable?
# Dataset: Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`), time split 70/15/15.
# Reference: logistic model fit on the earliest 70% of `Time` (train only).
# Monitored: the validation and test windows, cut into fixed 6-hour windows labelled by start hour.
# The dataset covers about two days, so monthly windows are impossible.
# Source for the FD06 notebook. Convert with `jupytext --to ipynb` and run on Kaggle, or set `FRAUD_DATA_PATH` locally.

# %%
import json
from pathlib import Path

import pandas as pd

from fraud_drift import plots
from fraud_drift.data import load_creditcard, time_split
from fraud_drift.metrics import pr_auc
from fraud_drift.model import FEATURES, MODEL_VERSION, add_features, fit_reference, score
from fraud_drift.trigger import PR_AUC_DROP, SCORE_FEATURE, retraining_trigger, write_decision_record
from fraud_drift.windows import DEFAULT_WINDOW_SECONDS, psi_table, rolling_pr_auc, window_hours

REPORT_DIR = Path("../reports") if Path.cwd().name == "notebooks" else Path("reports")
FIG_DIR = REPORT_DIR / "figures"
WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS
N_BINS = 10
TOP_FEATURES = 10
DATASET_NAME = "creditcardfraud"
TITLE_SUFFIX = (
    f"{DATASET_NAME}, time split val+test, model {MODEL_VERSION}, "
    f"{window_hours(WINDOW_SECONDS):g} h windows"
)

# %% [markdown]
# ## Reference model on the train window

# %%
df = add_features(load_creditcard())
split = time_split(df)
y = df["Class"]
print(f"rows: {len(df):,}  frauds: {int(y.sum())}")
print(f"train end (Time): {split.train_end:,.0f}s  val end: {split.val_end:,.0f}s")
for name, mask in (("train", split.train), ("val", split.val), ("test", split.test)):
    print(f"{name}: {int(mask.sum()):,} rows, {int(y[mask].sum())} frauds")

model = fit_reference(df, split.train)
df["score"] = score(model, df)

# %% [markdown]
# ## Reference PR-AUC on the validation window (out of sample)

# %%
reference_pr_auc = pr_auc(y[split.val], df.loc[split.val, "score"])
print(f"reference PR-AUC (validation): {reference_pr_auc:.4f}")

# %% [markdown]
# ## Monitored windows: validation and test, per window PSI and PR-AUC

# %%
reference = df.loc[split.train]
monitored = df.loc[split.val | split.test]
monitor_cols = [*FEATURES, SCORE_FEATURE]

psi_by_window = psi_table(
    reference, monitored, monitor_cols, n_bins=N_BINS, window_seconds=WINDOW_SECONDS
)
rolling = rolling_pr_auc(reference_pr_auc, monitored, window_seconds=WINDOW_SECONDS)
print(f"monitored windows: {rolling['window_start'].nunique()}")
print(rolling)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
psi_by_window.to_csv(REPORT_DIR / "psi_by_window.csv", index=False)
rolling.to_csv(REPORT_DIR / "rolling_pr_auc.csv", index=False)

# %% [markdown]
# ## Alert rules and retraining trigger
#
# Alerts: score or feature PSI above 0.25 is an alert, 0.10 to 0.25 is a warning.
# Retraining: score PSI above 0.25 in 2 consecutive windows, or PR-AUC more than 30% below
# the validation reference in 2 consecutive windows with at least 5 frauds.

# %%
status_counts = psi_by_window["status"].value_counts().to_dict()
print("PSI status counts:", status_counts)

record = retraining_trigger(
    psi_by_window,
    rolling,
    reference_pr_auc=reference_pr_auc,
    window_hours=window_hours(WINDOW_SECONDS),
)
record["dataset"] = DATASET_NAME
record["split"] = "validation and test windows, reference = train window"
record["model_version"] = MODEL_VERSION
record["window_label"] = "start hour, fixed-width windows"
print(json.dumps(record, indent=2))
write_decision_record(record, REPORT_DIR / "retraining_trigger.json")

# %% [markdown]
# ## Figures: PSI at the worst score window and rolling PR-AUC

# %%
score_psi = psi_by_window[psi_by_window["feature"] == SCORE_FEATURE]
worst_start = float(score_psi.sort_values("psi").iloc[-1]["window_start"])
at_worst = psi_by_window[psi_by_window["window_start"] == worst_start]
top = at_worst.sort_values("psi", ascending=False).head(TOP_FEATURES)
if SCORE_FEATURE not in set(top["feature"]):
    top = pd.concat([top, at_worst[at_worst["feature"] == SCORE_FEATURE]])
print(top)

plots.plot_psi_bars(
    top,
    f"PSI at worst window (start hour {worst_start:g}) and top features, {TITLE_SUFFIX}",
    FIG_DIR / "fd06_psi_worst_window.png",
)
plots.plot_rolling_pr_auc(
    rolling,
    reference_pr_auc,
    PR_AUC_DROP,
    f"Rolling PR-AUC vs validation reference, {TITLE_SUFFIX}",
    FIG_DIR / "fd06_rolling_pr_auc.png",
)
