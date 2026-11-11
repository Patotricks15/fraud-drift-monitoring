# Data

Raw data is not committed and is never downloaded to the workstation.

| Field | Value |
|---|---|
| Kaggle slug | `mlg-ulb/creditcardfraud` |
| Kaggle reference | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| License (per Kaggle API, checked 2026-11-08) | Database: Open Database License |
| Description | Real anonymized card transactions, September 2013 (284,807 rows, 492 frauds, PCA features V1-V28) |
| Notebook read path | `/kaggle/input/creditcardfraud/creditcard.csv` (attach the dataset in a Kaggle Notebook) |
| Local override | Set `FRAUD_DATA_PATH` to a local copy, for development only |
| Retrieval date | 2026-11-08 (metadata); data read in place in Kaggle |

The `Time` column is seconds from the first transaction, so the full dataset covers about two days. This is why monitoring uses 6-hour windows instead of monthly windows.

No card, account or device identifiers are present, so drift is monitored at the population level only.
