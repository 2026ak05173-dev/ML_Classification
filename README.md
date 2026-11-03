# ML_Classification

Machine Learning classification pipeline on one public dataset, evaluated and demonstrated in a Streamlit app.

> Work in progress. Results, observations, the deployed app link and the final README sections are added as each stage is completed.

## Assignment scope

- Train and evaluate Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM on the same dataset.
- Report Accuracy, AUC, Precision, Recall, F1 and MCC for each model.
- Save one fitted scikit-learn pipeline per model as a `.pkl` file under `model/`.
- Provide a Streamlit app (deployed on Streamlit Community Cloud) that uploads a test CSV, lets the user pick a model, and shows metrics and a confusion matrix using the saved models only (no retraining).

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

| Property | Value |
|---|---|
| Instances | 569 |
| Input features | 30 real-valued (10 nucleus measurements x mean, standard error, worst) |
| Target | `diagnosis`: malignant (`M`) or benign (`B`) |
| Task | Binary classification |
| Missing values (per UCI) | None; re-verified by the data checks in `src/data_prep.py` |
| Eligibility | Exceeds the assignment minimums of 12 features and 500 instances |

Features are computed from digitized images of fine needle aspirates of breast masses; only the numerical table is used.

### Handling decisions

- The original files are kept unchanged in `data/` (`wdbc.data`, `wdbc.names`) for reference
- Column names follow the UCI feature order (see `data/wdbc.names`).
- The ID column is dropped; all 30 measurements are kept as predictors.
- Target encoding is explicit everywhere: **malignant = 1, benign = 0**.


## Data validation and shared split

Run from the project root:

```bash
python -m src.data_prep   # validates the data, creates the split, writes test_data.csv
python -m pytest tests    # data checks and split tests
```

Verified results:

- 569 rows x 32 raw columns; no missing or non-finite values; no duplicate IDs or duplicate feature rows.
- Class counts: 357 benign (0), 212 malignant (1), matching UCI.
- One stratified 80:20 split with `random_state=42`: 455 training rows (285 / 170) and 114 test rows (72 / 42).
- All five models will use these same rows and feature order. Preprocessing is fitted on the training partition only (inside each model pipeline).
- `test_data.csv` holds the 114 held-out rows: 30 raw feature columns plus the encoded `diagnosis` column, no index column.
