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

## Classification pipelines

[`src/train_models.py`](src/train_models.py) defines five baseline scikit-learn pipelines, all fitted on the shared training split only:

| Model | Pipeline | Configuration |
|---|---|---|
| Logistic Regression | StandardScaler -> LogisticRegression | `max_iter=2000`, `random_state=42` |
| Decision Tree | passthrough -> DecisionTreeClassifier | `random_state=42` |
| KNN | StandardScaler -> KNeighborsClassifier | `n_neighbors=5` |
| Naive Bayes | StandardScaler -> GaussianNB | Gaussian variant |
| Support Vector Machine | StandardScaler -> SVC | RBF kernel, `probability=False` |

These are untuned starting configurations.

## Evaluation and saved models

```bash
pip install -r requirements.txt
python -m src.train_models   # fits, saves model/*.pkl, reloads and verifies, writes results/metrics_summary.csv
```

[`src/evaluation.py`](src/evaluation.py) holds the single evaluation function shared by training and the Streamlit app.

- AUC uses the class-1 probability (Logistic Regression, Decision Tree, KNN, Naive Bayes) or `decision_function` score (SVM), never hard labels. With a single true class AUC is reported as unavailable.
- Precision, recall and F1 are for the malignant class (`pos_label=1`) with `zero_division=0`; the confusion matrix uses label order `[0, 1]`.
- Each complete fitted pipeline (scaler included) is saved with `pickle` to `model/` (`logistic_regression.pkl`, `decision_tree.pkl`, `knn.pkl`, `naive_bayes.pkl`, `svm.pkl`). Each file is reloaded and checked for identical predictions and scores. Only load `.pkl` files produced by this project; pickle files from untrusted sources can execute code.
- Tested with Python 3.10.16 and the versions pinned in `requirements.txt`; the models must be loaded with the same scikit-learn version.

Held-out test results (114 rows, from `results/metrics_summary.csv`; observations are added after final verification on BITS Virtual Lab):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
