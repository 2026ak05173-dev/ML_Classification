# ML_Classification

Machine Learning : five classification models on one public dataset, demonstrated in an interactive Streamlit app.

## a. Problem statement

Predict whether a breast mass is **malignant** or **benign** from 30 numeric measurements of cell nuclei computed from digitized fine needle aspirate images. This is a binary classification task. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same training split, evaluated on the same held-out rows with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and compared. A Streamlit app lets a user upload a test CSV, choose a model and inspect its metrics and confusion matrix using the saved models only.


## b. Dataset description

**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

| Property | Value |
|---|---|
| Instances | 569 |
| Input features | 30 real-valued (10 nucleus measurements x mean, standard error, worst) |
| Target | `diagnosis`: malignant (`M`) or benign (`B`) |
| Task | Binary classification |
| Class counts (verified) | 357 benign, 212 malignant |
| Missing values | None (UCI states none; re-verified by `src/data_prep.py`) |
| Eligibility | Exceeds the assignment minimums of 12 features and 500 instances |

The ten base measurements are radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each is given as a mean (`_mean`), standard error (`_se`) and worst value (`_worst`), giving 30 features. Features are computed from images; only the numerical table is used.

**Data handling**

- The original files are kept unchanged in `data/` (`wdbc.data`, `wdbc.names`) for reference.
- Column names follow the UCI feature order (see `data/wdbc.names`). The ID column is dropped; all 30 measurements are kept as predictors.
- Target encoding is explicit everywhere: **malignant = 1, benign = 0**.
- Data checks: 569 rows x 32 raw columns, no missing or non-finite values, no duplicate IDs or duplicate feature rows, class counts matching UCI.
- One stratified 80:20 split with `random_state=42`: 455 training rows (285 benign / 170 malignant) and 114 test rows (72 / 42). All five models use the same rows and feature order.
- Preprocessing (standard scaling) is fitted on the training partition only, inside each model pipeline.
- `test_data.csv` holds the 114 held-out rows: 30 raw feature columns plus the encoded `diagnosis` column, no index column.


## c. GitHub Repository Link

<https://github.com/2026ak05173-dev/ML_Classification>


### Streamlit app (Streamlit App deployment URL - https://mlclassification-cancer-diagnosis.streamlit.app/)

[`app.py`](app.py) provides:

- CSV upload (test data only), a button to evaluate the bundled `test_data.csv`, and a download of that example file.
- A model dropdown for the five saved pipelines.
- The six evaluation metrics, a labelled confusion matrix (benign/malignant) and a classification report for the selected model.
- A comparison table of all five models computed on the same uploaded rows.
- Saved-model inference only: pipelines are loaded from `model/`; the app never calls `fit`.

### Run locally

```bash
pip install -r requirements.txt
python -m src.data_prep       # validate data, create the split, write test_data.csv
python -m src.train_models    # fit, save model/*.pkl, reload and verify, write results/metrics_summary.csv
streamlit run app.py
python -m pytest tests        # needs pytest (not listed in requirements.txt)
```


## d. Models used

All five models are untuned baseline scikit-learn pipelines fitted on the shared training split. Results below are measured on the 114 held-out rows in `test_data.csv` and come from `results/metrics_summary.csv`; the app shows the same values when evaluating the bundled file.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Precision, Recall and F1 are for the malignant class (1). Confusion-matrix error counts on the same rows (false positives = benign predicted malignant; false negatives = malignant predicted benign): Logistic Regression 1 FP / 3 FN, Decision Tree 4 / 4, KNN 1 / 4, Naive Bayes 3 / 6, SVM 0 / 3.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall (accuracy 0.9649, F1 0.9512) and the highest AUC (0.9960). It made 4 errors on 114 rows (1 FP, 3 FN), so it separates the classes well and is close to the SVM. |
| Decision Tree | Weakest AUC (0.9246) and lower accuracy (0.9298) than the linear and kernel models, with 8 errors split evenly (4 FP, 4 FN). Its probability scores are coarse (leaf proportions), which may explain the lower AUC.  |
| KNN | Accuracy 0.9561 with high precision (0.9744) but recall of 0.9048: 1 FP and 4 FN. AUC 0.9823 is good but below Logistic Regression, SVM and Naive Bayes. Fixed at k=5 without tuning. |
| Naive Bayes | Lowest accuracy (0.9211), recall (0.8571), F1 (0.8889) and MCC (0.8292); it missed 6 of 42 malignant cases. Its AUC is high (0.9891), so it ranks cases well while its default decision threshold gives more false negatives. The feature-independence assumption may be a factor.  |
| Support Vector Machine | Best accuracy (0.9737), precision (1.0000, no false positives), F1 (0.9630) and MCC (0.9442), with 3 FN and AUC 0.9947 (second to Logistic Regression). |
| **Overall winner for this dataset** | **Support Vector Machine**, using the predeclared selection metric (malignant-class F1: 0.9630 vs 0.9512 for Logistic Regression). It also has the best MCC and fewest errors (3 of 114), with recall tied with Logistic Regression at 0.9286. The margin is one or two test rows from a single split with untuned models, so Logistic Regression is a near tie and no general superiority or clinical suitability is claimed. |

