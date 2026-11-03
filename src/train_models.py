"""Define, fit, evaluate and persist the five required classification pipelines."""

import pickle
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_prep import PROJECT_ROOT, RANDOM_STATE, load_and_split
from src.evaluation import METRIC_NAMES, evaluate_model, positive_class_scores

MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"

# Keys are file-name stems used for model/<key>.pkl.
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


def build_pipelines():
    """Return unfitted baseline pipelines keyed by model key.

    Every pipeline is scaler -> classifier so preprocessing is fitted on
    training data only and travels with the model. The decision tree uses a
    passthrough step because it is insensitive to feature scaling.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=False)),
        ]),
    }


def fit_pipelines(X_train, y_train):
    """Fit every pipeline on the training partition only."""
    pipelines = build_pipelines()
    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)
    return pipelines


def model_path(key, model_dir=MODEL_DIR):
    return model_dir / f"{key}.pkl"


def save_pipelines(pipelines, model_dir=MODEL_DIR):
    """Pickle each complete fitted pipeline (scaler included)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for key, pipeline in pipelines.items():
        with open(model_path(key, model_dir), "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pipeline(key, model_dir=MODEL_DIR):
    """Load one trusted, project-produced pipeline. Never load untrusted pickles."""
    path = model_path(key, model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.name}. Run `python -m src.train_models`."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pipelines(model_dir=MODEL_DIR):
    return {key: load_pipeline(key, model_dir) for key in MODEL_NAMES}


def verify_reloaded(pipelines, X_test, model_dir=MODEL_DIR):
    """Raise if a reloaded pipeline's predictions or scores differ from memory."""
    for key, original in pipelines.items():
        reloaded = load_pipeline(key, model_dir)
        if not np.array_equal(original.predict(X_test), reloaded.predict(X_test)):
            raise RuntimeError(f"{key}: reloaded predictions differ from the in-memory model.")
        if not np.allclose(positive_class_scores(original, X_test),
                           positive_class_scores(reloaded, X_test)):
            raise RuntimeError(f"{key}: reloaded scores differ from the in-memory model.")


def metrics_table(pipelines, X, y):
    """One row per model with the six required metrics (full precision)."""
    rows = []
    for key, pipeline in pipelines.items():
        result = evaluate_model(pipeline, X, y)
        rows.append({"ML Model Name": MODEL_NAMES[key], **result["metrics"]})
    return pd.DataFrame(rows, columns=["ML Model Name"] + METRIC_NAMES)


def main():
    X_train, X_test, y_train, y_test, _ = load_and_split()
    pipelines = fit_pipelines(X_train, y_train)

    save_pipelines(pipelines)
    verify_reloaded(pipelines, X_test)

    table = metrics_table(pipelines, X_test, y_test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_PATH, index=False)

    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
          f"numpy {np.__version__}, pandas {pd.__version__}")
    print(f"Saved {len(pipelines)} pipelines to {MODEL_DIR.name}/ and verified reload.")
    print(table.round(4).to_string(index=False))
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
