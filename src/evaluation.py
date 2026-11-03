"""Shared evaluation used by both training and the Streamlit app."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
LABELS = [0, 1]
LABEL_DISPLAY = ["benign", "malignant"]
AUC_SINGLE_CLASS_NOTE = (
    "AUC is unavailable because the evaluated data contains only one true class."
)


def positive_class_scores(pipeline, X):
    """Continuous scores oriented toward class 1, never hard labels."""
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model has no positive class (1).")
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, classes.index(1)]
    scores = pipeline.decision_function(X)
    # Binary decision_function is already positive for classes_[1]; flip if class 1 is first.
    return scores if classes.index(1) == 1 else -scores


def evaluate_model(pipeline, X, y):
    """Return the six required metrics, the [0, 1] confusion matrix and notes.

    AUC is None (not zero) when only one true class is present. Precision,
    recall and F1 use zero_division=0, so they are 0 when undefined.
    """
    y_true = np.asarray(y)
    y_pred = pipeline.predict(X)

    notes = []
    if len(np.unique(y_true)) < 2:
        auc = None
        notes.append(AUC_SINGLE_CLASS_NOTE)
    else:
        auc = float(roc_auc_score(y_true, positive_class_scores(pipeline, X)))

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        notes.append("MCC is 0 by convention when predictions or true labels are single-class.")

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
        "notes": notes,
    }
