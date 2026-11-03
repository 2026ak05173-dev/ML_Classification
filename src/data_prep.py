"""Load, validate and split the UCI Breast Cancer Wisconsin (Diagnostic) data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
LABEL_MAP = {"B": 0, "M": 1}
CLASS_NAMES = {0: "benign", 1: "malignant"}

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXPECTED_ROWS = 569
EXPECTED_CLASS_COUNTS = {0: 357, 1: 212}

# UCI feature order: each base measurement as mean, standard error, worst.
BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
FEATURE_COLUMNS = (
    [f"{name}_mean" for name in BASE_FEATURES]
    + [f"{name}_se" for name in BASE_FEATURES]
    + [f"{name}_worst" for name in BASE_FEATURES]
)
RAW_COLUMNS = [ID_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS


def load_raw_data(path=RAW_DATA_PATH):
    """Read wdbc.data (headerless) and apply the UCI column names."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    return df


def validate_raw_data(df):
    """Check the raw table and return a report; raise ValueError on any problem.

    Duplicates are reported but never dropped, so unexpected records are
    investigated rather than silently removed.
    """
    problems = []

    if list(df.columns) != RAW_COLUMNS:
        problems.append("Columns do not match the expected 32 UCI columns.")
    if len(df) != EXPECTED_ROWS:
        problems.append(f"Expected {EXPECTED_ROWS} rows, found {len(df)}.")

    unknown_labels = set(df[TARGET_COLUMN].unique()) - set(LABEL_MAP)
    if unknown_labels:
        problems.append(f"Unexpected diagnosis values: {sorted(unknown_labels)}.")

    features = df[FEATURE_COLUMNS]
    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"Non-numeric feature columns: {non_numeric}.")
    else:
        if features.isna().any().any():
            problems.append("Missing values found in feature columns.")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            problems.append("Non-finite values found in feature columns.")

    if df[ID_COLUMN].duplicated().any():
        problems.append("Duplicate ID values found.")

    if problems:
        raise ValueError(" ".join(problems))

    encoded = df[TARGET_COLUMN].map(LABEL_MAP)
    class_counts = {int(k): int(v) for k, v in encoded.value_counts().sort_index().items()}
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            f"Class counts {class_counts} differ from the UCI-documented {EXPECTED_CLASS_COUNTS}."
        )

    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "class_counts": class_counts,
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "missing_values": 0,
    }


def prepare_features_target(df):
    """Return X (30 predictors, UCI order) and y (malignant=1, benign=0)."""
    X = df[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    y = df[TARGET_COLUMN].map(LABEL_MAP).astype(int).reset_index(drop=True)
    y.name = TARGET_COLUMN
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split shared by all five models."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def export_test_csv(X_test, y_test, path=TEST_DATA_PATH):
    """Write held-out raw features and encoded target, without an index column."""
    out = X_test.copy()
    out[TARGET_COLUMN] = y_test.to_numpy()
    out.to_csv(path, index=False)
    return out


def load_and_split():
    """Load, validate and split the bundled data; return the splits and report."""
    df = load_raw_data()
    report = validate_raw_data(df)
    X, y = prepare_features_target(df)
    return (*split_data(X, y), report)


def main():
    X_train, X_test, y_train, y_test, report = load_and_split()
    export_test_csv(X_test, y_test)

    print(f"Rows: {report['n_rows']}  Features: {report['n_features']}")
    print(f"Class counts (0=benign, 1=malignant): {report['class_counts']}")
    print(f"Duplicate feature rows: {report['duplicate_feature_rows']}")
    print(f"Train: {len(X_train)} rows, class counts {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test:  {len(X_test)} rows, class counts {y_test.value_counts().sort_index().to_dict()}")
    print(f"Wrote {TEST_DATA_PATH.name}")


if __name__ == "__main__":
    main()
