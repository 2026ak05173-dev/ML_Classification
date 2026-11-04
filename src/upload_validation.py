"""Read and validate an uploaded test CSV before it reaches the saved models."""

import csv
import io

import numpy as np
import pandas as pd

from src.data_prep import FEATURE_COLUMNS, TARGET_COLUMN

MAX_ROWS = 10_000
MAX_LISTED = 8


class UploadError(ValueError):
    """Raised with one or more actionable messages about an invalid upload."""

    def __init__(self, messages):
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__(" ".join(self.messages))


def _listed(items):
    items = [str(i) for i in items]
    shown = ", ".join(items[:MAX_LISTED])
    return shown + (f" (+{len(items) - MAX_LISTED} more)" if len(items) > MAX_LISTED else "")


def read_uploaded_csv(data):
    """Parse CSV bytes into a DataFrame.

    Duplicate headers are checked on the raw header row, because pandas would
    otherwise rename them silently (e.g. ``a`` -> ``a.1``).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not data or not data.strip():
        raise UploadError("The file is empty. Upload a CSV with a header row and data rows.")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError("The file is not valid UTF-8 text. Save it as a UTF-8 CSV and try again.")

    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error):
        raise UploadError("The file has no readable header row.")
    header = [h.strip() for h in header]
    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        raise UploadError(f"Duplicate column names found: {_listed(duplicates)}. Column names must be unique.")

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise UploadError("The file is empty. Upload a CSV with a header row and data rows.")
    except pd.errors.ParserError as exc:
        raise UploadError(f"The file could not be parsed as CSV: {exc}")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_test_frame(df):
    """Return (X, y) with features in training order and y as ints 0/1.

    All problems found are reported together so the user can fix them in one pass.
    """
    if df.empty:
        raise UploadError("The file has a header but no data rows.")
    if len(df) > MAX_ROWS:
        raise UploadError(
            f"The file has {len(df)} rows; upload test data only (at most {MAX_ROWS} rows)."
        )

    problems = []
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(
            f"Missing required columns: {_listed(missing)}. "
            f"The file needs all {len(FEATURE_COLUMNS)} feature columns plus the '{TARGET_COLUMN}' column."
        )
    unexpected = [c for c in df.columns if c not in required]
    if unexpected:
        problems.append(
            f"Unexpected columns: {_listed(unexpected)}. Remove them; "
            "extra columns (such as an ID or index) are not used as predictors."
        )
    if problems:
        raise UploadError(problems)

    features = df[FEATURE_COLUMNS]
    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        problems.append(f"Non-numeric values in feature columns: {_listed(non_numeric)}.")
    else:
        values = features.to_numpy(dtype=float)
        if np.isnan(values).any():
            cols = [c for c in FEATURE_COLUMNS if features[c].isna().any()]
            problems.append(f"Missing values in feature columns: {_listed(cols)}.")
        if np.isinf(values).any():
            cols = [c for c in FEATURE_COLUMNS if np.isinf(features[c].to_numpy(dtype=float)).any()]
            problems.append(f"Infinite values in feature columns: {_listed(cols)}.")

    target = df[TARGET_COLUMN]
    numeric_target = pd.to_numeric(target, errors="coerce")
    bad = target[numeric_target.isna() | ~numeric_target.isin([0, 1])]
    if len(bad):
        shown = _listed(sorted({str(v) for v in bad.unique()}))
        problems.append(
            f"Invalid '{TARGET_COLUMN}' values: {shown}. Use 0 (benign) or 1 (malignant); blanks are not allowed."
        )

    if problems:
        raise UploadError(problems)

    return features.astype(float).reset_index(drop=True), numeric_target.astype(int).reset_index(drop=True)


def load_test_upload(data):
    """Read and validate CSV bytes in one step."""
    return validate_test_frame(read_uploaded_csv(data))
