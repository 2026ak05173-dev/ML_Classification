import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_prep as dp


@pytest.fixture(scope="module")
def raw():
    return dp.load_raw_data()


@pytest.fixture(scope="module")
def splits(raw):
    X, y = dp.prepare_features_target(raw)
    return dp.split_data(X, y)


def test_raw_data_shape_and_columns(raw):
    assert raw.shape == (569, 32)
    assert list(raw.columns) == dp.RAW_COLUMNS
    assert len(dp.FEATURE_COLUMNS) == 30


def test_validation_report(raw):
    report = dp.validate_raw_data(raw)
    assert report["n_rows"] == 569
    assert report["n_features"] == 30
    assert report["class_counts"] == {0: 357, 1: 212}


def test_validation_rejects_missing_values(raw):
    bad = raw.copy()
    bad.loc[0, "radius_mean"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        dp.validate_raw_data(bad)


def test_validation_rejects_infinite_values(raw):
    bad = raw.copy()
    bad.loc[0, "area_worst"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        dp.validate_raw_data(bad)


def test_validation_rejects_unknown_label(raw):
    bad = raw.copy()
    bad.loc[0, "diagnosis"] = "X"
    with pytest.raises(ValueError, match="diagnosis"):
        dp.validate_raw_data(bad)


def test_target_mapping_malignant_is_one(raw):
    _, y = dp.prepare_features_target(raw)
    assert (y[raw["diagnosis"] == "M"] == 1).all()
    assert (y[raw["diagnosis"] == "B"] == 0).all()


def test_features_exclude_id_and_target(raw):
    X, _ = dp.prepare_features_target(raw)
    assert list(X.columns) == dp.FEATURE_COLUMNS
    assert dp.ID_COLUMN not in X.columns
    assert dp.TARGET_COLUMN not in X.columns


def test_split_sizes_and_stratification(splits):
    X_train, X_test, y_train, y_test = splits
    assert len(X_train) + len(X_test) == 569
    assert len(X_test) == 114
    assert set(X_train.index).isdisjoint(X_test.index)
    full_rate = (212) / 569
    assert abs(y_train.mean() - full_rate) < 0.01
    assert abs(y_test.mean() - full_rate) < 0.01


def test_split_is_reproducible(raw, splits):
    X, y = dp.prepare_features_target(raw)
    again = dp.split_data(X, y)
    assert list(again[1].index) == list(splits[1].index)


def test_bundled_test_csv_matches_split(splits):
    import pandas as pd

    _, X_test, _, y_test = splits
    bundled = pd.read_csv(dp.TEST_DATA_PATH)
    assert list(bundled.columns) == dp.FEATURE_COLUMNS + [dp.TARGET_COLUMN]
    assert len(bundled) == len(X_test)
    assert bundled[dp.TARGET_COLUMN].tolist() == y_test.tolist()
    np.testing.assert_allclose(bundled[dp.FEATURE_COLUMNS].to_numpy(), X_test.to_numpy())
