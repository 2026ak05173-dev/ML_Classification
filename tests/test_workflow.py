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


from src import train_models as tm


@pytest.fixture(scope="module")
def fitted(splits):
    X_train, _, y_train, _ = splits
    return tm.fit_pipelines(X_train, y_train)


def test_five_required_pipelines():
    pipelines = tm.build_pipelines()
    assert set(pipelines) == set(tm.MODEL_NAMES)
    assert len(pipelines) == 5


def test_pipeline_configuration():
    p = tm.build_pipelines()
    assert p["logistic_regression"].named_steps["model"].max_iter == 2000
    assert p["knn"].named_steps["model"].n_neighbors == 5
    assert p["svm"].named_steps["model"].kernel == "rbf"
    assert p["decision_tree"].named_steps["scaler"] == "passthrough"
    for key in ("logistic_regression", "knn", "naive_bayes", "svm"):
        assert p[key].named_steps["scaler"].__class__.__name__ == "StandardScaler"


def test_scaler_fitted_on_training_data_only(splits, fitted):
    X_train, X_test, _, _ = splits
    scaler = fitted["svm"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, X_test.mean().to_numpy())


def test_fitted_pipelines_predict_binary_labels(splits, fitted):
    _, X_test, _, _ = splits
    for pipeline in fitted.values():
        assert list(pipeline.classes_) == [0, 1]
        preds = pipeline.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(np.unique(preds)) <= {0, 1}


import pandas as pd
from sklearn.metrics import roc_auc_score

from src import evaluation as ev


def test_evaluate_model_metrics_and_confusion_matrix(splits, fitted):
    _, X_test, _, y_test = splits
    for key, pipeline in fitted.items():
        result = ev.evaluate_model(pipeline, X_test, y_test)
        assert list(result["metrics"]) == ev.METRIC_NAMES
        assert all(0 <= v <= 1 for k, v in result["metrics"].items() if k != "MCC")
        cm = result["confusion_matrix"]
        assert cm.shape == (2, 2) and cm.sum() == len(y_test)
        assert cm[0].sum() == (y_test == 0).sum()


def test_auc_uses_scores_not_labels(splits, fitted):
    _, X_test, _, y_test = splits
    for key in ("svm", "logistic_regression"):
        result = ev.evaluate_model(fitted[key], X_test, y_test)
        label_auc = roc_auc_score(y_test, fitted[key].predict(X_test))
        assert result["metrics"]["AUC"] != pytest.approx(label_auc)
    svm_scores = ev.positive_class_scores(fitted["svm"], X_test)
    np.testing.assert_allclose(svm_scores, fitted["svm"].decision_function(X_test))


def test_single_class_auc_is_unavailable(splits, fitted):
    _, X_test, _, y_test = splits
    mask = (y_test == 1).to_numpy()
    result = ev.evaluate_model(fitted["knn"], X_test[mask], y_test[mask])
    assert result["metrics"]["AUC"] is None
    assert result["notes"]
    assert result["confusion_matrix"].shape == (2, 2)


def test_pickle_roundtrip_matches_memory(splits, fitted, tmp_path):
    _, X_test, _, _ = splits
    tm.save_pipelines(fitted, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.pkl")) == sorted(
        f"{k}.pkl" for k in tm.MODEL_NAMES
    )
    tm.verify_reloaded(fitted, X_test, tmp_path)


def test_missing_model_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_models"):
        tm.load_pipeline("svm", tmp_path)


def test_committed_models_and_metrics_are_consistent(splits):
    _, X_test, _, y_test = splits
    loaded = tm.load_pipelines()
    for pipeline in loaded.values():
        assert hasattr(pipeline, "named_steps")
    table = tm.metrics_table(loaded, X_test, y_test)
    saved = pd.read_csv(tm.METRICS_PATH)
    pd.testing.assert_frame_equal(table, saved, check_exact=False, rtol=1e-9)
