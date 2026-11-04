"""Streamlit app: evaluate the five saved breast-cancer classifiers on an uploaded test CSV.

The app only loads the fitted pipelines from model/ and scores them; it never
calls fit, so uploaded data cannot change a model.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from src.data_prep import FEATURE_COLUMNS, TARGET_COLUMN, TEST_DATA_PATH
from src.evaluation import LABEL_DISPLAY, METRIC_NAMES, evaluate_model
from src.train_models import MODEL_NAMES, load_pipelines
from src.upload_validation import UploadError, load_test_upload

st.set_page_config(page_title="Breast Cancer Classifier Lab", layout="wide")


@st.cache_resource
def get_models():
    # Trusted, project-produced pickles only; no user-supplied model files.
    return load_pipelines()


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def confusion_figure(matrix, title):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=LABEL_DISPLAY)
    ax.set_yticks([0, 1], labels=LABEL_DISPLAY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    return fig


st.title("Breast Cancer Classifier Lab")
st.caption(
    "Compare five saved scikit-learn pipelines on a test CSV. "
    "Educational use only; not for clinical decisions."
)

with st.sidebar:
    st.header("About the data")
    st.write("**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository")
    st.write(f"{len(FEATURE_COLUMNS)} numeric features, binary target `{TARGET_COLUMN}`")
    st.write("Target: **0 = benign, 1 = malignant**")
    st.write(
        f"Upload a CSV with all {len(FEATURE_COLUMNS)} feature columns plus `{TARGET_COLUMN}`. "
        "Models are loaded from disk and are never retrained on your upload."
    )
    if TEST_DATA_PATH.exists():
        st.download_button(
            "Download example test_data.csv",
            TEST_DATA_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

try:
    models = get_models()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # incompatible library version, corrupt file, etc.
    st.error(
        "A saved model could not be loaded, possibly because of a scikit-learn "
        f"version mismatch. Details: {exc}"
    )
    st.stop()

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if st.button("Evaluate the bundled test_data.csv instead"):
    st.session_state["use_bundled"] = True
if uploaded is not None:
    st.session_state["use_bundled"] = False

source = None
if uploaded is not None:
    source = uploaded.getvalue()
    source_label = f"uploaded file `{uploaded.name}`"
elif st.session_state.get("use_bundled") and TEST_DATA_PATH.exists():
    source = TEST_DATA_PATH.read_bytes()
    source_label = "bundled `test_data.csv`"

if source is None:
    st.info("Upload a test CSV (or use the bundled example) to see results.")
    st.stop()

try:
    X, y = load_test_upload(source)
except UploadError as exc:
    st.error("Could not use this file:")
    for message in exc.messages:
        st.markdown(f"- {message}")
    st.stop()

counts = y.value_counts().reindex([0, 1], fill_value=0)
st.success(
    f"Evaluating {len(X)} rows from the {source_label}: "
    f"{counts[0]} benign, {counts[1]} malignant."
)
if (counts == 0).any():
    st.warning(
        "The file contains only one true class. AUC cannot be computed and is shown as N/A; "
        "precision, recall, F1 and MCC may be 0 where they are undefined."
    )

results = {key: evaluate_model(pipeline, X, y) for key, pipeline in models.items()}

choice = st.selectbox("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
selected = results[choice]

st.subheader(f"{MODEL_NAMES[choice]}: evaluation metrics")
columns = st.columns(len(METRIC_NAMES))
for column, name in zip(columns, METRIC_NAMES):
    column.metric(name, fmt(selected["metrics"][name]))
for note in selected["notes"]:
    st.caption(note)

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    figure = confusion_figure(selected["confusion_matrix"], MODEL_NAMES[choice])
    st.pyplot(figure)
    plt.close(figure)
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, models[choice].predict(X), labels=[0, 1], target_names=LABEL_DISPLAY,
        zero_division=0, output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("All five models on the same rows")
comparison = pd.DataFrame(
    [{"ML Model Name": MODEL_NAMES[key], **{m: fmt(r["metrics"][m]) for m in METRIC_NAMES}}
     for key, r in results.items()]
)
st.dataframe(comparison, hide_index=True, use_container_width=True)
