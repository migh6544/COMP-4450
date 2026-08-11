"""Streamlit monitoring dashboard for Assignment 5 - Model Monitoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATASET_PATH: Final[Path] = BASE_DIR / "IMDB Dataset.csv"
LOG_PATH: Final[Path] = Path("/logs/prediction_logs.json")
API_SERVICE_URL: Final[str] = os.getenv(
    "API_SERVICE_URL", "http://sentiment-api:8000"
).rstrip("/")
VALID_SENTIMENTS: Final[list[str]] = ["negative", "positive"]


st.set_page_config(
    page_title="Sentiment Model Monitoring",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_reference_data(path: str) -> pd.DataFrame:
    """Load and validate the IMDB reference/training dataset."""
    df = pd.read_csv(path, usecols=["review", "sentiment"])
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["text_length_words"] = df["review"].str.split().str.len()
    return df


def load_prediction_logs(path: Path) -> pd.DataFrame:
    """Read the shared JSON Lines log, ignoring a partially written final line."""
    columns = [
        "timestamp",
        "request_text",
        "predicted_sentiment",
        "true_sentiment",
    ]

    if not path.is_file():
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as log_file:
            for line_number, raw_line in enumerate(log_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # The dashboard may read while the API is appending. A partial
                    # final line can safely be skipped and picked up on the next rerun.
                    continue

                if all(column in record for column in columns):
                    rows.append(record)
    except OSError as exc:
        st.warning(f"Could not read prediction logs: {exc}")
        return pd.DataFrame(columns=columns)

    logs = pd.DataFrame(rows, columns=columns)
    if logs.empty:
        return logs

    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce", utc=True)
    logs["request_text"] = logs["request_text"].astype(str)
    logs["predicted_sentiment"] = (
        logs["predicted_sentiment"].astype(str).str.strip().str.lower()
    )
    logs["true_sentiment"] = (
        logs["true_sentiment"].astype(str).str.strip().str.lower()
    )
    logs["text_length_words"] = logs["request_text"].str.split().str.len()
    return logs


def compute_feedback_metrics(logs: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """Compute accuracy and positive-class precision from valid feedback rows."""
    if logs.empty:
        return None, None, 0

    valid = logs[
        logs["predicted_sentiment"].isin(VALID_SENTIMENTS)
        & logs["true_sentiment"].isin(VALID_SENTIMENTS)
    ]
    if valid.empty:
        return None, None, 0

    y_true = valid["true_sentiment"]
    y_pred = valid["predicted_sentiment"]
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(
        precision_score(
            y_true,
            y_pred,
            pos_label="positive",
            zero_division=0,
        )
    )
    return accuracy, precision, len(valid)


def api_is_healthy() -> bool:
    """Check the FastAPI container over the shared Docker network."""
    try:
        response = requests.get(f"{API_SERVICE_URL}/health", timeout=1.5)
        return response.ok and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def plot_data_drift(reference: pd.Series, production: pd.Series) -> plt.Figure:
    """Compare reference and inference text-length distributions."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    combined_max = max(int(reference.max()), int(production.max()))
    bins = min(50, max(10, combined_max // 25))

    ax.hist(
        reference,
        bins=bins,
        density=True,
        alpha=0.55,
        label="IMDB reference data",
    )
    ax.hist(
        production,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Logged inference requests",
    )
    ax.set_title("Data Drift: Review Length Distribution")
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_target_drift(reference_df: pd.DataFrame, logs: pd.DataFrame) -> plt.Figure:
    """Compare reference labels with the distribution of production predictions."""
    reference_distribution = (
        reference_df["sentiment"]
        .value_counts(normalize=True)
        .reindex(VALID_SENTIMENTS, fill_value=0)
        * 100
    )
    prediction_distribution = (
        logs["predicted_sentiment"]
        .value_counts(normalize=True)
        .reindex(VALID_SENTIMENTS, fill_value=0)
        * 100
    )

    comparison = pd.DataFrame(
        {
            "Training/reference sentiment": reference_distribution,
            "Logged predicted sentiment": prediction_distribution,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Target / Prediction Drift")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Share of observations (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


try:
    reference_df = load_reference_data(str(DATASET_PATH))
except Exception as exc:  # pragma: no cover - container asset failure
    st.error(f"Unable to load the IMDB reference dataset: {exc}")
    st.stop()

logs_df = load_prediction_logs(LOG_PATH)
accuracy, precision, feedback_count = compute_feedback_metrics(logs_df)

st.title("Sentiment Model Monitoring Dashboard")
st.caption(
    "Reference data: IMDB Dataset.csv | Production data: /logs/prediction_logs.json"
)

# The assignment explicitly requires a prominent warning when accuracy < 80%.
if accuracy is not None and accuracy < 0.80:
    st.error(
        f"⚠️ MODEL PERFORMANCE ALERT: Accuracy is {accuracy:.1%}, below the 80% threshold."
    )

if st.button("Refresh dashboard"):
    st.rerun()

status_col, request_col, feedback_col = st.columns(3)
status_col.metric("FastAPI Service", "Online" if api_is_healthy() else "Unavailable")
request_col.metric("Logged Predictions", f"{len(logs_df):,}")
feedback_col.metric("Feedback Records", f"{feedback_count:,}")

st.divider()
st.subheader("Model Accuracy & User Feedback")

metric_col1, metric_col2 = st.columns(2)
if accuracy is None:
    metric_col1.metric("Accuracy", "N/A")
    metric_col2.metric("Precision (positive)", "N/A")
    st.info(
        "No valid feedback has been logged yet. Send requests to POST /predict "
        "with both text and true_sentiment, or run evaluate.py."
    )
else:
    metric_col1.metric("Accuracy", f"{accuracy:.2%}")
    metric_col2.metric("Precision (positive)", f"{precision:.2%}")
    st.caption(
        f"Metrics are calculated across all {feedback_count:,} logged records with valid true labels."
    )

st.divider()
st.subheader("Data Drift Analysis")
st.write(
    "This assignment uses review length in **words** as the monitored input feature. "
    "The chart compares the IMDB reference distribution with live inference requests."
)

if logs_df.empty:
    st.info("No inference requests are available yet, so data drift cannot be plotted.")
else:
    data_drift_figure = plot_data_drift(
        reference_df["text_length_words"],
        logs_df["text_length_words"],
    )
    st.pyplot(data_drift_figure, clear_figure=True)

st.divider()
st.subheader("Target Drift Analysis")
st.write(
    "The training/reference sentiment distribution is compared with the distribution "
    "of the model's logged production predictions."
)

if logs_df.empty:
    st.info("No model predictions are available yet, so target drift cannot be plotted.")
else:
    target_drift_figure = plot_target_drift(reference_df, logs_df)
    st.pyplot(target_drift_figure, clear_figure=True)

st.divider()
st.subheader("Recent Prediction Logs")
if logs_df.empty:
    st.info("The shared prediction log is currently empty.")
else:
    display_columns = [
        "timestamp",
        "request_text",
        "predicted_sentiment",
        "true_sentiment",
        "text_length_words",
    ]
    recent = logs_df.sort_values("timestamp", ascending=False).head(20)
    st.dataframe(recent[display_columns], use_container_width=True, hide_index=True)
