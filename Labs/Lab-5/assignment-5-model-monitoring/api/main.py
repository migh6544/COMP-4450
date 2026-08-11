"""FastAPI prediction service for Assignment 5 - Model Monitoring.

The service loads a trained IMDB sentiment model, exposes POST /predict,
and appends one JSON object per prediction to /logs/prediction_logs.json.
The /logs directory is intended to be backed by a named Docker volume that
is shared with the Streamlit monitoring container.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Final

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODEL_PATH: Final[Path] = BASE_DIR / "sentiment_model.pkl"
LOG_PATH: Final[Path] = Path("/logs/prediction_logs.json")
VALID_SENTIMENTS: Final[set[str]] = {"positive", "negative"}
LOG_LOCK = Lock()


def load_model(model_path: Path) -> Pipeline:
    """Load and validate the serialized sentiment model."""
    if not model_path.is_file():
        raise RuntimeError(f"Required model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - startup failure path
        raise RuntimeError(f"Unable to load sentiment model: {exc}") from exc

    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded sentiment model does not provide predict().")

    return model


MODEL: Final[Pipeline] = load_model(MODEL_PATH)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Assignment 5 Sentiment Prediction Service",
    description=(
        "Serves IMDB sentiment predictions and logs each prediction with "
        "ground-truth feedback for the monitoring dashboard."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(
        ...,
        min_length=1,
        description="Movie review text to classify.",
        examples=["An excellent film with a moving story."],
    )
    true_sentiment: str = Field(
        ...,
        description="Ground-truth feedback supplied through Postman/evaluate.py.",
        examples=["positive"],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must contain at least one non-whitespace character")
        return cleaned

    @field_validator("true_sentiment")
    @classmethod
    def validate_true_sentiment(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in VALID_SENTIMENTS:
            raise ValueError("true_sentiment must be 'positive' or 'negative'")
        return cleaned


class PredictionResponse(BaseModel):
    sentiment: str


class HealthResponse(BaseModel):
    status: str


def predict_label(text: str) -> str:
    """Run the trained model and normalize the resulting class label."""
    try:
        predicted = str(MODEL.predict([text])[0]).strip().lower()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model could not generate a prediction.",
        ) from exc

    if predicted not in VALID_SENTIMENTS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"The model returned an unexpected class label: {predicted}",
        )
    return predicted


def append_prediction_log(
    request_text: str,
    predicted_sentiment: str,
    true_sentiment: str,
) -> None:
    """Append one JSON object as one line in the shared prediction log."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_text": request_text,
        "predicted_sentiment": predicted_sentiment,
        "true_sentiment": true_sentiment,
    }

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # A simple in-process lock prevents concurrent worker threads from
        # interleaving writes to the JSON Lines file.
        with LOG_LOCK:
            with LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction succeeded, but the monitoring log could not be written.",
        ) from exc


@app.get("/health", response_model=HealthResponse, summary="Service health check")
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sentiment and record monitoring feedback",
)
def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    predicted_sentiment = predict_label(request.text)
    append_prediction_log(
        request_text=request.text,
        predicted_sentiment=predicted_sentiment,
        true_sentiment=request.true_sentiment,
    )
    return PredictionResponse(sentiment=predicted_sentiment)
