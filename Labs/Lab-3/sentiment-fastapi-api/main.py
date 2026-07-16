"""FastAPI backend for movie-review sentiment analysis.

The application loads the TF-IDF + Multinomial Naive Bayes pipeline created
in Assignment 1 and exposes four endpoints required by the assignment:

- GET  /health
- POST /predict
- POST /predict_proba
- GET  /example
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Final

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Application files
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODEL_PATH: Final[Path] = BASE_DIR / "sentiment_model.pkl"
DATASET_PATH: Final[Path] = BASE_DIR / "IMDB Dataset.csv"


# ---------------------------------------------------------------------------
# Asset loading and validation
# ---------------------------------------------------------------------------
def load_model(model_path: Path) -> Pipeline:
    """Load and validate the trained sentiment-analysis pipeline."""
    if not model_path.is_file():
        raise RuntimeError(f"Required model file not found: {model_path.name}")

    try:
        loaded_model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - startup failure path
        raise RuntimeError(f"Unable to load {model_path.name}: {exc}") from exc

    if not hasattr(loaded_model, "predict"):
        raise RuntimeError("The loaded model does not provide predict().")
    if not hasattr(loaded_model, "predict_proba"):
        raise RuntimeError("The loaded model does not provide predict_proba().")
    if not hasattr(loaded_model, "classes_"):
        raise RuntimeError("The loaded model does not expose classes_.")

    return loaded_model


def load_training_reviews(dataset_path: Path) -> list[str]:
    """Load the review column used by the /example endpoint."""
    if not dataset_path.is_file():
        raise RuntimeError(f"Required dataset file not found: {dataset_path.name}")

    try:
        dataset = pd.read_csv(dataset_path, usecols=["review"])
    except ValueError as exc:
        raise RuntimeError(
            f"{dataset_path.name} must contain a 'review' column."
        ) from exc
    except Exception as exc:  # pragma: no cover - startup failure path
        raise RuntimeError(f"Unable to load {dataset_path.name}: {exc}") from exc

    reviews = dataset["review"].dropna().astype(str).tolist()
    if not reviews:
        raise RuntimeError(f"{dataset_path.name} contains no usable reviews.")

    return reviews


MODEL: Final[Pipeline] = load_model(MODEL_PATH)
TRAINING_REVIEWS: Final[list[str]] = load_training_reviews(DATASET_PATH)


# ---------------------------------------------------------------------------
# FastAPI configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Movie Review Sentiment API",
    description=(
        "A containerized FastAPI backend that classifies IMDB movie reviews "
        "as positive or negative."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------
class SentimentRequest(BaseModel):
    """JSON request body accepted by the prediction endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        description="Movie-review text to classify.",
        examples=["This movie was a masterpiece!"],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text must contain at least one non-whitespace character.")
        return cleaned


class HealthResponse(BaseModel):
    status: str


class SentimentResponse(BaseModel):
    sentiment: str


class ProbabilityResponse(BaseModel):
    sentiment: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ExampleResponse(BaseModel):
    review: str


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_label(text: str) -> str:
    """Return the model's predicted sentiment label."""
    try:
        return str(MODEL.predict([text])[0]).lower()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model could not generate a prediction.",
        ) from exc


def predicted_class_probability(text: str, predicted_label: str) -> float:
    """Return the probability corresponding to the predicted class."""
    try:
        class_labels = [str(label).lower() for label in MODEL.classes_]
        predicted_index = class_labels.index(predicted_label)
        probabilities = MODEL.predict_proba([text])[0]
        return float(probabilities[predicted_index])
    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model returned an unexpected class configuration.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model could not generate class probabilities.",
        ) from exc


# ---------------------------------------------------------------------------
# Required API endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Confirm that the API is running",
)
def health_check() -> HealthResponse:
    """Return a simple service-health response."""
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=SentimentResponse,
    summary="Predict the sentiment of a movie review",
)
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    """Classify the supplied review as positive or negative."""
    sentiment = predict_label(request.text)
    return SentimentResponse(sentiment=sentiment)


@app.post(
    "/predict_proba",
    response_model=ProbabilityResponse,
    summary="Predict sentiment and confidence",
)
def predict_sentiment_with_probability(
    request: SentimentRequest,
) -> ProbabilityResponse:
    """Return the predicted sentiment and its predicted-class probability."""
    sentiment = predict_label(request.text)
    probability = predicted_class_probability(request.text, sentiment)
    return ProbabilityResponse(sentiment=sentiment, probability=probability)


@app.get(
    "/example",
    response_model=ExampleResponse,
    summary="Return a random review from the training dataset",
)
def get_training_example() -> ExampleResponse:
    """Return one random review for convenient API testing."""
    return ExampleResponse(review=random.choice(TRAINING_REVIEWS))
