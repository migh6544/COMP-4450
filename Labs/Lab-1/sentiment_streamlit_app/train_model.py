"""
train_model.py

Train and save a sentiment analysis model for movie reviews.

Context:
- Input data: Kaggle "IMDB Dataset of 50K Movie Reviews"
- Expected file name: IMDB Dataset.csv
- Expected columns: review, sentiment
- Model approach: TF-IDF text vectorization + Multinomial Naive Bayes
- Output artifact: sentiment_model.pkl

Run once from the project folder:
    python train_model.py

After this script finishes, run the Streamlit app:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Constants
# Keeping file paths as constants makes the script easier to update and debug.
DATA_FILE: Final[Path] = Path("IMDB Dataset.csv")
MODEL_FILE: Final[Path] = Path("sentiment_model.pkl")

# Dataset should contain these exact columns.
REQUIRED_COLUMNS: Final[set[str]] = {"review", "sentiment"}

# Limit vocabulary size so the saved model is reasonably sized for GitHub.
# 50,000 is large enough for a strong baseline while keeping the project practical.
MAX_FEATURES: Final[int] = 50_000


# Data loading and validation
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the IMDB reviews dataset.

    Parameters
    ----------
    csv_path:
        Path to the Kaggle CSV file. This file should be named "IMDB Dataset.csv" and placed in the same directory as this script.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing non-empty review and sentiment values.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist in the project folder.
    ValueError
        If required columns are missing or if the dataset becomes empty after
        removing missing values.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Download the Kaggle dataset file "
            "and place it in the same folder as train_model.py."
        )

    data = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required column(s): {sorted(missing_columns)}. "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}."
        )

    # Keep only the columns needed.
    data = data[["review", "sentiment"]].copy()

    # Remove rows where either the review text or label is missing.
    data = data.dropna(subset=["review", "sentiment"])

    # Normalize data types. This prevents unexpected errors if pandas infers
    # a non-string type for any row.
    data["review"] = data["review"].astype(str)
    data["sentiment"] = data["sentiment"].astype(str).str.lower().str.strip()

    # Optional defensive cleaning: keep only the two expected sentiment classes.
    data = data[data["sentiment"].isin(["positive", "negative"])]

    if data.empty:
        raise ValueError(
            "No valid rows remain after cleaning. Confirm the CSV contains "
            "'review' text and 'positive'/'negative' sentiment labels."
        )

    return data


# Model construction
def build_model_pipeline() -> Pipeline:
    """Create the sentiment analysis model pipeline.

    The pipeline keeps preprocessing and classification together. This is a
    best-practice pattern because the exact same text transformation used during
    training is automatically applied during inference in the Streamlit app.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A scikit-learn Pipeline with:
        1. TfidfVectorizer for text-to-numeric feature conversion.
        2. MultinomialNB for sentiment classification.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=MAX_FEATURES,
                    ngram_range=(1, 2),  # unigrams + bigrams improve text baseline quality
                ),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )


def train_and_save_model(data: pd.DataFrame, output_path: Path) -> None:
    """Train the sentiment model pipeline and save it to disk.

    Parameters
    ----------
    data:
        Cleaned DataFrame with "review" and "sentiment" columns.
    output_path:
        File path where the trained pipeline will be saved.
    """
    X = data["review"]
    y = data["sentiment"]

    model_pipeline = build_model_pipeline()

    print(f"Training model on {len(data):,} reviews...")
    model_pipeline.fit(X, y)

    print(f"Saving trained model pipeline to '{output_path}'...")
    joblib.dump(model_pipeline, output_path)

    print("Training complete.")
    print("Next step: run the app with `streamlit run app.py`.")


def main() -> None:
    """Main script entry point."""
    print("Loading dataset...")
    data = load_dataset(DATA_FILE)

    print("Dataset loaded successfully.")
    print(f"Class distribution:\n{data['sentiment'].value_counts().to_string()}")

    train_and_save_model(data, MODEL_FILE)


if __name__ == "__main__":
    main()
