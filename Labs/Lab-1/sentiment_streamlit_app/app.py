"""
app.py

Streamlit web app for movie review sentiment analysis.

This app loads a saved scikit-learn Pipeline from sentiment_model.pkl and uses it
to classify user-entered movie review text as positive or negative.

Run from the project folder:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILE: Final[Path] = Path("sentiment_model.pkl")
POSITIVE_LABEL: Final[str] = "positive"
NEGATIVE_LABEL: Final[str] = "negative"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sentiment model...")
def load_model(model_path: Path) -> Pipeline:
    """Load the trained sentiment analysis pipeline from disk.

    The assignment specifically asks for @st.cache_data. Caching prevents
    Streamlit from reloading the model every time the user interacts with the
    app, which improves responsiveness.

    Parameters
    ----------
    model_path:
        Path to the saved joblib model artifact.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The trained TF-IDF + Naive Bayes pipeline.

    Raises
    ------
    FileNotFoundError
        If sentiment_model.pkl has not been created yet.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            "sentiment_model.pkl was not found. Run `python train_model.py` first."
        )

    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_sentiment(model: Pipeline, text: str) -> tuple[str, pd.DataFrame]:
    """Predict sentiment and class probabilities for one text input.

    Parameters
    ----------
    model:
        Trained scikit-learn Pipeline loaded from sentiment_model.pkl.
    text:
        User-provided movie review text.

    Returns
    -------
    tuple[str, pd.DataFrame]
        The predicted sentiment label and a probability DataFrame indexed by
        sentiment class.
    """
    # scikit-learn text pipelines expect an iterable of documents.
    # Since we are predicting one review, we wrap the string inside a list.
    predicted_label = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    probability_df = pd.DataFrame(
        {
            "Sentiment": model.classes_,
            "Probability": probabilities,
        }
    ).set_index("Sentiment")

    return predicted_label, probability_df


def display_prediction(predicted_label: str, probability_df: pd.DataFrame) -> None:
    """Render the prediction result and probability chart in Streamlit.

    Parameters
    ----------
    predicted_label:
        The predicted sentiment class, usually "positive" or "negative".
    probability_df:
        DataFrame containing class probabilities.
    """
    predicted_probability = probability_df.loc[predicted_label, "Probability"]

    st.subheader("Prediction Result")

    if predicted_label == POSITIVE_LABEL:
        st.success(f"Predicted Sentiment: Positive 👍")
    elif predicted_label == NEGATIVE_LABEL:
        st.error(f"Predicted Sentiment: Negative 👎")
    else:
        # Defensive fallback in case model labels differ unexpectedly.
        st.info(f"Predicted Sentiment: {predicted_label}")

    st.write(f"Model confidence: **{predicted_probability:.2%}**")

    st.subheader("Prediction Probabilities")
    st.bar_chart(probability_df)


# ---------------------------------------------------------------------------
# Streamlit page layout
# ---------------------------------------------------------------------------
def main() -> None:
    """Build and run the Streamlit user interface."""
    st.set_page_config(
        page_title="Movie Review Sentiment Analyzer",
        page_icon="🎬",
        layout="centered",
    )

    st.title("🎬 Movie Review Sentiment Analyzer")

    st.write(
        "This app predicts whether a movie review is positive or negative. "
        "It uses a TF-IDF vectorizer and a Multinomial Naive Bayes classifier "
        "trained on the IMDB movie review dataset."
    )

    st.divider()

    # Load model after rendering the basic page so any file error can be shown
    # clearly inside the Streamlit interface.
    try:
        model = load_model(MODEL_FILE)
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python train_model.py", language="bash")
        st.stop()

    review_text = st.text_area(
        label="Enter a movie review to analyze:",
        height=180,
        placeholder=(
            "Example: The acting was excellent, the story was engaging, "
            "and the ending was very satisfying."
        ),
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked:
        cleaned_text = review_text.strip()

        if not cleaned_text:
            st.warning("Please enter a movie review before clicking Analyze.")
            st.stop()

        predicted_label, probability_df = predict_sentiment(model, cleaned_text)
        display_prediction(predicted_label, probability_df)

    st.divider()

    with st.expander("How this app works"):
        st.write(
            "1. The training script loads the IMDB reviews dataset.\n"
            "2. Text reviews are transformed into TF-IDF features.\n"
            "3. A Multinomial Naive Bayes classifier learns to classify sentiment.\n"
            "4. The trained pipeline is saved as `sentiment_model.pkl`.\n"
            "5. This Streamlit app loads the saved pipeline and uses it for inference."
        )


if __name__ == "__main__":
    main()
