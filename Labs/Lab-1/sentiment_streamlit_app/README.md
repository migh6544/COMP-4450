# Movie Review Sentiment Analyzer

This project is a Streamlit web application that predicts whether a movie review is **positive** or **negative**. It trains a TF-IDF + Multinomial Naive Bayes model on the Kaggle IMDB Dataset of 50K Movie Reviews, saves the trained pipeline, and serves it through an interactive web app.

## Files included

- `train_model.py` — loads the IMDB dataset, trains the model pipeline, and saves `sentiment_model.pkl`
- `app.py` — Streamlit app for user input and sentiment prediction
- `requirements.txt` — Python libraries needed to run the project
- `.gitignore` — excludes local environment files and the Kaggle CSV dataset
- `README.md` — setup and run instructions

## How to run locally

- Clone the repository:

  ```bash
  git clone <your-github-repository-url>
  cd <your-repository-folder>
  ```

- Create a virtual environment:

  ```bash
  python -m venv .venv
  ```

- Activate the virtual environment:

  macOS/Linux:

  ```bash
  source .venv/bin/activate
  ```

  Windows:

  ```bash
  .venv\Scripts\activate
  ```

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Download the dataset from Kaggle:

  Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- Place the file below in the project folder:

  ```text
  IMDB Dataset.csv
  ```

- Train and save the model:

  ```bash
  python train_model.py
  ```

- Run the Streamlit app:

  ```bash
  streamlit run app.py
  ```

- Open the local URL shown in the terminal and enter a movie review to test the app.
