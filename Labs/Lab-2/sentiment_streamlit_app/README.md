# Movie Review Sentiment Analyzer

## Project Description

This project is a Streamlit web application that predicts whether a movie review is positive or negative.

The application uses a trained scikit-learn pipeline saved as `model.pkl`. The model combines TF-IDF text vectorization with a Multinomial Naive Bayes classifier trained on the IMDB movie review dataset.

The goal of this assignment is to package the Streamlit application into a reproducible Docker container so that it can run consistently across different machines.

## Project Structure

```text
.
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
├── app.py
├── requirements.txt
└── model.pkl
