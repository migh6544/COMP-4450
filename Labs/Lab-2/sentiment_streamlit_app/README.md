# Movie Review Sentiment Analyzer

## Project Description

This project is a Streamlit web application that predicts whether a movie review is positive or negative.

The application uses a trained scikit-learn pipeline saved as `model.pkl`. The model combines TF-IDF text vectorization with a Multinomial Naive Bayes classifier trained on the IMDB movie review dataset.

The goal of this assignment is to package the Streamlit application into a reproducible Docker container so that it can run consistently across different machines.

## Project Structure

```text
.
├── .gitignore
├── .dockerignore
├── Dockerfile
├── Makefile
├── README.md
├── app.py
├── requirements.txt
├── model.pkl
└── train_model.py
```

The assignment-required files are:

```text
.gitignore
Dockerfile
Makefile
README.md
app.py
requirements.txt
model.pkl
```

The `train_model.py` file is included as a development file that documents how the model was trained. The raw dataset file `IMDB Dataset.csv` is not required to run the Dockerized app because the trained model is already included as `model.pkl`.

## Prerequisites

To run this project, you need:

- Docker installed
- Docker Desktop running
- Git installed, if cloning from GitHub
- Make installed, if using the provided Makefile

No local Python virtual environment is required when running the application through Docker.

## How to Run with Make

### 1. Clone the repository

```bash
git clone https://github.com/migh6544/COMP-4450.git
cd COMP-4450/Labs/Lab-2/sentiment_streamlit_app
```

### 2. Build the Docker image

```bash
make build
```

This builds a Docker image named `sentiment-app`.

### 3. Run the Docker container

```bash
make run
```

This starts the Streamlit app inside a Docker container and maps the container's port `8501` to your local machine's port `8501`.

### 4. Open the application

Visit this URL in your browser:

```text
http://localhost:8501
```

Enter a movie review and click the **Analyze** button.

### 5. Stop the application

Return to the terminal where the container is running and press:

```text
Ctrl + C
```

### 6. Clean up the Docker image

```bash
make clean
```

This removes the local Docker image named `sentiment-app`.

## Manual Docker Commands

If `make` is not available, use these Docker commands directly.

Build the image:

```bash
docker build -t sentiment-app .
```

Run the container:

```bash
docker run --rm -p 8501:8501 sentiment-app
```

Open the app in your browser:

```text
http://localhost:8501
```

Remove the image:

```bash
docker rmi -f sentiment-app
```

## Application Files

- `app.py` — Streamlit application for movie review sentiment prediction.
- `model.pkl` — trained scikit-learn sentiment analysis pipeline.
- `requirements.txt` — Python dependencies required by the application.
- `Dockerfile` — instructions for building the Docker image.
- `Makefile` — convenience commands for building, running, and cleaning the Docker image.
- `.gitignore` — prevents local environment files, cache files, and unnecessary artifacts from being committed.
- `.dockerignore` — prevents unnecessary local files from being copied into the Docker image.
- `train_model.py` — training script used to create `model.pkl`.

## Reproducibility Notes

This project includes the application code, dependency list, Docker configuration, Makefile, and trained model artifact. The Docker image packages the Python runtime, Python dependencies, Streamlit application code, and trained model together.

This supports reproducibility because another user can clone the repository and run the same application with the same dependencies using simple Make commands:

```bash
make build
make run
```

The app should then be available at:

```text
http://localhost:8501
```
