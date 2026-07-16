# Movie Review Sentiment API

A dedicated FastAPI backend that serves the sentiment-analysis model created in Assignment 1. The model is a scikit-learn pipeline combining TF-IDF text vectorization with a Multinomial Naive Bayes classifier trained on the **IMDB Dataset of 50K Movie Reviews**.

The application is packaged as a Docker container and exposes four REST API endpoints for health checking, sentiment prediction, confidence scoring, and retrieving a random training review.

## Project Structure

```text
.
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
├── main.py
├── requirements.txt
├── sentiment_model.pkl
└── IMDB Dataset.csv
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Confirm that the API is running |
| `POST` | `/predict` | Predict whether supplied review text is positive or negative |
| `POST` | `/predict_proba` | Return the predicted sentiment and predicted-class probability |
| `GET` | `/example` | Return a random review from the original IMDB dataset |

## Request and Response Examples

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Request body:

```json
{
  "text": "This movie was a masterpiece!"
}
```

Example response:

```json
{
  "sentiment": "positive"
}
```

### `POST /predict_proba`

Request body:

```json
{
  "text": "The acting was excellent and the story was moving."
}
```

Example response:

```json
{
  "sentiment": "positive",
  "probability": 0.95
}
```

The probability value depends on the supplied review.

### `GET /example`

Example response:

```json
{
  "review": "A randomly selected review from the IMDB dataset."
}
```

## Prerequisites

- Docker installed and running
- Git installed
- Make installed, if using the provided Makefile

No local Python environment is required when the application is run through Docker.

## Build and Run with Make

### 1. Build the Docker image

```bash
make build
```

This creates an image named `sentiment-fastapi-api`.

### 2. Run the container

```bash
make run
```

The API will be available at:

```text
http://localhost:8000
```

### 3. Open the generated FastAPI documentation

```text
http://localhost:8000/docs
```

The Swagger interface can be used to inspect and execute all four endpoints.

### 4. Stop the container

Press `Ctrl+C` in the terminal running the container.

### 5. Remove the Docker image

```bash
make clean
```

## Manual Docker Commands

Build:

```bash
docker build -t sentiment-fastapi-api .
```

Run:

```bash
docker run --rm -p 8000:8000 sentiment-fastapi-api
```

Remove the image:

```bash
docker rmi -f sentiment-fastapi-api
```

## Optional Local Python Execution

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and start the API:

```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Test with cURL

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie was a masterpiece!"}'
```

Prediction with probability:

```bash
curl -X POST \
  http://localhost:8000/predict_proba \
  -H "Content-Type: application/json" \
  -d '{"text":"The acting was terrible and the story was boring."}'
```

Training example:

```bash
curl http://localhost:8000/example
```

## Postman Desktop Testing

Create four requests in Postman Desktop:

1. `GET http://localhost:8000/health`
2. `POST http://localhost:8000/predict`
3. `POST http://localhost:8000/predict_proba`
4. `GET http://localhost:8000/example`

For each POST request, choose **Body → raw → JSON** and use:

```json
{
  "text": "This movie was a masterpiece!"
}
```

Also test validation by sending an empty body or blank `text`. FastAPI and Pydantic should return `422 Unprocessable Entity`.