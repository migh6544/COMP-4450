# Assignment 5 - Model Monitoring

## Overview

This project implements the Assignment 5 multi-container MLOps monitoring architecture for an IMDB sentiment-classification model.

Two independent Docker containers are used:

1. **FastAPI Prediction Service** - serves `POST /predict` and logs every prediction plus ground-truth feedback.
2. **Streamlit Monitoring Dashboard** - reads the shared prediction log and displays model/data monitoring metrics.

The containers share:

- a **named Docker network** so the Streamlit service can reach the FastAPI service by container name, and
- a **named Docker volume** mounted at `/logs` so prediction activity persists and is visible to both services.

The project uses the supplied trained `sentiment_model.pkl`, the 50,000-row `IMDB Dataset.csv`, and the instructor-provided `test.json` evaluation data.

---

## Architecture

```text
                           POST /predict
                                |
                                v
                    +-----------------------+
                    | FastAPI container     |
                    | sentiment-api         |
                    |                       |
                    | sentiment_model.pkl   |
                    +-----------+-----------+
                                |
                                | append one JSON object per line
                                v
                    +-----------------------+
                    | Named Docker volume   |
                    | sentiment-prediction- |
                    | logs                  |
                    |                       |
                    | /logs/                |
                    | prediction_logs.json  |
                    +-----------+-----------+
                                |
                                | read-only mount
                                v
                    +-----------------------+
                    | Streamlit container   |
                    | sentiment-monitoring  |
                    |                       |
                    | IMDB Dataset.csv      |
                    +-----------------------+
                       |       |        |
                       v       v        v
                     Data    Target   Accuracy /
                     Drift   Drift    Precision
                                      + <80% alert
```

Both containers are attached to `sentiment-monitoring-network`. The dashboard also calls the API's `/health` endpoint over that network.

---

## Project Structure

```text
assignment-5-model-monitoring/
├── api/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .dockerignore
│   └── sentiment_model.pkl
├── monitoring/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .dockerignore
│   └── IMDB Dataset.csv
├── evaluate.py
├── test.json
├── requirements.txt
├── Makefile
├── README.md
└── .gitignore
```

---

## Monitoring Log Format

Every successful call to `POST /predict` appends one JSON object as a new line in:

```text
/logs/prediction_logs.json
```

Example log entry:

```json
{"timestamp":"2026-08-10T04:20:31.385212+00:00","request_text":"This movie was fantastic.","predicted_sentiment":"positive","true_sentiment":"positive"}
```

This is a **JSON Lines / NDJSON** file: each line is an independent JSON object.

The required fields are:

- `timestamp`
- `request_text`
- `predicted_sentiment`
- `true_sentiment`

Because this assignment does not include a feedback frontend, `true_sentiment` is supplied in the POST request from Postman, curl, or `evaluate.py`.

---

## Prerequisites

Install:

- Docker Desktop / Docker Engine
- GNU Make
- Python 3.10+ if running `evaluate.py` directly from the host

Verify Docker:

```bash
docker --version
docker info
```

---

## Build the Two Docker Images

From the repository root:

```bash
make build
```

This builds:

- `assignment5-sentiment-api`
- `assignment5-monitoring-dashboard`

---

## Run the Full Stack

```bash
make run
```

The Makefile will:

1. Create the shared Docker network if it does not exist.
2. Create the named log volume if it does not exist.
3. Start the FastAPI container.
4. Start the Streamlit monitoring container.
5. Mount the same named volume at `/logs` in both containers.

Open:

- FastAPI service: `http://localhost:8000`
- FastAPI Swagger docs: `http://localhost:8000/docs`
- Streamlit monitoring dashboard: `http://localhost:8501`

Check container status:

```bash
make status
```

Inspect container logs:

```bash
make logs
```

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Predict Sentiment and Record Feedback

The request must include both the review text and the true sentiment label.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "An absolute masterpiece. The acting was flawless.",
    "true_sentiment": "positive"
  }'
```

Example response:

```json
{"sentiment":"positive"}
```

A corresponding record is appended to the shared `/logs/prediction_logs.json` file.

Valid true labels are:

```text
positive
negative
```

The API normalizes their capitalization to lowercase.

---

## Postman Testing

Create a `POST` request to:

```text
http://localhost:8000/predict
```

Select **Body -> raw -> JSON** and use:

```json
{
  "text": "The movie was boring and predictable.",
  "true_sentiment": "negative"
}
```

Each request automatically becomes another monitoring record.

---

## Streamlit Monitoring Dashboard

Open:

```text
http://localhost:8501
```

The dashboard reads `/logs/prediction_logs.json` from the named Docker volume and implements the required monitoring views.

### 1. Data Drift Analysis

The monitored input feature is **review length in words**.

The dashboard compares:

- the word-count distribution of reviews in `IMDB Dataset.csv`, and
- the word-count distribution of logged inference requests.

The same word-count definition is used for both:

```python
len(text.split())
```

A density-normalized histogram makes the 50,000-row reference dataset comparable with the smaller inference sample.

### 2. Target / Prediction Drift Analysis

The dashboard compares:

- the sentiment-label distribution in the IMDB reference/training dataset, and
- the distribution of `predicted_sentiment` in the production logs.

The supplied IMDB dataset is balanced at 25,000 positive and 25,000 negative reviews, so its reference distribution is 50% / 50%.

### 3. Model Accuracy and User Feedback

Using the logged `true_sentiment` values, the dashboard calculates:

- **Accuracy** across all valid feedback records
- **Precision** for the `positive` class

### 4. Alerting

If accuracy falls below 80%, the dashboard displays a prominent `st.error()` alert near the top of the page.

If no feedback has been collected yet, the dashboard displays `N/A` and does not create a false alert.

---

## Run the Evaluation Script

The instructor-provided `test.json` is already included in the repository root.

Its records have the structure:

```json
[
  {
    "text": "An absolute masterpiece.",
    "true_label": "positive"
  }
]
```

`evaluate.py` maps `true_label` to the API's `true_sentiment` field so every evaluation request is also logged for monitoring.

### Install the host-side dependency

Recommended virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Execute

Make sure the Docker stack is already running:

```bash
make run
```

Then run:

```bash
python evaluate.py
```

or:

```bash
make evaluate
```

The script loops through every test record, sends it to:

```text
http://localhost:8000/predict
```

and prints the final score in the form:

```text
Final accuracy: 95.40% (166/174)
```

With the included `sentiment_model.pkl` and supplied `test.json`, the completed project was locally verified at **91.95% accuracy (160/174)**. Re-running the same deterministic scikit-learn pipeline should reproduce that result in the specified environment.

Optional arguments:

```bash
python evaluate.py --url http://localhost:8000/predict --test-file test.json
```

---

## Inspect the Shared Prediction Log

The named volume is:

```text
sentiment-prediction-logs
```

You can inspect its contents with a temporary container:

```bash
docker run --rm \
  -v sentiment-prediction-logs:/logs:ro \
  alpine cat /logs/prediction_logs.json
```

---

## Stop and Clean the Project

```bash
make clean
```

This:

- stops/removes both containers,
- removes `sentiment-monitoring-network`, and
- removes `sentiment-prediction-logs`.

**Important:** removing the named volume also deletes the persisted prediction logs.

To remove the two built Docker images as well:

```bash
make clean-all
```

---

## Fresh End-to-End Run

```bash
make build
make run
```

Test the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"A wonderful and moving film.","true_sentiment":"positive"}'
```

Run the full evaluation set:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py
```

Then open:

```text
http://localhost:8501
```

The evaluation requests will populate the dashboard automatically because the API writes them into the named log volume.

---

## GitHub Submission

The assignment requires a **new public GitHub repository** rather than reuse of a repository from a previous assignment.

Example setup after creating a new empty public repository on GitHub:

```bash
git init
git add .
git commit -m "Complete Assignment 5 model monitoring project"
git branch -M main
git remote add origin <YOUR-NEW-REPOSITORY-URL>
git push -u origin main
```

Do not commit local virtual-environment folders or runtime logs. The included `.gitignore` already excludes them.

---

## Assignment Requirement Checklist

- [x] FastAPI service in a separate `api/` directory
- [x] Required `POST /predict` endpoint
- [x] Every prediction logged to `/logs/prediction_logs.json`
- [x] `timestamp` logged
- [x] `request_text` logged
- [x] `predicted_sentiment` logged
- [x] `true_sentiment` logged
- [x] Streamlit service in a separate `monitoring/` directory
- [x] Streamlit reads the shared `/logs` volume
- [x] Data-drift histogram comparing IMDB and inference text lengths
- [x] Target/prediction-drift bar chart
- [x] Accuracy from feedback
- [x] Precision from feedback
- [x] `st.error()` alert below 80% accuracy
- [x] Root `evaluate.py`
- [x] Uses supplied `test.json`
- [x] Two separate Dockerfiles
- [x] Shared named Docker volume
- [x] Shared Docker network
- [x] Makefile `build`, `run`, and `clean`
- [x] README architecture and step-by-step instructions
- [x] curl examples
- [x] Evaluation-script instructions
- [ ] Create and submit a **new public GitHub repository**
