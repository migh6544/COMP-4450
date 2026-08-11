"""Evaluate the running Assignment 5 FastAPI service with test.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests


DEFAULT_API_URL = "http://localhost:8000/predict"
DEFAULT_TEST_FILE = Path(__file__).resolve().parent / "test.json"
VALID_LABELS = {"positive", "negative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send every test.json record to the running sentiment API."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_API_URL,
        help=f"Prediction endpoint URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="Path to the evaluation JSON file (default: ./test.json)",
    )
    return parser.parse_args()


def load_test_data(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Test file not found: {path}")

    with path.open("r", encoding="utf-8") as test_file:
        records = json.load(test_file)

    if not isinstance(records, list) or not records:
        raise ValueError("The test file must contain a non-empty JSON list.")

    validated: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")

        text = str(record.get("text", "")).strip()
        true_label = str(record.get("true_label", "")).strip().lower()
        if not text:
            raise ValueError(f"Record {index} has missing/blank text.")
        if true_label not in VALID_LABELS:
            raise ValueError(
                f"Record {index} has invalid true_label={true_label!r}; "
                "expected 'positive' or 'negative'."
            )
        validated.append({"text": text, "true_label": true_label})

    return validated


def main() -> int:
    args = parse_args()

    try:
        records = load_test_data(args.test_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    correct = 0
    session = requests.Session()

    print(f"Evaluating {len(records)} reviews against {args.url}\n")

    for index, item in enumerate(records, start=1):
        payload = {
            "text": item["text"],
            # Assignment 5 requires ground-truth feedback to be written to
            # prediction_logs.json, so map test.json's true_label to the API field.
            "true_sentiment": item["true_label"],
        }

        try:
            response = session.post(args.url, json=payload, timeout=15)
            response.raise_for_status()
            predicted = str(response.json()["sentiment"]).strip().lower()
        except (requests.RequestException, KeyError, ValueError) as exc:
            print(f"ERROR on record {index}: {exc}", file=sys.stderr)
            return 1

        is_correct = predicted == item["true_label"]
        correct += int(is_correct)
        marker = "PASS" if is_correct else "MISS"
        print(
            f"[{index:03d}/{len(records):03d}] {marker} | "
            f"expected={item['true_label']:<8} predicted={predicted}"
        )

    accuracy = correct / len(records)
    print("\n" + "=" * 60)
    print(f"Final accuracy: {accuracy:.2%} ({correct}/{len(records)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
