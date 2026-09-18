"""Run the LunorAI RAG evaluation set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT_DIR / "evaluation" / "questions.json"

API_URL = "http://127.0.0.1:8000/api/chat"

FALLBACK_ANSWER = (
    "I couldn't find that information in the knowledge base."
)


def load_questions() -> list[dict]:
    """Load evaluation questions from JSON."""

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "questions.json must contain a list."
        )

    return questions


def evaluate_question(question_data: dict) -> dict:
    """Evaluate one question against the running API."""

    question_id = question_data["id"]
    question = question_data["question"]
    question_type = question_data["type"]

    try:
        response = requests.post(
            API_URL,
            json={
                "question": question,
            },
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "answer",
            "",
        )

        sources = result.get(
            "sources",
            [],
        )

        if question_type == "answerable":
            passed = (
                bool(answer.strip())
                and answer.strip() != FALLBACK_ANSWER
                and bool(sources)
            )

            reason = (
                "Answer returned with retrieved sources."
                if passed
                else "Missing answer or question was incorrectly rejected."
            )

        elif question_type == "unanswerable":
            passed = (
                answer.strip()
                == FALLBACK_ANSWER
            )

            reason = (
                "Correctly rejected unsupported information."
                if passed
                else "Model returned unsupported information."
            )

        else:
            passed = False
            reason = (
                f"Unknown question type: {question_type}"
            )

        return {
            "id": question_id,
            "question": question,
            "type": question_type,
            "passed": passed,
            "answer": answer,
            "source_count": len(sources),
            "reason": reason,
        }

    except requests.RequestException as exc:
        return {
            "id": question_id,
            "question": question,
            "type": question_type,
            "passed": False,
            "answer": "",
            "source_count": 0,
            "reason": f"API request failed: {exc}",
        }

    except Exception as exc:
        return {
            "id": question_id,
            "question": question,
            "type": question_type,
            "passed": False,
            "answer": "",
            "source_count": 0,
            "reason": f"Evaluation failed: {exc}",
        }


def print_result(result: dict) -> None:
    """Print one evaluation result."""

    status = "PASS" if result["passed"] else "FAIL"

    print(
        f"[{status}] "
        f"{result['id']} "
        f"({result['type']})"
    )

    print(
        f"  Q: {result['question']}"
    )

    print(
        f"  A: {result['answer']}"
    )

    print(
        f"  Sources: {result['source_count']}"
    )

    print(
        f"  {result['reason']}"
    )

    print()


def main() -> None:
    """Run the complete evaluation."""

    questions = load_questions()

    print()
    print("=" * 60)
    print("LunorAI Evaluation")
    print("=" * 60)
    print()

    results = []

    for question_data in questions:
        result = evaluate_question(
            question_data
        )

        results.append(result)

        print_result(result)

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    answerable = [
        result
        for result in results
        if result["type"] == "answerable"
    ]

    unanswerable = [
        result
        for result in results
        if result["type"] == "unanswerable"
    ]

    answerable_passed = sum(
        result["passed"]
        for result in answerable
    )

    unanswerable_passed = sum(
        result["passed"]
        for result in unanswerable
    )

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"Overall: "
        f"{passed}/{total} passed"
    )

    if answerable:
        print(
            f"Answerable: "
            f"{answerable_passed}/{len(answerable)} passed"
        )

    if unanswerable:
        print(
            f"Unanswerable: "
            f"{unanswerable_passed}/{len(unanswerable)} passed"
        )

    print()

    if passed == total:
        print("RESULT: PASS")
    else:
        print("RESULT: REVIEW REQUIRED")

    print()

    output_file = (
        ROOT_DIR
        / "evaluation"
        / "results.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "total": total,
                "passed": passed,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Detailed results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()