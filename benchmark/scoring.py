"""Scoring module: evaluates model responses against expected answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptScore:
    """Score for a single prompt response."""

    prompt_id: str
    category: str
    expected: str
    actual: str
    match_type: str
    passed: bool
    points: float  # 0.0 or 1.0 for now; can be fractional in future


def score_response(
    response: str,
    expected: str,
    match_type: str,
) -> bool:
    """Return True if *response* satisfies the *expected* check."""
    response_clean = response.strip()

    if match_type == "exact_number":
        return _match_number(response_clean, expected)
    elif match_type == "exact_word":
        return response_clean.lower() == expected.lower()
    elif match_type == "contains_word":
        # The response must contain the expected word as a whole word
        pattern = rf"\b{re.escape(expected)}\b"
        return bool(re.search(pattern, response_clean, re.IGNORECASE))
    elif match_type == "contains_phrase":
        return expected.lower() in response_clean.lower()
    else:
        # Fallback: substring match
        return expected.lower() in response_clean.lower()


def _match_number(response: str, expected: str) -> bool:
    """Extract the first number from *response* and compare to *expected*."""
    # Strip commas and underscores used as digit separators
    cleaned = response.replace(",", "").replace("_", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not numbers:
        return False
    try:
        actual_val = float(numbers[0])
        expected_val = float(expected)
        return abs(actual_val - expected_val) < 1e-6
    except ValueError:
        return False


def build_scores(
    results: list[dict],
    prompts: list[dict],
) -> list[PromptScore]:
    """
    Pair inference results with prompt metadata and return scored list.

    *results* is a list of dicts with keys: prompt_id, response, error.
    *prompts* is the loaded prompt catalogue (list of prompt dicts).
    """
    prompt_map = {p["id"]: p for p in prompts}
    scores: list[PromptScore] = []

    for result in results:
        pid = result.get("prompt_id", "")
        prompt_meta = prompt_map.get(pid)
        if prompt_meta is None:
            continue

        response = result.get("response", "")
        error = result.get("error")
        if error:
            response = ""

        passed = score_response(
            response,
            prompt_meta["expected"],
            prompt_meta["match_type"],
        )
        scores.append(
            PromptScore(
                prompt_id=pid,
                category=prompt_meta["category"],
                expected=prompt_meta["expected"],
                actual=response[:200],  # truncate long responses in report
                match_type=prompt_meta["match_type"],
                passed=passed,
                points=1.0 if passed else 0.0,
            )
        )

    return scores


def aggregate_scores(scores: list[PromptScore]) -> dict:
    """Return per-category and overall accuracy metrics."""
    if not scores:
        return {"overall": 0.0, "by_category": {}}

    categories: dict[str, list[float]] = {}
    for s in scores:
        categories.setdefault(s.category, []).append(s.points)

    by_cat = {cat: sum(pts) / len(pts) for cat, pts in categories.items()}
    overall = sum(s.points for s in scores) / len(scores)

    return {"overall": overall, "by_category": by_cat}
