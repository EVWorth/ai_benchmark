"""Tests for the reporter module."""

import json
from pathlib import Path

import pytest

from benchmark.runners.base import InferenceResult
from benchmark.scoring import PromptScore, build_scores
from benchmark.reporter import save_json


def _make_result(prompt_id, response, tps=42.5):
    return InferenceResult(
        model="test-model",
        backend="ollama",
        prompt="test prompt",
        response=response,
        prompt_tokens=8,
        completion_tokens=4,
        total_tokens=12,
        elapsed_seconds=4 / tps,
        tokens_per_second=tps,
        extra={"prompt_id": prompt_id},
    )


@pytest.fixture()
def sample_results_and_scores(benchmark_prompts):
    correct_map = {
        "math_001": "408",
        "reasoning_001": "YES",
        "knowledge_001": "Au",
        "coding_001": "def factorial(n): pass",
    }
    results = [_make_result(p["id"], correct_map.get(p["id"], "wrong")) for p in benchmark_prompts]
    raw = [
        {"prompt_id": r.extra["prompt_id"], "response": r.response, "error": None}
        for r in results
    ]
    scores = build_scores(raw, benchmark_prompts)
    return results, scores


class TestSaveJson:
    def test_creates_file(self, tmp_path, sample_results_and_scores):
        results, scores = sample_results_and_scores
        out = tmp_path / "results.json"
        save_json(results, scores, "test-model", "ollama", out)
        assert out.exists()

    def test_json_structure(self, tmp_path, sample_results_and_scores):
        results, scores = sample_results_and_scores
        out = tmp_path / "results.json"
        save_json(results, scores, "test-model", "ollama", out)

        data = json.loads(out.read_text())
        assert data["model"] == "test-model"
        assert data["backend"] == "ollama"
        assert "timestamp" in data
        assert "summary" in data
        assert "results" in data
        assert "scores" in data

    def test_summary_fields(self, tmp_path, sample_results_and_scores):
        results, scores = sample_results_and_scores
        out = tmp_path / "results.json"
        save_json(results, scores, "test-model", "ollama", out)

        summary = json.loads(out.read_text())["summary"]
        assert "avg_tokens_per_second" in summary
        assert "quality_score" in summary
        assert "quality_by_category" in summary
        assert "total_prompts" in summary
        assert "passed" in summary
        assert "errors" in summary

    def test_avg_tps_is_positive(self, tmp_path, sample_results_and_scores):
        results, scores = sample_results_and_scores
        out = tmp_path / "results.json"
        save_json(results, scores, "test-model", "ollama", out)

        data = json.loads(out.read_text())
        assert data["summary"]["avg_tokens_per_second"] > 0

    def test_results_count_matches(self, tmp_path, sample_results_and_scores, benchmark_prompts):
        results, scores = sample_results_and_scores
        out = tmp_path / "results.json"
        save_json(results, scores, "test-model", "ollama", out)

        data = json.loads(out.read_text())
        assert len(data["results"]) == len(benchmark_prompts)
        assert len(data["scores"]) == len(benchmark_prompts)
