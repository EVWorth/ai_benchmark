"""Tests for the engine module (run_benchmark, summarise_run)."""

from unittest.mock import MagicMock, patch

import pytest

from benchmark.runners.base import InferenceResult
from benchmark.engine import run_benchmark, summarise_run, load_prompts


def _make_result(prompt_id, response, completion_tokens=10, tps=40.0):
    r = InferenceResult(
        model="test-model",
        backend="ollama",
        prompt="test prompt",
        response=response,
        prompt_tokens=5,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 5,
        elapsed_seconds=completion_tokens / tps,
        tokens_per_second=tps,
        extra={"prompt_id": prompt_id},
    )
    return r


class TestRunBenchmark:
    def test_run_benchmark_calls_runner(self, benchmark_prompts):
        mock_runner = MagicMock()
        mock_runner.backend_name = "ollama"

        def fake_run(model, prompt, **kwargs):
            return _make_result("math_001", "408")

        mock_runner.run.side_effect = fake_run

        results, scores = run_benchmark(
            runner=mock_runner,
            model="test-model",
            prompts=benchmark_prompts,
        )

        # Should have called run() once per prompt
        assert mock_runner.run.call_count == len(benchmark_prompts)
        assert len(results) == len(benchmark_prompts)
        # Scores should be produced for each prompt
        assert len(scores) == len(benchmark_prompts)

    def test_run_benchmark_scores_correctly(self, benchmark_prompts):
        mock_runner = MagicMock()
        mock_runner.backend_name = "ollama"

        correct_answers = {
            "math_001": "408",
            "reasoning_001": "YES",
            "knowledge_001": "Au",
            "coding_001": "def factorial(n): pass",
        }

        def fake_run(model, prompt, **kwargs):
            pid = benchmark_prompts[mock_runner.run.call_count - 1]["id"]
            return _make_result(pid, correct_answers.get(pid, "wrong"))

        mock_runner.run.side_effect = fake_run

        results, scores = run_benchmark(
            runner=mock_runner,
            model="test-model",
            prompts=benchmark_prompts,
        )

        passed = sum(1 for s in scores if s.passed)
        assert passed == len(benchmark_prompts)

    def test_prompt_id_attached_to_extra(self, benchmark_prompts):
        mock_runner = MagicMock()
        mock_runner.backend_name = "ollama"

        call_counter = {"n": 0}

        def fake_run(model, prompt, **kwargs):
            idx = call_counter["n"]
            call_counter["n"] += 1
            return _make_result(benchmark_prompts[idx]["id"], "some answer")

        mock_runner.run.side_effect = fake_run

        results, _ = run_benchmark(
            runner=mock_runner,
            model="test-model",
            prompts=benchmark_prompts,
        )

        for result, prompt_meta in zip(results, benchmark_prompts):
            assert result.extra.get("prompt_id") == prompt_meta["id"]


class TestSummariseRun:
    def test_summary_structure(self, benchmark_prompts):
        from benchmark.scoring import build_scores, PromptScore

        results = [_make_result(p["id"], "408", tps=50.0) for p in benchmark_prompts]
        raw = [
            {"prompt_id": r.extra["prompt_id"], "response": r.response, "error": None}
            for r in results
        ]
        scores = build_scores(raw, benchmark_prompts)

        summary = summarise_run(results, scores, "test-model", "ollama")

        assert summary["model"] == "test-model"
        assert summary["backend"] == "ollama"
        assert summary["avg_tps"] > 0
        assert 0.0 <= summary["quality_score"] <= 1.0
        assert summary["total_prompts"] == len(benchmark_prompts)
        assert summary["errors"] == 0

    def test_summary_error_count(self, benchmark_prompts):
        from benchmark.scoring import PromptScore

        results = []
        for p in benchmark_prompts:
            r = InferenceResult(
                model="test-model",
                backend="ollama",
                prompt="prompt",
                response="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                elapsed_seconds=1.0,
                tokens_per_second=0.0,
                error="Connection refused",
                extra={"prompt_id": p["id"]},
            )
            results.append(r)

        raw = [
            {"prompt_id": r.extra["prompt_id"], "response": r.response, "error": r.error}
            for r in results
        ]
        from benchmark.scoring import build_scores
        scores = build_scores(raw, benchmark_prompts)

        summary = summarise_run(results, scores, "test-model", "ollama")
        assert summary["errors"] == len(benchmark_prompts)
        assert summary["avg_tps"] == 0.0


class TestLoadPrompts:
    def test_loads_all_prompts(self):
        prompts = load_prompts()
        assert len(prompts) >= 20

    def test_category_filter(self):
        math_prompts = load_prompts(categories=["math"])
        assert all(p["category"] == "math" for p in math_prompts)
        assert len(math_prompts) > 0

    def test_multiple_category_filter(self):
        prompts = load_prompts(categories=["math", "coding"])
        cats = {p["category"] for p in prompts}
        assert cats == {"math", "coding"}

    def test_unknown_category_returns_empty(self):
        prompts = load_prompts(categories=["nonexistent_category"])
        assert prompts == []
