"""Tests for the scoring module."""
import pytest

from benchmark.scoring import score_response, build_scores, aggregate_scores, PromptScore


class TestScoreResponse:
    def test_exact_number_match(self):
        assert score_response("408", "408", "exact_number") is True

    def test_exact_number_with_spaces(self):
        assert score_response("  408  ", "408", "exact_number") is True

    def test_exact_number_with_commas(self):
        assert score_response("300,000,000", "300000000", "exact_number") is True

    def test_exact_number_wrong(self):
        assert score_response("409", "408", "exact_number") is False

    def test_exact_number_from_sentence(self):
        # Should extract the first number in the response
        assert score_response("The answer is 12.", "12", "exact_number") is True

    def test_contains_word_yes(self):
        assert score_response("YES", "YES", "contains_word") is True

    def test_contains_word_case_insensitive(self):
        assert score_response("yes, that is correct", "YES", "contains_word") is True

    def test_contains_word_partial_no_match(self):
        # "YESTERDAY" should not match whole-word "YES"
        assert score_response("YESTERDAY", "YES", "contains_word") is False

    def test_contains_word_au(self):
        assert score_response("Au", "Au", "contains_word") is True

    def test_contains_phrase(self):
        assert score_response("def factorial(n):\n    ...", "def factorial", "contains_phrase") is True

    def test_contains_phrase_missing(self):
        assert score_response("def fact(n): pass", "def factorial", "contains_phrase") is False

    def test_contains_phrase_case_insensitive(self):
        assert score_response("O(LOG N)", "O(log n)", "contains_phrase") is True

    def test_exact_word_match(self):
        assert score_response("Python", "Python", "exact_word") is True

    def test_exact_word_mismatch(self):
        assert score_response("python language", "Python", "exact_word") is False

    def test_empty_response(self):
        assert score_response("", "408", "exact_number") is False

    def test_no_number_in_response(self):
        assert score_response("I don't know", "12", "exact_number") is False


class TestBuildScores:
    def test_basic_scoring(self, benchmark_prompts):
        results = [
            {"prompt_id": "math_001", "response": "408", "error": None},
            {"prompt_id": "reasoning_001", "response": "YES", "error": None},
        ]
        scores = build_scores(results, benchmark_prompts)
        assert len(scores) == 2
        assert scores[0].passed is True
        assert scores[1].passed is True

    def test_failed_prompt(self, benchmark_prompts):
        results = [
            {"prompt_id": "math_001", "response": "999", "error": None},
        ]
        scores = build_scores(results, benchmark_prompts)
        assert scores[0].passed is False

    def test_error_response_fails(self, benchmark_prompts):
        results = [
            {"prompt_id": "math_001", "response": "408", "error": "Connection refused"},
        ]
        scores = build_scores(results, benchmark_prompts)
        assert scores[0].passed is False

    def test_missing_prompt_id_skipped(self, benchmark_prompts):
        results = [
            {"prompt_id": "nonexistent_999", "response": "408", "error": None},
        ]
        scores = build_scores(results, benchmark_prompts)
        assert len(scores) == 0

    def test_response_truncated_in_score(self, benchmark_prompts):
        long_response = "Au " + "x" * 300
        results = [
            {"prompt_id": "knowledge_001", "response": long_response, "error": None},
        ]
        scores = build_scores(results, benchmark_prompts)
        assert len(scores[0].actual) <= 200


class TestAggregateScores:
    def test_all_pass(self):
        scores = [
            PromptScore("a", "math", "1", "1", "exact_number", True, 1.0),
            PromptScore("b", "math", "2", "2", "exact_number", True, 1.0),
        ]
        agg = aggregate_scores(scores)
        assert agg["overall"] == 1.0
        assert agg["by_category"]["math"] == 1.0

    def test_half_pass(self):
        scores = [
            PromptScore("a", "math", "1", "1", "exact_number", True, 1.0),
            PromptScore("b", "math", "2", "9", "exact_number", False, 0.0),
        ]
        agg = aggregate_scores(scores)
        assert agg["overall"] == 0.5

    def test_multiple_categories(self):
        scores = [
            PromptScore("a", "math", "1", "1", "exact_number", True, 1.0),
            PromptScore("b", "reasoning", "YES", "YES", "contains_word", True, 1.0),
            PromptScore("c", "reasoning", "YES", "NO", "contains_word", False, 0.0),
        ]
        agg = aggregate_scores(scores)
        assert agg["by_category"]["math"] == 1.0
        assert agg["by_category"]["reasoning"] == 0.5
        assert abs(agg["overall"] - 2 / 3) < 1e-9

    def test_empty_scores(self):
        agg = aggregate_scores([])
        assert agg["overall"] == 0.0
        assert agg["by_category"] == {}
