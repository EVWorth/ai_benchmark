"""Pytest configuration and shared fixtures."""
import pytest


@pytest.fixture()
def benchmark_prompts():
    """Return a small subset of prompts for fast testing."""
    return [
        {
            "id": "math_001",
            "category": "math",
            "prompt": "What is 17 multiplied by 24? Answer with only the number.",
            "expected": "408",
            "match_type": "exact_number",
        },
        {
            "id": "reasoning_001",
            "category": "reasoning",
            "prompt": "All roses are flowers. Do roses need water? Answer only YES or NO.",
            "expected": "YES",
            "match_type": "contains_word",
        },
        {
            "id": "knowledge_001",
            "category": "knowledge",
            "prompt": "What is the chemical symbol for gold? Answer with only the symbol.",
            "expected": "Au",
            "match_type": "contains_word",
        },
        {
            "id": "coding_001",
            "category": "coding",
            "prompt": "Write a Python factorial function.",
            "expected": "def factorial",
            "match_type": "contains_phrase",
        },
    ]
