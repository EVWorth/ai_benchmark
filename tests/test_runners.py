"""Tests for backend runners (mocked HTTP)."""

import pytest
import responses as resp_lib  # the `responses` library

from benchmark.runners.ollama_runner import OllamaRunner
from benchmark.runners.llamacpp_runner import LlamaCppRunner
from benchmark.runners.base import InferenceResult


OLLAMA_HOST = "http://localhost:11434"
LLAMACPP_HOST = "http://localhost:8080"


class TestOllamaRunner:
    @resp_lib.activate
    def test_successful_inference(self):
        resp_lib.add(
            resp_lib.POST,
            f"{OLLAMA_HOST}/api/generate",
            json={
                "response": "408",
                "eval_count": 3,
                "prompt_eval_count": 12,
                "eval_duration": 500_000_000,
                "load_duration": 100_000_000,
                "total_duration": 600_000_000,
            },
            status=200,
        )
        runner = OllamaRunner(host=OLLAMA_HOST)
        result = runner.run("llama3", "What is 17*24?")

        assert isinstance(result, InferenceResult)
        assert result.success is True
        assert result.response == "408"
        assert result.completion_tokens == 3
        assert result.prompt_tokens == 12
        assert result.tokens_per_second > 0
        assert result.backend == "ollama"

    @resp_lib.activate
    def test_server_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.POST,
            f"{OLLAMA_HOST}/api/generate",
            status=500,
            body="Internal Server Error",
        )
        runner = OllamaRunner(host=OLLAMA_HOST)
        result = runner.run("llama3", "hello")

        assert result.success is False
        assert result.error is not None

    @resp_lib.activate
    def test_list_models(self):
        resp_lib.add(
            resp_lib.GET,
            f"{OLLAMA_HOST}/api/tags",
            json={"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]},
            status=200,
        )
        runner = OllamaRunner(host=OLLAMA_HOST)
        models = runner.list_models()
        assert models == ["llama3:8b", "mistral:7b"]

    @resp_lib.activate
    def test_list_models_empty(self):
        resp_lib.add(
            resp_lib.GET,
            f"{OLLAMA_HOST}/api/tags",
            json={"models": []},
            status=200,
        )
        runner = OllamaRunner(host=OLLAMA_HOST)
        assert runner.list_models() == []

    @resp_lib.activate
    def test_options_forwarded(self):
        """Ensure temperature and num_predict are passed in the payload."""
        def request_callback(request):
            import json
            body = json.loads(request.body)
            assert body["options"]["temperature"] == 0.0
            assert body["options"]["num_predict"] == 128
            return (200, {}, json.dumps({
                "response": "ok",
                "eval_count": 1,
                "prompt_eval_count": 5,
            }))

        resp_lib.add_callback(
            resp_lib.POST,
            f"{OLLAMA_HOST}/api/generate",
            callback=request_callback,
            content_type="application/json",
        )
        runner = OllamaRunner(host=OLLAMA_HOST)
        result = runner.run("llama3", "hi", options={"temperature": 0.0, "num_predict": 128})
        assert result.success is True

    def test_backend_name(self):
        assert OllamaRunner().backend_name == "ollama"


class TestLlamaCppRunner:
    @resp_lib.activate
    def test_successful_inference(self):
        resp_lib.add(
            resp_lib.POST,
            f"{LLAMACPP_HOST}/completion",
            json={
                "content": "12",
                "tokens_evaluated": 10,
                "tokens_predicted": 2,
                "timings": {
                    "predicted_per_second": 45.2,
                    "prompt_per_second": 120.0,
                },
            },
            status=200,
        )
        runner = LlamaCppRunner(host=LLAMACPP_HOST)
        result = runner.run("my_model", "What is sqrt(144)?")

        assert result.success is True
        assert result.response == "12"
        assert result.completion_tokens == 2
        assert result.prompt_tokens == 10
        assert result.backend == "llamacpp"

    @resp_lib.activate
    def test_server_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.POST,
            f"{LLAMACPP_HOST}/completion",
            status=503,
        )
        runner = LlamaCppRunner(host=LLAMACPP_HOST)
        result = runner.run("my_model", "hello")

        assert result.success is False
        assert result.error is not None

    @resp_lib.activate
    def test_list_models_v1(self):
        resp_lib.add(
            resp_lib.GET,
            f"{LLAMACPP_HOST}/v1/models",
            json={"data": [{"id": "mistral-7b-q4"}]},
            status=200,
        )
        runner = LlamaCppRunner(host=LLAMACPP_HOST)
        assert runner.list_models() == ["mistral-7b-q4"]

    @resp_lib.activate
    def test_list_models_fallback_empty(self):
        # If /v1/models returns an error, should return []
        resp_lib.add(
            resp_lib.GET,
            f"{LLAMACPP_HOST}/v1/models",
            status=404,
        )
        runner = LlamaCppRunner(host=LLAMACPP_HOST)
        assert runner.list_models() == []

    def test_backend_name(self):
        assert LlamaCppRunner().backend_name == "llamacpp"

    @resp_lib.activate
    def test_tps_calculated(self):
        """TPS must be completion_tokens / elapsed_seconds."""
        resp_lib.add(
            resp_lib.POST,
            f"{LLAMACPP_HOST}/completion",
            json={
                "content": "hello world",
                "tokens_evaluated": 5,
                "tokens_predicted": 3,
                "timings": {},
            },
            status=200,
        )
        runner = LlamaCppRunner(host=LLAMACPP_HOST)
        result = runner.run("model", "say hi")
        # TPS = completion_tokens / elapsed; we can only verify it's > 0
        assert result.tokens_per_second > 0
