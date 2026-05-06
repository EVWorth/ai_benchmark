"""llama.cpp backend runner.

Connects to a llama.cpp server started with `llama-server` (or the
legacy `server` binary).  The server exposes an OpenAI-compatible API
at /v1/chat/completions as well as its native /completion endpoint.

Default base URL: http://localhost:8080
"""

from __future__ import annotations

import requests

from .base import BaseRunner


class LlamaCppRunner(BaseRunner):
    """Runner for models served by llama.cpp server."""

    DEFAULT_HOST = "http://localhost:8080"

    def __init__(self, host: str = DEFAULT_HOST, timeout: int = 120) -> None:
        super().__init__(host, timeout)

    @property
    def backend_name(self) -> str:
        return "llamacpp"

    # ------------------------------------------------------------------
    # BaseRunner implementation
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        llama.cpp server doesn't have a multi-model endpoint; return the
        currently loaded model name if available, otherwise an empty list.
        """
        try:
            url = f"{self.host}/v1/models"
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except requests.RequestException:
            return []

    def _infer(self, model: str, prompt: str, **kwargs) -> dict:
        """
        Use the /completion endpoint (non-streaming).
        The `model` parameter is informational only because llama.cpp
        server loads a single model at start time.
        """
        url = f"{self.host}/completion"
        payload: dict = {
            "prompt": prompt,
            "stream": False,
            "cache_prompt": False,
        }
        # Pass optional generation parameters (temperature, n_predict, etc.)
        payload.update(kwargs)

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        timings = data.get("timings", {})
        # tokens_evaluated = prompt tokens; tokens_predicted = completion tokens
        prompt_tokens = data.get("tokens_evaluated", 0)
        completion_tokens = data.get("tokens_predicted", 0)

        return {
            "response": data.get("content", ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "extra": {
                "predicted_per_second": timings.get("predicted_per_second"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "timings": timings,
            },
        }
