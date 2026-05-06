"""Ollama backend runner.

Supports both the native Ollama REST API and the OpenAI-compatible endpoint.
Default base URL: http://localhost:11434
"""

from __future__ import annotations

import requests

from .base import BaseRunner


class OllamaRunner(BaseRunner):
    """Runner for models served by Ollama (https://ollama.com)."""

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, host: str = DEFAULT_HOST, timeout: int = 120) -> None:
        super().__init__(host, timeout)

    @property
    def backend_name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------
    # BaseRunner implementation
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return model names available in Ollama."""
        url = f"{self.host}/api/tags"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def _infer(self, model: str, prompt: str, **kwargs) -> dict:
        """
        Call Ollama's /api/generate endpoint (non-streaming).
        Returns raw response dict normalised to BaseRunner schema.
        """
        url = f"{self.host}/api/generate"
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        # Pass through any extra options (temperature, num_predict, etc.)
        options = kwargs.pop("options", {})
        options.update(kwargs)
        if options:
            payload["options"] = options

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns eval_count (completion tokens) and prompt_eval_count
        completion_tokens = data.get("eval_count", 0)
        prompt_tokens = data.get("prompt_eval_count", 0)

        return {
            "response": data.get("response", ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "extra": {
                "eval_duration_ns": data.get("eval_duration"),
                "load_duration_ns": data.get("load_duration"),
                "total_duration_ns": data.get("total_duration"),
            },
        }
