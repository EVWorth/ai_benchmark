"""Abstract base class for inference backend runners."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class InferenceResult:
    """Single inference result from a backend."""

    model: str
    backend: str
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class BaseRunner(ABC):
    """Interface every backend runner must implement."""

    def __init__(self, host: str, timeout: int = 120) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, model: str, prompt: str, **kwargs) -> InferenceResult:
        """Run *prompt* against *model* and return a timed result."""
        t0 = time.perf_counter()
        try:
            raw = self._infer(model, prompt, **kwargs)
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - t0
            return InferenceResult(
                model=model,
                backend=self.backend_name,
                prompt=prompt,
                response="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                elapsed_seconds=elapsed,
                tokens_per_second=0.0,
                error=str(exc),
            )
        elapsed = time.perf_counter() - t0

        tps = raw["completion_tokens"] / elapsed if elapsed > 0 else 0.0
        return InferenceResult(
            model=model,
            backend=self.backend_name,
            prompt=prompt,
            response=raw["response"],
            prompt_tokens=raw["prompt_tokens"],
            completion_tokens=raw["completion_tokens"],
            total_tokens=raw["prompt_tokens"] + raw["completion_tokens"],
            elapsed_seconds=elapsed,
            tokens_per_second=tps,
            extra=raw.get("extra", {}),
        )

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short lowercase name of the backend (e.g. 'ollama')."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names from the backend."""

    @abstractmethod
    def _infer(self, model: str, prompt: str, **kwargs) -> dict:
        """
        Run inference and return a dict with keys:
          response (str), prompt_tokens (int), completion_tokens (int),
          extra (dict, optional).
        """
