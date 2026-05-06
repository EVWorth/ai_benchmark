"""Core benchmark runner: orchestrates prompt execution and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .runners.base import BaseRunner, InferenceResult
from .scoring import PromptScore, build_scores

console = Console()

DEFAULT_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "benchmark_prompts.json"


def load_prompts(path: Optional[Path] = None, categories: Optional[list[str]] = None) -> list[dict]:
    """Load benchmark prompts from JSON file, optionally filtering by category."""
    prompts_path = path or DEFAULT_PROMPTS_PATH
    with open(prompts_path) as f:
        prompts = json.load(f)

    if categories:
        prompts = [p for p in prompts if p["category"] in categories]

    return prompts


def run_benchmark(
    runner: BaseRunner,
    model: str,
    prompts: Optional[list[dict]] = None,
    categories: Optional[list[str]] = None,
    prompts_path: Optional[Path] = None,
    generation_kwargs: Optional[dict] = None,
) -> tuple[list[InferenceResult], list[PromptScore]]:
    """
    Run all prompts against *model* using *runner*.

    Returns a tuple of (inference_results, prompt_scores).
    """
    if prompts is None:
        prompts = load_prompts(prompts_path, categories)

    gen_kwargs = generation_kwargs or {}
    results: list[InferenceResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Running {len(prompts)} prompts on {model}...", total=len(prompts)
        )

        for prompt_meta in prompts:
            progress.update(
                task,
                description=f"[cyan]{prompt_meta['id']}[/cyan]",
            )
            result = runner.run(model, prompt_meta["prompt"], **gen_kwargs)
            # Attach prompt_id into extra so the reporter can link them
            result.extra["prompt_id"] = prompt_meta["id"]
            results.append(result)
            progress.advance(task)

    # Build scored list
    raw_scores = [
        {
            "prompt_id": r.extra.get("prompt_id", ""),
            "response": r.response,
            "error": r.error,
        }
        for r in results
    ]
    scores = build_scores(raw_scores, prompts)

    return results, scores


def summarise_run(
    results: list[InferenceResult],
    scores: list[PromptScore],
    model: str,
    backend: str,
) -> dict:
    """Return a compact summary dict suitable for comparison tables."""
    from .scoring import aggregate_scores

    agg = aggregate_scores(scores)
    successful = [r for r in results if r.success]
    avg_tps = (
        sum(r.tokens_per_second for r in successful) / len(successful)
        if successful
        else 0.0
    )

    return {
        "model": model,
        "backend": backend,
        "avg_tps": round(avg_tps, 3),
        "quality_score": round(agg["overall"], 4),
        "total_prompts": len(scores),
        "errors": sum(1 for r in results if not r.success),
    }
