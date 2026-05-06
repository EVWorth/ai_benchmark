"""CLI entry point for ai-benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import requests
import click

from rich.console import Console
from .engine import load_prompts, run_benchmark, summarise_run
from .reporter import print_run_summary, print_comparison_table, save_json
from .runners.llamacpp_runner import LlamaCppRunner
from .runners.ollama_runner import OllamaRunner

console = Console()

BACKENDS = {
    "ollama": OllamaRunner,
    "llamacpp": LlamaCppRunner,
}


def _make_runner(backend: str, host: Optional[str]):
    cls = BACKENDS.get(backend)
    if cls is None:
        console.print(f"[red]Unknown backend '{backend}'. Choose: {list(BACKENDS)}[/red]")
        sys.exit(1)
    kwargs = {}
    if host:
        kwargs["host"] = host
    return cls(**kwargs)


@click.group()
@click.version_option(package_name="ai_benchmark")
def main():
    """Benchmark local AI models for tokens-per-second and reasoning quality."""


# ---------------------------------------------------------------------------
# benchmark run
# ---------------------------------------------------------------------------

@main.command("run")
@click.option("--backend", "-b", default="ollama", show_default=True,
              type=click.Choice(list(BACKENDS)), help="Inference backend to use.")
@click.option("--host", default=None, help="Override the backend server URL.")
@click.option("--model", "-m", required=True, help="Model name to benchmark.")
@click.option("--categories", "-c", multiple=True,
              help="Restrict to these prompt categories (repeatable).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Save full results to this JSON file.")
@click.option("--temperature", default=0.0, show_default=True, type=float,
              help="Sampling temperature (0 = greedy/deterministic).")
@click.option("--max-tokens", default=256, show_default=True, type=int,
              help="Max completion tokens per prompt.")
def run_cmd(backend, host, model, categories, output, temperature, max_tokens):
    """Run the benchmark suite against a single model."""
    runner = _make_runner(backend, host)

    gen_kwargs: dict = {}
    if backend == "ollama":
        gen_kwargs["options"] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    elif backend == "llamacpp":
        gen_kwargs["temperature"] = temperature
        gen_kwargs["n_predict"] = max_tokens

    prompts = load_prompts(categories=list(categories) if categories else None)
    console.print(
        f"\n[bold]Backend:[/bold] {backend}  "
        f"[bold]Model:[/bold] {model}  "
        f"[bold]Prompts:[/bold] {len(prompts)}\n"
    )

    results, scores = run_benchmark(
        runner=runner,
        model=model,
        prompts=prompts,
        generation_kwargs=gen_kwargs,
    )

    print_run_summary(results, scores, model, backend)

    if output:
        save_json(results, scores, model, backend, Path(output))


# ---------------------------------------------------------------------------
# benchmark compare
# ---------------------------------------------------------------------------

@main.command("compare")
@click.option("--results-dir", "-d", required=True, type=click.Path(exists=True),
              help="Directory containing JSON result files to compare.")
def compare_cmd(results_dir):
    """Compare results from multiple saved JSON files."""
    result_files = sorted(Path(results_dir).glob("*.json"))
    if not result_files:
        console.print("[red]No JSON result files found in the directory.[/red]")
        sys.exit(1)

    summaries = []
    for f in result_files:
        try:
            data = json.loads(f.read_text())
            summaries.append({
                "backend": data.get("backend", "?"),
                "model": data.get("model", f.stem),
                "avg_tps": data["summary"]["avg_tokens_per_second"],
                "quality_score": data["summary"]["quality_score"],
                "total_prompts": data["summary"]["total_prompts"],
                "errors": data["summary"]["errors"],
            })
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            console.print(f"[yellow]Skipping {f.name}: {exc}[/yellow]")

    if summaries:
        # Sort by TPS descending
        summaries.sort(key=lambda s: s["avg_tps"], reverse=True)
        print_comparison_table(summaries)
    else:
        console.print("[red]No valid result files to compare.[/red]")


# ---------------------------------------------------------------------------
# benchmark list-models
# ---------------------------------------------------------------------------

@main.command("list-models")
@click.option("--backend", "-b", default="ollama", show_default=True,
              type=click.Choice(list(BACKENDS)), help="Backend to query.")
@click.option("--host", default=None, help="Override the backend server URL.")
def list_models_cmd(backend, host):
    """List models available in the specified backend."""
    runner = _make_runner(backend, host)
    try:
        models = runner.list_models()
    except requests.RequestException as exc:
        console.print(f"[red]Could not reach {backend} server: {exc}[/red]")
        sys.exit(1)

    if models:
        for m in models:
            console.print(f"  {m}")
    else:
        console.print("[yellow]No models found (server may have no model loaded).[/yellow]")


if __name__ == "__main__":
    main()
