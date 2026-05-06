"""Report generator: produces console tables and JSON output."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box

from .runners.base import InferenceResult
from .scoring import PromptScore, aggregate_scores

console = Console()


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_run_summary(
    results: list[InferenceResult],
    scores: list[PromptScore],
    model: str,
    backend: str,
) -> None:
    """Pretty-print a per-run summary table to stdout."""
    agg = aggregate_scores(scores)
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)

    # ---- per-prompt detail table ----
    detail = Table(
        title=f"[bold]{backend} / {model}[/bold] – Prompt Results",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    detail.add_column("ID", style="dim", no_wrap=True)
    detail.add_column("Category", style="cyan")
    detail.add_column("TPS", justify="right", style="green")
    detail.add_column("Tokens", justify="right")
    detail.add_column("Pass", justify="center")
    detail.add_column("Expected", style="yellow", no_wrap=True)
    detail.add_column("Got", style="magenta")

    result_map = {r.extra.get("prompt_id", ""): r for r in results if hasattr(r, "extra")}

    for s in scores:
        r = result_map.get(s.prompt_id)
        tps_str = f"{r.tokens_per_second:.1f}" if r else "—"
        tokens_str = str(r.completion_tokens) if r else "—"
        pass_icon = "[green]✓[/green]" if s.passed else "[red]✗[/red]"
        got_preview = s.actual[:50].replace("\n", " ") if s.actual else "ERROR"
        detail.add_row(
            s.prompt_id,
            s.category,
            tps_str,
            tokens_str,
            pass_icon,
            s.expected[:30],
            got_preview,
        )

    console.print(detail)

    # ---- summary stats ----
    summary = Table(box=box.MINIMAL, show_header=False, padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column(justify="right")

    successful_results = [r for r in results if r.success]
    if successful_results:
        avg_tps = sum(r.tokens_per_second for r in successful_results) / len(
            successful_results
        )
        avg_tokens = sum(r.completion_tokens for r in successful_results) / len(
            successful_results
        )
        summary.add_row("Avg tokens/sec", f"[bold green]{avg_tps:.2f}[/bold green]")
        summary.add_row("Avg completion tokens", f"{avg_tokens:.1f}")

    summary.add_row("Quality score", f"[bold]{_pct(agg['overall'])}[/bold]  ({passed}/{total})")
    for cat, acc in sorted(agg["by_category"].items()):
        summary.add_row(f"  {cat}", _pct(acc))

    console.print(summary)


def print_comparison_table(run_summaries: list[dict]) -> None:
    """Print a multi-model comparison table."""
    if not run_summaries:
        return

    tbl = Table(
        title="[bold]Model Comparison[/bold]",
        box=box.HEAVY_OUTLINE,
        show_lines=True,
    )
    tbl.add_column("Backend", style="dim")
    tbl.add_column("Model", style="bold cyan", no_wrap=True)
    tbl.add_column("Avg TPS", justify="right", style="green")
    tbl.add_column("Quality", justify="right")
    tbl.add_column("Prompts", justify="right")
    tbl.add_column("Errors", justify="right", style="red")

    for s in run_summaries:
        tbl.add_row(
            s.get("backend", ""),
            s.get("model", ""),
            f"{s.get('avg_tps', 0):.2f}",
            _pct(s.get("quality_score", 0)),
            str(s.get("total_prompts", 0)),
            str(s.get("errors", 0)),
        )

    console.print(tbl)


def save_json(
    results: list[InferenceResult],
    scores: list[PromptScore],
    model: str,
    backend: str,
    output_path: Path,
) -> None:
    """Persist full results + scores to a JSON file."""
    agg = aggregate_scores(scores)
    successful = [r for r in results if r.success]
    avg_tps = (
        sum(r.tokens_per_second for r in successful) / len(successful)
        if successful
        else 0.0
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "backend": backend,
        "summary": {
            "avg_tokens_per_second": round(avg_tps, 3),
            "quality_score": round(agg["overall"], 4),
            "quality_by_category": {k: round(v, 4) for k, v in agg["by_category"].items()},
            "total_prompts": len(scores),
            "passed": sum(1 for s in scores if s.passed),
            "errors": sum(1 for r in results if not r.success),
        },
        "results": [
            {
                "prompt_id": r.extra.get("prompt_id", ""),
                "prompt": r.prompt,
                "response": r.response,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "tokens_per_second": round(r.tokens_per_second, 3),
                "error": r.error,
            }
            for r in results
        ],
        "scores": [asdict(s) for s in scores],
    }

    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"[dim]Results saved to {output_path}[/dim]")
