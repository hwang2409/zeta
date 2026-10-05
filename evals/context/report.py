"""Render context benchmark JSONL results as Markdown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results, keeping the last complete record for each resume key."""
    records: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise TypeError(f"invalid result on line {number}")
        records[row["key"]] = row
    return list(records.values())


def wilson(passes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval."""
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = passes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _tokens(row: dict[str, Any]) -> float:
    usage = row.get("usage", {})
    if not isinstance(usage, dict):
        return 0.0
    return sum(
        _number(usage, key)
        for key in (
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    )


def _cache_ratio(rows: Iterable[dict[str, Any]]) -> float:
    read = input_tokens = writes = 0.0
    for row in rows:
        usage = row.get("usage", {})
        if not isinstance(usage, dict):
            continue
        read += _number(usage, "cache_read_tokens")
        input_tokens += _number(usage, "input_tokens")
        writes += _number(usage, "cache_write_tokens")
    total = read + input_tokens + writes
    return read / total if total else 0.0


def _label(cap: object) -> str:
    return f"{int(cap):,}" if isinstance(cap, int) else str(cap)


def _summary(rows: list[dict[str, Any]]) -> list[str]:
    passed = sum(row.get("passed") is True for row in rows)
    low, high = wilson(passed, len(rows))
    tokens = [_tokens(row) for row in rows]
    costs = [_number(row, "estimated_cost_usd") for row in rows]
    return [
        f"{passed}/{len(rows)} ({passed / len(rows):.1%}; {low:.1%}–{high:.1%})",
        f"{statistics.mean(tokens):,.0f}",
        f"{statistics.median(tokens):,.0f}",
        f"{_cache_ratio(rows):.1%}",
        f"${statistics.mean(costs):.4f}",
        f"{statistics.mean(_number(row, 'wall_seconds') for row in rows):.1f}s",
        f"{statistics.mean(_number(row, 'compactions') for row in rows):.2f}",
        f"{statistics.mean(_number(row, 'recall_calls') for row in rows):.2f}",
    ]


def render(records: list[dict[str, Any]]) -> str:
    if not records:
        return "# Zeta context benchmark\n\nNo results.\n"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_label(row.get("token_budget"))].append(row)
    header = "| Cap | Pass rate (Wilson 95% CI) | Mean tokens | Median tokens | Cache-read ratio | Mean cost | Mean wall | Mean compactions | Mean recall calls |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = ["# Zeta context benchmark", "", header, divider]
    for strategy in sorted(grouped, key=lambda value: int(value.replace(",", ""))):
        lines.append("| " + " | ".join([strategy, *_summary(grouped[strategy])]) + " |")
    lines.extend(["", "## Per-task breakdown", ""])
    task_header = "| Cap | Task | Passed | Runs | Pass rate | Mean tokens | Mean cost | Mean wall |"
    lines.extend([task_header, "|---|---|---:|---:|---:|---:|---:|---:|"])
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_task[(_label(row.get("token_budget")), str(row.get("task")))].append(row)
    for (strategy, task), rows in sorted(by_task.items()):
        passed = sum(row.get("passed") is True for row in rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    task,
                    str(passed),
                    str(len(rows)),
                    f"{passed / len(rows):.1%}",
                    f"{statistics.mean(_tokens(row) for row in rows):,.0f}",
                    f"${statistics.mean(_number(row, 'estimated_cost_usd') for row in rows):.4f}",
                    f"{statistics.mean(_number(row, 'wall_seconds') for row in rows):.1f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = render(load_results(args.results))
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
