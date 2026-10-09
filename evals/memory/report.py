"""Render persistent-memory benchmark JSONL as a Markdown baseline report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.memory.bench import PRICES_PER_MILLION, STRATEGIES


def load_results(path: Path) -> list[dict[str, Any]]:
    """Keep the latest terminal result for each append-only resume key."""
    results: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise TypeError(f"invalid result on line {number}")
        results[row["key"]] = row
    return list(results.values())


def wilson(passes: int, total: int) -> tuple[float, float]:
    if not total:
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


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(row.get(key) is True for row in rows)


def _rate(value: int, total: int) -> str:
    return f"{value}/{total} ({value / total:.1%})" if total else "0/0"


def _usage(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get("usage", {}).get(key, 0)) for row in rows)


def render(rows: list[dict[str, Any]]) -> str:
    terminal = [row for row in rows if not row.get("infra_error")]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in terminal:
        grouped[(str(row.get("family")), str(row.get("strategy")))].append(row)
        by_strategy[str(row.get("strategy"))].append(row)
    lines = [
        "# Zeta memory benchmark: phase 1 baselines",
        "",
        "## Method",
        "",
        "Each cell runs two or three separate headless `zeta -p` sessions in an isolated fixture repository and private `ZETA_HOME`; no phase uses `--resume`. The phases share only the project registry and its five memory files. Hidden graders are copied only into a trusted temporary grading directory after each phase. The harness advertises only the `read` and `write` tools and uses `--yolo` only to auto-approve that explicit allowlist.",
        "",
        "The four strategies are S0 (empty project memory), S1 (oracle-quality text in current project memory), oracle-snippet (relevant source in the final prompt), and oracle-history (all prior prompts and assistant acknowledgements in the final prompt). Results are keyed by task, strategy, repetition, model, token budget, and Zeta revision. A provider/network failure is retried once and recorded.",
        "",
        "The pre-run estimate uses 180,000 total input tokens per cell (60,000 uncached plus 120,000 cache-read), as specified in the research plan. The full 144-cell matrix projects 25.92M input tokens, below the 40M guard.",
        "",
        "## Families",
        "",
        "- Stable project decision: reuse a tested opaque serialization choice.",
        "- Environment gotcha/procedure: recall an unusual validated operation.",
        "- Failure lesson: avoid an observed failed approach and use its replacement.",
        "- Updated/conflicting fact: prefer stronger new evidence over a superseded value.",
        "- Stale current state: do not resume work after completion evidence.",
        "- Abstention/absent memory: do not invent or repurpose a related value.",
        "",
        "Each family has two chains with non-inferable opaque literals.",
        "",
        "## Results by family",
        "",
        "| Family | Strategy | n | Pass | Wrong-memory | Correct abstention |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (family, strategy), group in sorted(grouped.items()):
        lines.append(
            f"| {family} | {strategy} | {len(group)} | "
            f"{_rate(_sum(group, 'passed'), len(group))} | "
            f"{_rate(_sum(group, 'wrong_memory'), len(group))} | "
            f"{_rate(_sum(group, 'correct_abstention'), len(group))} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Strategy | n | Pass (Wilson 95% CI) | Wrong-memory | Stale-fact | Abstention on absent-memory | Input | Cache read | Cache write | Output | Cost | Network drops |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        group = by_strategy.get(strategy, [])
        passes = _sum(group, "passed")
        low, high = wilson(passes, len(group))
        absent = [
            row for row in group if row.get("family") == "abstention-absent-memory"
        ]
        cost = sum(float(row.get("estimated_cost_usd") or 0) for row in group)
        lines.append(
            f"| {strategy} | {len(group)} | {_rate(passes, len(group))}; {low:.1%}–{high:.1%} | "
            f"{_rate(_sum(group, 'wrong_memory'), len(group))} | "
            f"{_rate(_sum(group, 'stale_fact_selected'), len(group))} | "
            f"{_rate(_sum(absent, 'correct_abstention'), len(absent))} | "
            f"{_usage(group, 'input_tokens'):,} | {_usage(group, 'cache_read_tokens'):,} | "
            f"{_usage(group, 'cache_write_tokens'):,} | {_usage(group, 'output_tokens'):,} | "
            f"${cost:.4f} | {sum(int(row.get('network_drops', 0)) for row in group)} |"
        )
    all_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in terminal)
    s0 = by_strategy.get("S0", [])
    s1 = by_strategy.get("S1", [])
    s0_recall = [row for row in s0 if row.get("family") != "abstention-absent-memory"]
    s1_recall = [row for row in s1 if row.get("family") != "abstention-absent-memory"]
    all_tokens = sum(
        _usage(terminal, key)
        for key in (
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    )
    lines.extend(
        [
            "",
            f"Measured terminal cells: {len(terminal)}. Total tokens including cache writes: {all_tokens:,}. Estimated model cost: ${all_cost:.4f}. Pricing: ${PRICES_PER_MILLION['gpt-5.6-luna']['input']:.2f}/M uncached input, ${PRICES_PER_MILLION['gpt-5.6-luna']['cache_read']:.2f}/M cache read, and ${PRICES_PER_MILLION['gpt-5.6-luna']['output']:.2f}/M output; cache writes have no added charge in this table.",
            "",
            "## Leakage check",
            "",
            "S0 has no project-specific literal in its final prompt, fixture, or project memory. A high S0 success rate on the five recall/action families indicates leakage or an invalid task and requires task repair. The absent-memory family is different: correct S0 abstention is expected and is not evidence of leakage.",
            "",
            "## Threats to validity",
            "",
            "- S1 is an oracle-quality read-path ceiling, not a measure of automatic extraction quality.",
            "- Compact opaque-literal tasks isolate memory but are less realistic than full software changes.",
            "- Three repetitions give wide confidence intervals; results can be sensitive to model and provider changes.",
            "- The oracle-history control reconstructs prior user prompts and assistant acknowledgements. It does not replay provider-internal state.",
            "- Always-in-prompt retrieval bytes are estimated from injected UTF-8 memory size. Token counts and costs come from request-level cache traces.",
            "- The benchmark does not yet cover cross-project preferences, injection/secret safety, provenance conflicts, or retrieval noise/scale.",
            "- Calibration exposed an ambiguous failure-lesson action contract. The contract was corrected before this fresh final-revision matrix; calibration rows are not included here.",
            "",
            "## What S2 must beat",
            "",
            f"Observed baselines: S0 passed {_sum(s0, 'passed')}/{len(s0)} overall and {_sum(s0_recall, 'passed')}/{len(s0_recall)} recall/action cells. Oracle-seeded S1 passed {_sum(s1, 'passed')}/{len(s1)} overall and {_sum(s1_recall, 'passed')}/{len(s1_recall)} recall/action cells, with {_sum(s1, 'wrong_memory')} wrong-memory selections.",
            "",
            "For the next lane, the minimum advancement target is at least 33/36 (91.7%) overall and 27/30 (90.0%) on recall/action cells, with 0/36 wrong-memory selections, 0/12 stale-fact selections, 6/6 correct abstentions, and a median of at most one grouped approval per useful session. The stretch target is the S1/oracle ceiling of 36/36. Extraction precision and recall must be reported separately so an end-to-end failure can be assigned to writing or use.",
            "",
            "S2 should also reduce the gap to oracle controls without adding secret/injection failures. A later S3 read path is worthwhile only if it matches S2 accuracy while reducing always-in-prompt tokens or clearly wins on noisy/large archives. Do not add embeddings or graphs unless they beat lexical/agentic lookup on predeclared paraphrase or multi-hop tasks without worse stale-fact selection.",
            "",
            "## Cuts",
            "",
            f"Planned cells: 144. Terminal cells: {len(terminal)}. Cells cut: {max(0, 144 - len(terminal))}. No family, strategy, or repetition was cut from the final matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = render(load_results(args.results))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
