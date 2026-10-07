"""Bounded real-line quality sample for the OptChat compactor experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals.optchat.eviction_replay import _signatures, active_message_records
from evals.optchat.real_compactor import LunaCompactor
from evals.optchat.strategy import NODE_BYTES, OptChatView, map_zeta_message
from zeta.protocol.types import Message


def _samples(
    records: list[tuple[int, Message]], limit: int
) -> list[tuple[int, str]]:
    sources: dict[tuple[str, str], list[int]] = defaultdict(list)
    candidates: list[tuple[int, str]] = []
    for seq, message in records:
        for kind, values in _signatures(message).items():
            for value in sorted(values):
                previous = sources[(kind, value)]
                if previous:
                    candidates.append((previous[0], value))
                previous.append(seq)
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[int(index * step)] for index in range(limit)]


def run(path: Path, *, entries: int, home: Path, limit: int = 10) -> dict[str, int | float]:
    records = active_message_records(path)[-entries:]
    samples = _samples(records, limit)
    targets: dict[int, list[str]] = defaultdict(list)
    for seq, value in samples:
        targets[seq].append(value)

    compactor = LunaCompactor(home=home)
    view = OptChatView()
    in_line = one_zoom = unmatched = 0
    for seq, message in records:
        for kind, text in map_zeta_message(message):
            source = f"{kind}: {text}"
            values = [
                value for value in targets.get(seq, ()) if value.casefold() in source.casefold()
            ]
            if values:
                if len(source.encode()) <= NODE_BYTES:
                    line = source
                else:
                    context = "\n".join(view.nodes[key].text for key in view.parts)
                    line = compactor(context, source)
                for value in values:
                    if value.casefold() in line.casefold():
                        in_line += 1
                    else:
                        one_zoom += 1
                remaining = [value for value in targets[seq] if value not in values]
                if remaining:
                    targets[seq] = remaining
                else:
                    targets.pop(seq)
            view.append(kind, text, source_seq=seq)
    unmatched = sum(len(values) for values in targets.values())
    total = in_line + one_zoom + unmatched
    return {
        "samples": total,
        "fact_in_level0_line": in_line,
        "fact_one_zoom_to_verbatim": one_zoom,
        "unmatched_mapping": unmatched,
        "calls": compactor.calls,
        "retries": compactor.retries,
        "overshoot_attempts": compactor.overshoot_attempts,
        "input_tokens": compactor.usage["input_tokens"],
        "cache_read_tokens": compactor.usage["cache_read_input_tokens"],
        "output_tokens": compactor.usage["output_tokens"],
        "total_input_tokens": compactor.total_input_tokens,
        "estimated_cost_usd": (
            compactor.usage["input_tokens"] * 0.20
            + compactor.usage["cache_read_input_tokens"] * 0.02
            + compactor.usage["output_tokens"] * 1.20
        )
        / 1_000_000,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--entries", type=int, default=300)
    parser.add_argument("--home", type=Path, default=Path.home() / ".zeta")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(
        json.dumps(
            run(args.path, entries=args.entries, home=args.home),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
