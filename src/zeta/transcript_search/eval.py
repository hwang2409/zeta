"""Content-free transcript-search evaluation runner."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class EvalResult:
    query_count: int
    recall_at: dict[int, float]
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "recall_at": {str(key): value for key, value in self.recall_at.items()},
            "mrr": self.mrr,
        }


def evaluate_manifest(manifest_path: Path, units_path: Path) -> EvalResult:
    """Rank a frozen synthetic corpus and return recall@k and MRR."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [
        json.loads(line)
        for line in units_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    documents = {unit["unit_id"]: _tokens(unit["text"]) for unit in units}
    frequencies = Counter(token for values in documents.values() for token in set(values))
    rankings: list[list[str]] = []
    targets: list[set[str]] = []
    for case in manifest["queries"]:
        query = set(_tokens(case["query"]))
        scored = []
        for unit_id, tokens in documents.items():
            counts = Counter(tokens)
            score = sum(
                (1.0 + math.log1p(counts[token]))
                * math.log((len(documents) + 1) / (frequencies[token] + 0.5))
                for token in query & counts.keys()
            )
            if score:
                scored.append((-score, unit_id))
        rankings.append([unit_id for _, unit_id in sorted(scored)])
        targets.append(set(case["target_unit_ids"]))
    recall = {
        k: sum(bool(set(ranking[:k]) & target) for ranking, target in zip(rankings, targets, strict=True))
        / len(targets)
        for k in (1, 5, 10)
    }
    reciprocal = []
    for ranking, target in zip(rankings, targets, strict=True):
        rank = next((index for index, unit_id in enumerate(ranking, 1) if unit_id in target), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    return EvalResult(len(targets), recall, sum(reciprocal) / len(reciprocal))


def _tokens(value: str) -> list[str]:
    return [match.group().casefold() for match in _TOKEN.finditer(value)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="evaluate transcript search fixtures")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("units", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_manifest(args.manifest, args.units).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
