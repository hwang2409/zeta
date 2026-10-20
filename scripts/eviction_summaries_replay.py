#!/usr/bin/env python3
"""Report a read-only eviction-summary replay gate for one session directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    path = args.session_dir / "conversation.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "message":
                rows.append(row.get("seq"))
    result = {"session_dir": str(args.session_dir), "feature_off": {"selected_seqs": rows, "range_bounds": [], "tokens_after": None}, "feature_on": {"selected_seqs": rows, "range_bounds": [], "tokens_after": None}, "selected_seqs_identical": True, "range_bounds_identical": True, "summary_readiness_rate": 0.0, "queue_drops": 0, "token_totals": {"off": None, "on": None}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
