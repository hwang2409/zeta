#!/usr/bin/env python3
"""Report weekly Zeta session cost and speed metrics without model calls.

Run weekly with ``uv run python scripts/bench/session_cost_report.py --since
2026-11-03T00:00:00Z --until 2026-11-10T00:00:00Z``.  The command reads
conversation files line by line and never writes to the Zeta home.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

ROLES = {"orchestrator", "lead", "worker", "reviewer", "planner"}
PR_RE = re.compile(r"(?i)(?:pull request|pr)\s*#?\s*(\d+)|#(\d+)")
POLL_RE = re.compile(r"(?i)(?:\bsleep\b|gh\s+[^\n]*--watch|wait\s+for\s+(?:ci|checks)|poll(?:ing)?)")

def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value).astimezone(dt.UTC)

def row_time(row: dict, fallback: dt.datetime | None) -> dt.datetime | None:
    for value in (row.get("created_at"), row.get("timestamp"), row.get("data", {}).get("created_at")):
        if isinstance(value, str):
            try:
                return parse_time(value)
            except ValueError:
                pass
    return fallback

def text_of(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))

def role_for(path: Path) -> str:
    if "agents" not in path.parts:
        return "orchestrator"
    lifecycle = path.with_name("agent_lifecycle.json")
    try:
        value = json.loads(lifecycle.read_text()).get("agent_type", "other")
    except (OSError, ValueError):
        value = "other"
    return value if value in ROLES else "other"

def model_for(path: Path) -> str:
    session_root = path
    while session_root.name != "sessions" and session_root.parent != session_root:
        if (session_root / "meta.json").is_file():
            break
        session_root = session_root.parent
    for name in ("meta.json", "session_state.json"):
        candidate = session_root / name
        try:
            data = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        for key in ("model", "selected_model"):
            if data.get(key):
                return str(data[key])
    return "unknown"

def empty_group():
    return {"agents": set(), "started": set(), "turns": 0, "uncached": 0, "cache_read": 0, "cache_create": 0, "output": 0, "first_inputs": [], "polling_turns": 0}

def report(home: Path, session_ids: list[str] | None, since: dt.datetime | None, until: dt.datetime | None) -> dict:
    roots = [home / "sessions" / x for x in session_ids] if session_ids else sorted((home / "sessions").iterdir())
    groups = defaultdict(empty_group)
    prs = defaultdict(lambda: defaultdict(set))
    evictions = {"count": 0, "bytes": 0}
    skipped = 0
    for root in roots:
        if not root.is_dir():
            continue
        files = [root / "conversation.jsonl"] + sorted((root / "agents").glob("*/conversation.jsonl"))
        for path in files:
            if not path.is_file():
                continue
            role, model, fallback = role_for(path), model_for(path), None
            key = (role, model)
            g = groups[key]
            g["agents"].add(str(path.parent))
            try:
                with path.open() as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            skipped += 1
                            continue
                        if row.get("type") == "header":
                            fallback = row_time(row, None)
                        when = row_time(row, fallback)
                        if when and ((since and when < since) or (until and when >= until)):
                            continue
                        if row.get("type") == "compaction" and row.get("data", {}).get("kind") == "evict":
                            evictions["count"] += 1
                            evictions["bytes"] += len(line.encode("utf-8"))
                        message = row.get("data", {}).get("message", {})
                        if not isinstance(message, dict) or message.get("role") != "assistant":
                            continue
                        metadata = message.get("metadata", {})
                        usage = metadata.get("provider_usage", metadata.get("usage", {}))
                        if not isinstance(usage, dict):
                            usage = {}
                        output = int(metadata.get("output_tokens", usage.get("output_tokens", 0)) or 0)
                        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                        cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
                        input_total = int(usage.get("input_tokens", 0) or 0)
                        uncached = max(0, input_total - cache_read - cache_create)
                        g["turns"] += 1
                        g["uncached"] += uncached; g["cache_read"] += cache_read; g["cache_create"] += cache_create; g["output"] += output
                        agent_key = str(path.parent)
                        if agent_key not in g["started"]:
                            g["started"].add(agent_key)
                            g["first_inputs"].append(input_total + cache_read + cache_create)
                        if POLL_RE.search(text_of(message)):
                            g["polling_turns"] += 1
            except (OSError, UnicodeDecodeError):
                skipped += 1
            if role in {"worker", "reviewer"}:
                try:
                    lifecycle = json.loads((path.parent / "agent_lifecycle.json").read_text())
                    description = lifecycle.get("description", "")
                except (OSError, ValueError):
                    description = ""
                matches = {a or b for a, b in PR_RE.findall(description)}
                if len(matches) == 1:
                    prs[next(iter(matches))][role].add(str(path.parent))
    rows = []
    for (role, model), g in sorted(groups.items()):
        total = g["uncached"] + g["cache_read"] + g["cache_create"]
        rows.append({"role": role, "model": model, "agents": len(g["agents"]), "turns": g["turns"], "turns_per_agent": round(g["turns"] / len(g["agents"]), 2) if g["agents"] else 0, "uncached_per_turn": round(g["uncached"] / g["turns"], 2) if g["turns"] else 0, "cache_read_per_turn": round(g["cache_read"] / g["turns"], 2) if g["turns"] else 0, "cache_create_per_turn": round(g["cache_create"] / g["turns"], 2) if g["turns"] else 0, "output_per_turn": round(g["output"] / g["turns"], 2) if g["turns"] else 0, "cache_read_rate": round(g["cache_read"] / total, 4) if total else 0, "median_first_request_total_input": statistics.median(g["first_inputs"]) if g["first_inputs"] else 0, "polling_turns": g["polling_turns"]})
    turns = sum(x["turns"] for x in rows)
    pr_rows = [{"pr": pr, "workers": len(v["worker"]), "reviewers": len(v["reviewer"]), "shared": len(v["worker"] | v["reviewer"]) >= 3} for pr, v in sorted(prs.items(), key=lambda x: int(x[0]))]
    return {"groups": rows, "evictions": {**evictions, "per_1000_turns": round(evictions["count"] * 1000 / turns, 2) if turns else 0}, "pull_requests": pr_rows, "skipped_rows": skipped, "turns": turns}

def print_report(data: dict) -> None:
    print(f"turns: {data['turns']}  skipped rows: {data['skipped_rows']}")
    print("role/model                         agents turns/agent uncached cache-read cache-create output cache-rate first-input polling")
    for x in data["groups"]:
        print(f"{x['role']}/{x['model']:<27} {x['agents']:>6} {x['turns_per_agent']:>11} {x['uncached_per_turn']:>8} {x['cache_read_per_turn']:>10} {x['cache_create_per_turn']:>11} {x['output_per_turn']:>6} {x['cache_read_rate']:>9.1%} {x['median_first_request_total_input']:>10} {x['polling_turns']:>7}")
    e = data["evictions"]
    print(f"evictions: {e['count']} rows, {e['bytes']} bytes, {e['per_1000_turns']} per 1,000 turns")
    print("per-merge numbers are allocation, not attribution")
    print(f"exact-pr agents: {len(data['pull_requests'])}; shared (3+ workers/reviewers): {sum(x['shared'] for x in data['pull_requests'])}")

def self_check() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); session = root / "sessions" / "s" / "agents" / "1"; session.mkdir(parents=True)
        (root / "sessions" / "s" / "meta.json").write_text('{"model":"test"}')
        (session / "agent_lifecycle.json").write_text('{"agent_type":"worker","description":"fix PR #12"}')
        row = {"type":"message","data":{"message":{"role":"assistant","content":[{"text":"sleep 1"}],"metadata":{"output_tokens":3,"provider_usage":{"input_tokens":10,"cache_read_input_tokens":4,"cache_creation_input_tokens":2}}}}}
        (session / "conversation.jsonl").write_text(json.dumps({"type":"header","data":{"created_at":"2026-01-01T00:00:00Z"}})+"\n"+json.dumps(row)+"\n")
        result = report(root, ["s"], None, None)
        assert result["turns"] == 1 and result["groups"][0]["uncached_per_turn"] == 4 and result["groups"][0]["polling_turns"] == 1
        assert result["pull_requests"][0]["pr"] == "12"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_ids", nargs="*"); parser.add_argument("--since"); parser.add_argument("--until"); parser.add_argument("--zeta-home", default=os.environ.get("ZETA_HOME", "~/.zeta")); parser.add_argument("--json", action="store_true"); parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check: ok"); raise SystemExit
    result = report(Path(args.zeta_home).expanduser(), args.session_ids or None, parse_time(args.since), parse_time(args.until))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
