#!/usr/bin/env python3
"""Report weekly Zeta session cost and speed metrics without model calls."""
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


def safe_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def read_json(path: Path) -> dict:
    if not safe_file(path):
        return {}
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            with os.fdopen(fd) as stream:
                fd = -1
                value = json.load(stream)
        finally:
            if fd >= 0:
                os.close(fd)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeDecodeError):
        return {}


def role_for(path: Path) -> str:
    if "agents" not in path.parts:
        return "orchestrator"
    value = read_json(path.parent / "agent_lifecycle.json").get("agent_type", "other")
    return value if value in ROLES else "other"


def model_for(path: Path, assistant_models: set[str]) -> str:
    if "agents" in path.parts:
        lifecycle = read_json(path.parent / "agent_lifecycle.json")
        for key in ("model", "selected_model"):
            if lifecycle.get(key):
                return str(lifecycle[key])
        if len(assistant_models) == 1:
            return next(iter(assistant_models))
        return "unknown"
    for candidate in (path.parent / "meta.json", path.parent / "session_state.json"):
        data = read_json(candidate)
        for key in ("model", "selected_model"):
            if data.get(key):
                return str(data[key])
    return "unknown"


def empty_group():
    return {"agents": set(), "turns": 0, "uncached": 0, "cache_read": 0, "cache_create": 0, "output": 0, "first_inputs": [], "polling_turns": 0}


def in_window(when: dt.datetime | None, since: dt.datetime | None, until: dt.datetime | None) -> bool:
    return when is not None and not (since and when < since) and not (until and when >= until)


def scan_file(path: Path, since: dt.datetime | None, until: dt.datetime | None) -> tuple[dict, int, int, int]:
    result = empty_group()
    result["assistant_models"] = set()
    result["eligible"] = False
    result["first_in_window"] = None
    evictions = evicted_bytes = skipped = 0
    fallback = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    skipped += 1
                    continue
                if row.get("type") == "header":
                    fallback = row_time(row, None)
                when = row_time(row, fallback)
                if not in_window(when, since, until):
                    continue
                if row.get("type") == "compaction" and row.get("data", {}).get("kind") == "evict":
                    evictions += 1
                    evicted_bytes += len(line.encode("utf-8"))
                message = row.get("data", {}).get("message", {})
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                metadata = message.get("metadata", {})
                usage = metadata.get("provider_usage", metadata.get("usage", {}))
                if not isinstance(usage, dict):
                    usage = {}
                for key in ("model", "selected_model"):
                    if message.get(key):
                        result["assistant_models"].add(str(message[key]))
                    if metadata.get(key):
                        result["assistant_models"].add(str(metadata[key]))
                output = int(metadata.get("output_tokens", usage.get("output_tokens", 0)) or 0)
                cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
                uncached = int(usage.get("input_tokens", 0) or 0)
                result["turns"] += 1
                result["uncached"] += uncached
                result["cache_read"] += cache_read
                result["cache_create"] += cache_create
                result["output"] += output
                result["eligible"] = True
                if result["first_in_window"] is None:
                    result["first_in_window"] = uncached + cache_read + cache_create
                if POLL_RE.search(text_of(message)):
                    result["polling_turns"] += 1
    except (OSError, UnicodeDecodeError):
        skipped += 1
    return result, evictions, evicted_bytes, skipped


def session_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        return []
    files = []
    conversation = root / "conversation.jsonl"
    if safe_file(conversation):
        files.append(conversation)
    agents = root / "agents"
    if agents.is_dir() and not agents.is_symlink():
        for entry in sorted(agents.iterdir()):
            if entry.is_dir() and not entry.is_symlink() and safe_file(entry / "conversation.jsonl"):
                files.append(entry / "conversation.jsonl")
    return files


def valid_session_id(value: str) -> bool:
    return bool(value) and not os.path.isabs(value) and value not in {".", ".."} and "/" not in value and "\\" not in value and ".." not in value


def report(home: Path, session_ids: list[str] | None, since: dt.datetime | None, until: dt.datetime | None) -> dict:
    home = home.expanduser().resolve()
    sessions = home / "sessions"
    if session_ids:
        roots = [sessions / x for x in session_ids if valid_session_id(x)]
    else:
        roots = [x for x in sorted(sessions.iterdir()) if x.is_dir() and not x.is_symlink()]
    groups = defaultdict(empty_group)
    prs = defaultdict(lambda: defaultdict(set))
    evictions = {"count": 0, "bytes": 0}
    skipped = 0
    for root in roots:
        for path in session_files(root):
            role = role_for(path)
            raw, count, size, bad = scan_file(path, since, until)
            skipped += bad
            evictions["count"] += count
            evictions["bytes"] += size
            if not raw["eligible"]:
                continue
            model = model_for(path, raw["assistant_models"])
            key = (role, model)
            g = groups[key]
            g["agents"].add(str(path.parent))
            for field in ("turns", "uncached", "cache_read", "cache_create", "output", "polling_turns"):
                g[field] += raw[field]
            g["first_inputs"].append(raw["first_in_window"])
            if role in {"worker", "reviewer"}:
                lifecycle = read_json(path.parent / "agent_lifecycle.json")
                matches = {a or b for a, b in PR_RE.findall(str(lifecycle.get("description", "")))}
                if len(matches) == 1:
                    prs[next(iter(matches))][role].add(str(path.parent))
    rows = []
    for (role, model), g in sorted(groups.items()):
        total = g["uncached"] + g["cache_read"] + g["cache_create"]
        turns = g["turns"]
        rows.append({"role": role, "model": model, "agents": len(g["agents"]), "turns": turns, "turns_per_agent": round(turns / len(g["agents"]), 2), "uncached_per_turn": round(g["uncached"] / turns, 2), "cache_read_per_turn": round(g["cache_read"] / turns, 2), "cache_create_per_turn": round(g["cache_create"] / turns, 2), "output_per_turn": round(g["output"] / turns, 2), "cache_read_rate": round(g["cache_read"] / total, 4) if total else 0, "median_first_request_total_input": statistics.median(g["first_inputs"]), "polling_turns": g["polling_turns"]})
    turns = sum(x["turns"] for x in rows)
    pr_rows = [{"pr": pr, "workers": len(v["worker"]), "reviewers": len(v["reviewer"]), "workers_3_plus": len(v["worker"]) >= 3, "reviewers_3_plus": len(v["reviewer"]) >= 3} for pr, v in sorted(prs.items(), key=lambda x: int(x[0]))]
    return {"groups": rows, "evictions": {**evictions, "per_1000_turns": round(evictions["count"] * 1000 / turns, 2) if turns else 0}, "pull_requests": pr_rows, "skipped_rows": skipped, "turns": turns}


def print_report(data: dict) -> None:
    print(f"turns: {data['turns']}  skipped rows: {data['skipped_rows']}")
    print("role/model                         agents turns/agent uncached cache-read cache-create output cache-rate first-input polling")
    for x in data["groups"]:
        print(f"{x['role']}/{x['model']:<27} {x['agents']:>6} {x['turns_per_agent']:>11} {x['uncached_per_turn']:>8} {x['cache_read_per_turn']:>10} {x['cache_create_per_turn']:>11} {x['output_per_turn']:>6} {x['cache_read_rate']:>9.1%} {x['median_first_request_total_input']:>10} {x['polling_turns']:>7}")
    e = data["evictions"]
    print(f"evictions: {e['count']} rows, {e['bytes']} bytes, {e['per_1000_turns']} per 1,000 turns")
    workers = sum(x["workers_3_plus"] for x in data["pull_requests"])
    reviewers = sum(x["reviewers_3_plus"] for x in data["pull_requests"])
    print(f"exact-pr groups: {len(data['pull_requests'])}; workers with 3+: {workers}/{sum(x['workers'] > 0 for x in data['pull_requests'])}; reviewers with 3+: {reviewers}/{sum(x['reviewers'] > 0 for x in data['pull_requests'])}")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); session = root / "sessions" / "s" / "agents" / "1"; session.mkdir(parents=True)
        (root / "sessions" / "s" / "meta.json").write_text('{"model":"test"}')
        (session / "agent_lifecycle.json").write_text('{"agent_type":"worker","description":"fix PR #12"}')
        row = {"type":"message","data":{"message":{"role":"assistant","content":[{"text":"sleep 1"}],"metadata":{"output_tokens":3,"provider_usage":{"input_tokens":10,"cache_read_input_tokens":4,"cache_creation_input_tokens":2}}}}}
        (session / "conversation.jsonl").write_text(json.dumps({"type":"header","data":{"created_at":"2026-01-01T00:00:00Z"}})+"\n"+json.dumps(row)+"\n")
        result = report(root, ["s"], None, None)
        assert result["turns"] == 1 and result["groups"][0]["uncached_per_turn"] == 10 and result["groups"][0]["polling_turns"] == 1
        assert result["pull_requests"][0]["pr"] == "12"
        outside = root / "outside"; outside.write_text("x")
        (root / "sessions" / "escape").symlink_to(outside)
        assert not report(root, None, None, None)["groups"] or all(x["agents"] == 1 for x in report(root, None, None, None)["groups"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_ids", nargs="*"); parser.add_argument("--since"); parser.add_argument("--until"); parser.add_argument("--zeta-home", default=os.environ.get("ZETA_HOME", "~/.zeta")); parser.add_argument("--json", action="store_true"); parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check: ok")
    else:
        data = report(Path(args.zeta_home), args.session_ids or None, parse_time(args.since), parse_time(args.until))
        print(json.dumps(data, indent=2) if args.json else print_report(data))
