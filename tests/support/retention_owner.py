"""shared retention owner fixtures and production-path helpers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from zeta.core.session import SessionManager
from zeta.session_retention import RetentionPlanner
from zeta.storage_format import open_writer

_DAY_NS = 86_400 * 1_000_000_000
_OLD = "2020-01-01T00:00:00+00:00"


def _manager(home):
    return SessionManager(home)


def _create_closed(manager, stack, **kwargs):
    opened = manager.create(
        provider="codex", model="gpt-5.6-sol", cwd=str(manager.home),
        auto_project=False, **kwargs,
    )
    opened.store.close()
    return opened.metadata.session_id


def _true_tree_bytes(session_dir) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(session_dir):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            total += os.path.getsize(path)
    return total


def _planner(stack, *args, **kwargs):
    planner = RetentionPlanner(*args, **kwargs)
    stack.callback(getattr(planner, "close", lambda: None))
    return planner


def _age_tree(session_dir, *, days=100):
    """Make a session look old: the root header time and every mtime."""
    _set_header_created_at(session_dir, _OLD)
    old = time.time_ns() - days * _DAY_NS
    for dirpath, _dirs, files in os.walk(session_dir, topdown=False):
        for name in files:
            os.utime(os.path.join(dirpath, name), ns=(old, old), follow_symlinks=False)
        os.utime(dirpath, ns=(old, old))


def _set_header_created_at(session_dir, text):
    transcript = session_dir / "conversation.jsonl"
    header, newline, rest = transcript.read_bytes().partition(b"\n")
    value = json.loads(header)
    value["data"]["created_at"] = text
    transcript.write_bytes(json.dumps(value).encode() + newline + rest)


def _run_sweep(planner, days=30):
    reasons = {}
    for _ in range(500):
        result = planner.run_pass(days=days)
        reasons.update({row.root_id: (row.closed, row.reason) for row in result.reasons})
        if result.complete:
            return result, reasons
    raise AssertionError("sweep did not complete in 500 passes")


def _run_sweep_without_error(planner, days=30):
    try:
        return _run_sweep(planner, days)
    except OSError:
        return None, {}


@contextmanager
def _opened_lease(home):
    lease = open_writer(home)
    try:
        yield lease
    finally:
        lease.close()
