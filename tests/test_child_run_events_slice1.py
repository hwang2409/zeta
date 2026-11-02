from __future__ import annotations

import pytest

from zeta.core.store import ConversationStore
from zeta.core.store._child_runs import parse_child_run_id


def test_nested_ids_resolve_lineage() -> None:
    assert parse_child_run_id("session", "session:4") == (4, None)
    assert parse_child_run_id("session", "session:4:2") == (2, "session:4")


@pytest.mark.parametrize(
    "run_id",
    [
        "../",
        "session/1",
        "session\\1",
        "session:%2f1",
        "session:\x001",
        "session:+1",
        "session:0",
        "session:01",
        "other:1",
        "session:",
    ],
)
def test_malformed_ids_never_resolve(run_id: str) -> None:
    with pytest.raises(ValueError):
        parse_child_run_id("session", run_id)


def test_overlong_and_deep_ids_rejected() -> None:
    with pytest.raises(ValueError):
        parse_child_run_id("session", "session:" + ":".join("1" for _ in range(65)))
    with pytest.raises(ValueError):
        parse_child_run_id("session", "session:" + "1" * 250)


def test_child_run_started_is_durable_and_loadable(tmp_path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    try:
        entry = store.append_child_run_started(
            "session:1", None, "worker", "do work", "2026-11-02T00:00:00+00:00"
        )
        assert entry.type == "child_run_started"
    finally:
        store.close()
    reopened = ConversationStore(tmp_path, session_id="session", _read_only=True)
    try:
        assert reopened.entries[0].type == "child_run_started"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_child_run_started_async(tmp_path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    try:
        entry = await store.append_child_run_started_async(
            "session:1", None, None, "task", "2026-11-02T00:00:00+00:00"
        )
        assert entry.type == "child_run_started"
    finally:
        store.close()
