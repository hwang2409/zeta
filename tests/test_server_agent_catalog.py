"""The catalog's observer contract and bounded continuation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract
from zeta.core import agent_run_catalog as catalog
from zeta.storage_format import open_writer

METHOD = "agent_run_catalog"


def _child(root: Path, run_id: str, *, broken: bool = False) -> Path:
    sid, *segments = run_id.split(":")
    path = root / sid
    for segment in segments:
        path = path / "agents" / segment
    path.mkdir(parents=True, exist_ok=True)
    lifecycle = {
        "handle": run_id, "state": "completed", "depth": len(segments),
        "started_at": "2026-11-19T00:00:00+00:00", "finished_at": "2026-11-19T00:00:02+00:00",
        "current_step": "completed", "description": "test child", "agent_type": "worker",
        "turns_used": 2, "tool_calls": 1, "elapsed": 2, "final_result": "hidden result canary",
    }
    (path / "agent_lifecycle.json").write_text("{" if broken else json.dumps(lifecycle))
    (path / "conversation.jsonl").write_text("hidden transcript canary")
    return path


@pytest.fixture
def formatted_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    with open_writer(home):
        pass
    return home


@pytest.mark.asyncio
async def test_observer_catalog_pages_without_attaching_a_session(formatted_home: Path) -> None:
    sessions = formatted_home / "sessions"
    _child(sessions, "a:1", broken=True)
    _child(sessions, "a:1:2")
    _child(sessions, "b:1")
    _child(sessions, "older:1")
    for name, stamp in (("a", 2), ("b", 2), ("older", 1)):
        os.utime(sessions / name, ns=(stamp, stamp))
    contract = await ServeContract.start(formatted_home.parent, "websocket")
    try:
        plain = await contract.connect("controller")
        await plain.request("hello", "hello", {"protocol_version": "1.2"})
        assert (await plain.request("catalog", METHOD))["error"]["code"] == -32601
        observer = await contract.connect("observer")
        await observer.request("hello", "hello", {
            "protocol_version": "1.2", "role": "observer", "features": ["serve_observers", "agent_run_catalog_v1"],
        })
        raw = await observer.request_raw("first", METHOD, {"limit": 1, "state": "completed"})
        first = json.loads(raw)["result"]
        assert "hidden result canary" not in raw and "hidden transcript canary" not in raw
        assert len(first["revision"]) == 64
        assert first["data"]["items"] == [{
            "run_id": "a:1:2", "session_id": "a", "parent_run_id": "a:1", "description": "test child",
            "agent_type": "worker", "status": "completed", "accepts_follow_ups": False,
            "turns_used": 2, "tool_count": 1, "current_step": "completed", "elapsed_seconds": 2,
            "started_at": "2026-11-19T00:00:00+00:00", "finished_at": "2026-11-19T00:00:02+00:00",
        }]
        cursor = first["data"]["next_cursor"]
        second = await observer.request("next", METHOD, {"cursor": cursor, "state": "completed"})
        assert [row["run_id"] for row in second["result"]["data"]["items"]] == ["b:1", "older:1"]
        assert "next_cursor" not in second["result"]["data"]
        scoped = await observer.request("scoped", METHOD, {"root_session_id": "b"})
        assert [row["run_id"] for row in scoped["result"]["data"]["items"]] == ["b:1"]
        for name in range(100, 227):
            (sessions / "budget" / "agents" / str(name)).mkdir(parents=True)
        _child(sessions, "budget:999")
        budget = {"root_session_id": "budget"}
        empty = (await observer.request("budget-first", METHOD, budget))["result"]["data"]
        assert empty["items"] == []
        continued = await observer.request("budget-next", METHOD, {**budget, "cursor": empty["next_cursor"]})
        assert [row["run_id"] for row in continued["result"]["data"]["items"]] == ["budget:999"]
        for params in ({"cursor": cursor}, {"root_session_id": "../a"}, {"limit": True}, {"offset": 1}):
            error = (await observer.request("invalid", METHOD, params))["error"]
            assert (error["code"], error["data"]) == (-32602, {"code": "invalid_params"})
    finally:
        await contract.close()


def test_catalog_bounds_metadata_reads_across_nonmatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with open_writer(tmp_path):
        pass
    sessions = tmp_path / "sessions"
    for name in ("1", "10", "2"):
        _child(sessions, f"root:{name}")
    monkeypatch.setattr(catalog, "_MAX_CANDIDATES", 2)
    original = catalog._metadata
    reads = []

    def read(fd, name):
        reads.append(name)
        return original(fd, name)

    monkeypatch.setattr(catalog, "_metadata", read)
    after = None
    for expected_path, expected_count in (("1", 1), (None, 2)):
        rows, keys, after = catalog.read_page(tmp_path, root_id="root", state="running", after=after, limit=50)
        assert rows == keys == []
        assert (after[2] if after is not None else None) == expected_path
        assert reads == ["agent_lifecycle.json"] * expected_count
        reads.clear()
