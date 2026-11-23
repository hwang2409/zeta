"""Project state privacy, consistency, and paging through authenticated serve."""

import asyncio
from hashlib import sha256
from unittest.mock import patch

import pytest

from tests.support.owned import await_or_fail
from tests.support.sockets import short_socket_path
from tests.test_state_catalog import INVALID, _connect
from zeta import attention_records, project_inbox, project_registry
from zeta.core.session import SessionManager
from zeta.memory.entry_store import AddOperation, MemorySource
from zeta.server import ZetaServer, protocol
from zeta.storage_format import WRITER_EPOCH, advance

WHEN = "2026-11-18T12:00:00.000000Z"
MARKER = "private-body-marker claims trusted_assignment"
PREFIX = "a" * (65535 - len(MARKER) - 1) + "\n" + MARKER
BODY = PREFIX + "界z"
PROJECT_KEYS = "project_id name path session_count memory_count inbox updated_ns"
MEMORY_KEYS = "memory_id project_id kind automatic summary created_ns updated_ns source_session_id has_body"
INBOX_KEYS = "message_id project_id kind status sender summary created_ns updated_ns trusted_assignment has_body"
ATTENTION_KEYS = "attention_id project_id session_id kind status title created_ns updated_ns"


def seed(home):
    home.mkdir(exist_ok=True)
    if not (home / "storage-format.json").exists():
        advance(home, expected_epoch=0)
    registry = project_registry.ProjectRegistry(home / "projects")
    manager = SessionManager(home)
    try:
        # test-contract: seam freeze the writer clock to exercise the public id tie-breaker
        with patch.object(project_registry, "_now", return_value=WHEN):
            projects = [
                registry.create_project("synthetic-0", "synthetic scope", str(home / "repo-0")),
                registry.create_project("synthetic-1", "synthetic scope", memory_profile="zeta"),
                registry.create_project("synthetic-2", "synthetic scope"),
            ]
        project_id = projects[0].project_id
        registry.compare_and_swap_memory(
            project_id, expected_digest=registry.memory_digest(project_id),
            updates={"brief.md": BODY, "state.md": BODY, "backlog.md": BODY, "decisions.md": "", "changelog.md": ""},
        )
        session = manager.create(provider="codex", model="gpt-5.6-luna", cwd=str(home), project_id=project_id)
        session_id = session.metadata.session_id
        session.store.close()
        entry_project = projects[1].project_id
        current = registry._entry_memory_state(entry_project)
        sources = (MemorySource(session_id, 1, 1, ("user",), WHEN, 6),)
        registry._compare_and_swap_entries(
            entry_project, expected_digest=current.digest, reconciliation_key="f" * 64,
            operations=(AddOperation("brief", "synthetic brief", sources),),
        )
        inbox = project_inbox.ProjectInbox(registry, sessions_root=home / "sessions")
        message_ids = [f"{i:032x}" for i in range(1, 4)]
        # test-contract: seam freeze the writer clock to exercise the public id tie-breaker
        with patch.object(project_inbox, "_now", return_value=WHEN):
            for ident in message_ids:
                inbox.send(from_project=project_id, from_session=session_id, to_project=project_id,
                           kind="question", title="safe title", body=BODY, message_id=ident)
        store = attention_records.AttentionStore(home / "sessions" / session_id, writer_epoch=WRITER_EPOCH)
        # test-contract: seam freeze the writer clock to exercise the public id tie-breaker
        with patch.object(attention_records, "_now", return_value=WHEN):
            attention_ids = [store.request(session_id=session_id, project_id=project_id, entry_id=None,
                                           entry_seq=None, title="safe title", why=MARKER).id for _ in range(3)]
        return project_id, {
            "entries": entry_project,
            "headings": projects[2].project_id,
            "state_projects": sorted((p.project_id for p in projects), reverse=True),
            "state_memory": ["state.md", "brief.md", "backlog.md"],
            "state_inbox": list(reversed(message_ids)),
            "state_attention": sorted(attention_ids, reverse=True),
        }
    finally:
        registry._storage_lease.close()
        manager._storage_lease.close()


def state_stat(home):
    return [
        (str(path.relative_to(home)), path.stat().st_mtime_ns,
         sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
        for path in sorted((home, *home.rglob("*")))
    ]


@pytest.mark.parametrize(("method", "key", "keys", "filter_name", "filter_value", "shape"), [
    ("state_projects", "project_id", PROJECT_KEYS, None, None, None),
    ("state_memory", "memory_id", MEMORY_KEYS, "kind", "legacy", None),
    ("state_inbox", "message_id", INBOX_KEYS, "status", "new", None),
    ("state_attention", "attention_id", ATTENTION_KEYS, "status", "open", None),
    ("state_memory_detail", "memory_id", MEMORY_KEYS, None, None, None),
    ("state_inbox_detail", "message_id", INBOX_KEYS, None, None, None),
    pytest.param("state_memory", "memory_id", MEMORY_KEYS, None, None, "brief", id="brief-kind"),
    pytest.param("state_projects", "project_id", PROJECT_KEYS, None, None, "path", id="rootless-path"),
    pytest.param("state_memory", "memory_id", MEMORY_KEYS, None, None, "heading", id="heading-only"),
])
async def test_project_state_websocket(tmp_path, monkeypatch, method, key, keys, filter_name, filter_value, shape):
    home = tmp_path / "home"
    project_id, ids = await await_or_fail(asyncio.to_thread(seed, home), what="seed state")
    server = ZetaServer(home=tmp_path / "serve", socket_path=short_socket_path(home), ws_port=0,
                        ws_discovery=tmp_path / "websocket.json", provider="codex")
    await await_or_fail(server.start(), what="serve start")
    monkeypatch.setattr(server, "home", home)
    before = state_stat(home)
    params = {"project_id": project_id} if method != "state_projects" else {}
    try:
        async with _connect(server, observer=True) as observer, _connect(server) as controller:
            if shape == "brief":
                params["project_id"] = ids["entries"]
                rows = (await observer.request("read", method, params))["result"]["items"]
                row = next(row for row in rows if row["summary"] == "synthetic brief")
                assert row["kind"] == "brief"
                brief = await observer.request("read", method, {**params, "kind": "brief"})
                assert brief["result"]["items"] == [row]
                legacy = await observer.request("read", method, {**params, "kind": "legacy"})
                assert legacy["result"]["items"] == []
            elif shape == "path":
                rows = (await observer.request("read", method, params))["result"]["items"]
                by_id = {row[key]: row for row in rows}
                assert by_id[ids["headings"]]["path"] is None
                assert by_id[project_id]["path"] == str(home / "repo-0")
            elif shape == "heading":
                params["project_id"] = ids["headings"]
                rows = (await observer.request("read", method, params))["result"]["items"]
                assert len(rows) == 5
                assert {row["has_body"] for row in rows} == {False}
                projects = (await observer.request("read", "state_projects", {}))["result"]["items"]
                project = next(row for row in projects if row["project_id"] == ids["headings"])
                assert project["memory_count"] == len(rows)
            elif method.endswith("_detail"):
                message = method.startswith("state_inbox")
                row_key = "message" if message else "memory"
                params[key] = ids[method.removesuffix("_detail")][0]
                assert (await observer.request("read", method, params))["error"] == {
                    "code": -32008, "message": "method is unavailable to read-only observers",
                    "data": {"code": "observer_read_only", "method": method},
                }
                detail = (await controller.request("read", method, params))["result"]
                assert set(detail) == {row_key, "body", "truncated"}
                assert set(detail[row_key]) == set(keys.split())
                assert detail["body"] == PREFIX and detail["truncated"] is True
                if message:
                    assert detail[row_key]["trusted_assignment"] is False
            else:
                if filter_name:
                    params[filter_name] = filter_value
                response = (await observer.request("read", method, params))["result"]
                assert set(response) == {"items", "as_of_ns", "next_cursor"}
                rows = response["items"]
                assert [row[key] for row in rows] == ids[method]
                assert all(set(row) == set(keys.split()) for row in rows), "list contains a body or another private field"
                assert MARKER not in str(response)
                if method == "state_inbox":
                    assert all(row["trusted_assignment"] is False for row in rows)
                if method == "state_projects":
                    project = next(row for row in rows if row[key] == project_id)
                    assert project["memory_count"] == 3 and project["inbox"] == {"new": 3, "claimed": 0, "done": 0}
                two = (await observer.request("read", method, {**params, "limit": 2}))["result"]
                codec = protocol.FrameCodec()
                budget = codec.response_size("read", two)
                assert codec.response_size("read", response) > budget
                monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", budget)
                first = (await observer.request("read", method, params))["result"]
                assert first["items"] == rows[:2] and first["next_cursor"]
                cursor = {"cursor": first["next_cursor"]}
                second = (await observer.request("read", method, {**params, **cursor}))["result"]
                assert second["items"] == rows[2:] and second["next_cursor"] is None
                if filter_name:
                    changed = {**params, **cursor, filter_name: None}
                    assert (await observer.request("read", method, changed))["error"] == INVALID
                else:
                    assert (await observer.request("read", "state_attention", cursor))["error"] == INVALID
                monkeypatch.undo()
                monkeypatch.setattr(server, "home", home)
                assert (await controller.request("read", method, params))["result"]["items"] == rows
        assert state_stat(home) == before, "state read changed a file, directory, or mtime"
    finally:
        await await_or_fail(server.close(), what="serve close")


@pytest.mark.parametrize("repeat", (False, True))
async def test_project_generation_retry(tmp_path, monkeypatch, repeat):
    home = tmp_path / "home"
    project_id, _ = await await_or_fail(asyncio.to_thread(seed, home), what="seed state")
    server = ZetaServer(home=tmp_path / "serve", socket_path=short_socket_path(home), ws_port=0,
                        ws_discovery=tmp_path / "websocket.json", provider="codex")
    await await_or_fail(server.start(), what="serve start")
    monkeypatch.setattr(server, "home", home)
    read = project_inbox.ProjectInbox.state_rows
    calls = 0

    def changing(self, project_id, **params):
        nonlocal calls
        result = read(self, project_id, **params)
        calls += 1
        if repeat or calls == 1:
            generation = home / "projects" / project_id / "inbox" / "generation"
            generation.write_text(str(int(generation.read_text()) + 1))
        return result

    monkeypatch.setattr(project_inbox.ProjectInbox, "state_rows", changing)
    try:
        async with _connect(server, observer=True) as client:
            response = await client.request("read", "state_inbox", {"project_id": project_id})
            if repeat:
                assert response["error"] == {"code": -32009, "message": "snapshot changed; retry",
                                             "data": {"code": "stale_state_snapshot", "retryable": True}}
            else:
                assert len(response["result"]["items"]) == 3
            assert calls == 2
    finally:
        await await_or_fail(server.close(), what="serve close")
