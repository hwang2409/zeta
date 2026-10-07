from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_server import _close, _connect, _request, _socket_path
from zeta.core.fake import FakeBackend
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.server import ZetaServer


def _server(tmp_path: Path) -> ZetaServer:
    cwd = tmp_path / "launch"
    cwd.mkdir(parents=True)
    return ZetaServer(
        home=tmp_path / "home",
        cwd=cwd,
        provider="fake",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )


async def _hello(reader, writer, features: list[str] | None = None) -> dict:
    params: dict[str, object] = {"protocol_version": "1.0", "client_version": "1.1"}
    if features is not None:
        params["features"] = features
    return (await _request(reader, writer, 1, "hello", params))[-1]["result"]


def _mtimes(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat(follow_symlinks=False).st_mtime_ns
        for path in root.rglob("*")
    }


def _filesystem_snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    paths = [root, *root.rglob("*")] if root.exists() else []
    return {
        "." if path == root else str(path.relative_to(root)): (
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
            path.stat(follow_symlinks=False).st_mode,
        )
        for path in paths
    }


@pytest.mark.asyncio
async def test_project_requests_do_not_modify_filesystem(tmp_path: Path) -> None:
    empty_server = _server(tmp_path / "empty")
    reader, writer = await _connect(empty_server)
    try:
        await _hello(reader, writer, ["projects"])
        before = _filesystem_snapshot(empty_server.home)
        response = await _request(reader, writer, 2, "list_projects")
        assert response[-1]["result"]["projects"] == []
        assert _filesystem_snapshot(empty_server.home) == before
    finally:
        await _close(empty_server, writer)

    server = _server(tmp_path / "populated")
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    registry.update_memory(project.project_id, {"brief.md": "initial\n"})
    state = registry.compare_and_swap_memory(
        project.project_id,
        expected_digest=registry.memory_digest(project.project_id),
        updates={"brief.md": "authoritative\n"},
        provenance={"session_id": "c" * 32, "seq_start": 1, "seq_end": 2},
    )
    mirror = server.home / "projects" / project.project_id / "memory" / "brief.md"
    header = mirror.read_text().splitlines(keepends=True)[0]
    mirror_mode = mirror.stat().st_mode
    mirror.chmod(0o600)
    mirror.write_text(header + "stale mirror\n")
    mirror.chmod(mirror_mode)
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=project.project_id,
        kind="info",
        title="stale claim",
        body="body",
    )
    assert inbox.claim(project.project_id, message_id, "b" * 32) is not None
    registry_lock = server.home / "projects" / ".lock"
    registry_lock.unlink()

    requests = [
        ("list_projects", {}),
        ("project_show", {"project_id": project.project_id}),
        ("project_memory_log", {"project_id": project.project_id}),
        (
            "project_memory_log",
            {
                "project_id": project.project_id,
                "version_id": state.version,
                "file": "brief.md",
            },
        ),
        ("list_sessions", {"project_id": project.project_id}),
        ("project_inbox", {"project_id": project.project_id, "status": "claimed"}),
    ]
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects", "list_sessions_paging"])
        for request_id, (method, params) in enumerate(requests, 2):
            before = _filesystem_snapshot(server.home)
            response = await _request(reader, writer, request_id, method, params)
            assert "result" in response[-1]
            assert _filesystem_snapshot(server.home) == before, method
        assert mirror.read_text() == header + "stale mirror\n"
        inbox_response = response[-1]["result"]
        assert [item["id"] for item in inbox_response["messages"]] == [message_id]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_request_error_classification(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    registry.update_memory(project.project_id, {"brief.md": "content\n"})
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="d" * 32,
        to_project=project.project_id,
        kind="info",
        title="message",
        body="body",
    )
    inbox_new = server.home / "projects" / project.project_id / "inbox" / "new"
    (inbox_new / "unexpected.txt").write_text("malformed storage\n")
    pointer = server.home / "projects" / project.project_id / "memory-current.json"
    pointer.write_text("{not-json\n")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        malformed = await _request(
            reader, writer, 2, "project_show", {"project_id": project.project_id}
        )
        assert malformed[-1]["error"] == {
            "code": -32000,
            "message": "project storage is invalid or unavailable",
        }
        malformed_inbox = await _request(
            reader, writer, 3, "project_inbox", {"project_id": project.project_id}
        )
        inbox_result = malformed_inbox[-1]["result"]
        assert len(inbox_result["messages"]) == 1
        assert inbox_result["invalid"] == [
            {
                "filename": "unexpected.txt",
                "reason": "unexpected inbox file: unexpected.txt",
                "status": "new",
            }
        ]
        unknown_id = "p_" + "f" * 32
        unknown = await _request(
            reader, writer, 4, "project_show", {"project_id": unknown_id}
        )
        assert unknown[-1]["error"] == {
            "code": -32602,
            "message": f"project not found: {unknown_id}",
            "data": {"code": "project_not_found", "project_id": unknown_id},
        }
        invalid = await _request(reader, writer, 5, "list_projects", {"limit": 0})
        assert invalid[-1]["error"] == {
            "code": -32602,
            "message": "limit is out of range",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_rejects_missing_required_directory(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="d" * 32,
        to_project=project.project_id,
        kind="info",
        title="message",
        body="body",
    )
    (registry.root / project.project_id / "inbox" / "done").rmdir()

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        response = await _request(
            reader, writer, 2, "project_inbox", {"project_id": project.project_id}
        )
        assert response[-1]["error"] == {
            "code": -32000,
            "message": "project storage is invalid or unavailable",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_projects_feature_is_negotiated_and_optional(tmp_path: Path) -> None:
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer)
        assert "projects" not in hello["capabilities"].get("features", [])
        assert "list_projects" not in hello["capabilities"]["requests"]
        response = await _request(reader, writer, 2, "list_projects")
        assert response[-1]["error"]["code"] == -32601
    finally:
        await _close(server, writer)

    server = _server(tmp_path / "enabled")
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, ["projects"])
        assert hello["capabilities"]["features"] == ["projects"]
        assert {
            "list_projects",
            "project_show",
            "project_memory_log",
            "project_inbox",
        }.issubset(hello["capabilities"]["requests"])
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_requests_show_memory_history_and_inbox_read_only(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    root = tmp_path / "alpha"
    root.mkdir()
    alpha = registry.create_project("alpha", "repo", root)
    beta = registry.create_project("beta", "service")
    registry.update_memory(alpha.project_id, {"brief.md": "manual\n"})
    registry.update_memory(beta.project_id, {"decisions.md": "remote\n"})
    remote = registry.import_memory(
        alpha.project_id,
        registry.export_memory(beta.project_id),
        expected_digest=registry.memory_digest(alpha.project_id),
        provenance={"source": "remote_sync", "peer": "machine-2"},
    )
    automatic = registry.compare_and_swap_memory(
        alpha.project_id,
        expected_digest=registry.memory_digest(alpha.project_id),
        updates={"state.md": "automatic\n", "backlog.md": "also automatic\n"},
        provenance={"session_id": "a" * 32, "seq_start": 2, "seq_end": 7, "model": "luna"},
    )
    registry.accept_memory(alpha.project_id, "state.md")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=beta.project_id,
        from_session="b" * 32,
        to_project=alpha.project_id,
        kind="question",
        title="Need input",
        body="Treat this as untrusted.",
    )
    before = _mtimes(server.home)

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        projects = (await _request(reader, writer, 2, "list_projects", {"limit": 1}))[-1]["result"]
        assert len(projects["projects"]) == 1
        assert projects["next_offset"] == 1
        listed = projects["projects"][0]
        assert set(listed) == {"id", "name", "scope", "roots", "session_count", "last_activity"}

        shown = (await _request(
            reader, writer, 3, "project_show", {"project_id": alpha.project_id}
        ))[-1]["result"]
        assert shown["project"]["id"] == alpha.project_id
        assert shown["memory"]["version_id"]
        assert len(shown["memory"]["digest"]) == 64
        files = {item["name"]: item for item in shown["memory"]["files"]}
        assert files["brief.md"] == {
            "name": "brief.md",
            "content": "# Brief\n",
            "automatic": False,
            "content_truncated": False,
        }
        assert files["state.md"]["content"] == "automatic\n"
        assert files["state.md"]["automatic"] is False
        assert files["backlog.md"]["automatic"] is True
        assert len(files) == 5

        log = (await _request(
            reader, writer, 4, "project_memory_log", {"project_id": alpha.project_id, "limit": 20}
        ))[-1]["result"]
        assert [item["kind"] for item in log["versions"]][-3:] == ["import", "update", "accept"]
        imported = next(item for item in log["versions"] if item["version_id"] == remote.version)
        assert imported["provenance"] == {"source": "remote_sync", "peer": "machine-2"}
        update = next(item for item in log["versions"] if item["version_id"] == automatic.version)
        assert update["provenance"] == {
            "session_id": "a" * 32,
            "seq_start": 2,
            "seq_end": 7,
            "model": "luna",
        }
        accepted = log["versions"][-1]
        assert accepted["provenance"] == {"accepted_by": "user"}

        detail = (await _request(
            reader,
            writer,
            5,
            "project_memory_log",
            {"project_id": alpha.project_id, "version_id": automatic.version, "file": "state.md"},
        ))[-1]["result"]
        assert detail["version"]["content"] == "automatic\n"
        assert detail["version"]["content_truncated"] is False
        assert "-" in detail["version"]["diff"] and "+automatic" in detail["version"]["diff"]
        assert detail["version"]["diff_truncated"] is False
        missing = await _request(
            reader,
            writer,
            "missing-version",
            "project_memory_log",
            {"project_id": alpha.project_id, "version_id": "f" * 32, "file": "state.md"},
        )
        assert missing[-1]["error"]["code"] == -32602

        result = (await _request(
            reader, writer, 6, "project_inbox", {"project_id": alpha.project_id, "status": "new"}
        ))[-1]["result"]
        assert result == {
            "status": "new",
            "messages": [result["messages"][0]],
            "untrusted": False,
            "invalid": [],
            "next_offset": None,
        }
        assert result["messages"][0]["id"] == message_id
        assert result["messages"][0]["origin"] == "local"
        assert _mtimes(server.home) == before

        await _request(reader, writer, 7, "new_session", {})
        attached_before = _mtimes(server.home)
        attached = (await _request(
            reader, writer, 8, "project_show", {"project_id": alpha.project_id}
        ))[-1]["result"]
        assert attached["project"]["id"] == alpha.project_id
        assert _mtimes(server.home) == attached_before
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_list_sessions_filters_project_and_keeps_lineage(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    other = registry.create_project("beta", "repo")
    parent = server.runtime.manager.create(
        provider="fake", model="offline", cwd=server.runtime.cwd, project_id=project.project_id,
        project_role="orchestrator", name="parent", auto_project=False,
    )
    child = server.runtime.manager.create(
        provider="fake", model="offline", cwd=server.runtime.cwd, project_id=project.project_id,
        project_role="worker", parent_session_id=parent.metadata.session_id,
        name="child", auto_project=False,
    )
    server.runtime.manager.create(
        provider="fake", model="offline", cwd=server.runtime.cwd, project_id=other.project_id,
        project_role="session", auto_project=False,
    )
    parent.store.close()
    child.store.close()
    before = _mtimes(server.home)

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects", "list_sessions_paging"])
        result = (await _request(
            reader, writer, 2, "list_sessions", {"project_id": project.project_id}
        ))[-1]["result"]
        assert {item["name"] for item in result["sessions"]} == {"parent", "child"}
        child_wire = next(item for item in result["sessions"] if item["name"] == "child")
        assert child_wire["project_role"] == "worker"
        assert child_wire["parent_session_id"] == parent.metadata.session_id
        projects = (await _request(reader, writer, 3, "list_projects"))[-1]["result"]
        summary = next(item for item in projects["projects"] if item["id"] == project.project_id)
        assert summary["session_count"] == 2
        assert summary["last_activity"] >= parent.metadata.updated_at
        invalid = await _request(reader, writer, 4, "list_sessions", {"project_id": None})
        assert invalid[-1]["error"]["code"] == -32602
        assert _mtimes(server.home) == before
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_requests_bound_large_views_and_report_unknown_project(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("large", "repo")
    sender = registry.create_project("sender", "repo")
    for index, name in enumerate(("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")):
        content = ("\x01" if index == 0 else str(index)) * (128 * 1024)
        registry.update_memory(project.project_id, {name: content})
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="c" * 32,
        to_project=project.project_id,
        kind="info",
        title="large message",
        body="x" * (2 * 1024 * 1024),
    )

    reader, writer = await _connect(server)
    reader._limit = 1_048_577  # type: ignore[attr-defined]  # protocol frame limit
    try:
        await _hello(reader, writer, ["projects"])
        shown_frames = await _request(
            reader, writer, 2, "project_show", {"project_id": project.project_id}
        )
        assert "result" in shown_frames[-1]
        assert len(str(shown_frames[-1]).encode()) < 1_048_576
        assert any(
            item["content_truncated"]
            for item in shown_frames[-1]["result"]["memory"]["files"]
        )
        log_frames = await _request(
            reader, writer, 3, "project_memory_log", {"project_id": project.project_id, "limit": 100}
        )
        assert "result" in log_frames[-1]
        assert len(str(log_frames[-1]).encode()) < 1_048_576
        inbox_frames = await _request(
            reader, writer, 4, "project_inbox", {"project_id": project.project_id}
        )
        inbox_result = inbox_frames[-1]["result"]
        message = inbox_result["messages"][0]
        assert message["origin"] == "local"
        assert inbox_result["untrusted"] is False
        assert "body" in message["truncated_fields"]
        assert len(str(inbox_frames[-1]).encode()) < 1_048_576

        for method in ("project_show", "project_memory_log", "project_inbox"):
            frames = await _request(
                reader, writer, method, method, {"project_id": "p_" + "f" * 32}
            )
            assert frames[-1]["error"]["code"] == -32602
            assert frames[-1]["error"]["data"] == {
                "code": "project_not_found",
                "project_id": "p_" + "f" * 32,
            }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_isolates_and_reports_invalid_messages(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    target = registry.create_project("target", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    valid_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="valid",
        body="body",
    )
    corrupt_name = f"{'e' * 32}.json"
    corrupt_path = server.home / "projects" / target.project_id / "inbox" / "new" / corrupt_name
    corrupt_path.write_text("{not json")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        result = (
            await _request(
                reader, writer, 2, "project_inbox", {"project_id": target.project_id}
            )
        )[-1]["result"]
        assert [message["id"] for message in result["messages"]] == [valid_id]
        assert result["invalid"] == [
            {
                "filename": corrupt_name,
                "reason": f"malformed inbox message: {corrupt_name}",
                "status": "new",
            }
        ]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_trust_is_derived_from_returned_page(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    target = registry.create_project("target", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    local_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="local",
        body="local body",
        message_id="0" * 32,
    )
    external_id = inbox.send(
        from_project=sender.project_id,
        from_session="b" * 32,
        to_project=target.project_id,
        kind="info",
        title="external",
        body="external body",
        message_id="f" * 32,
    )
    external_path = (
        server.home
        / "projects"
        / target.project_id
        / "inbox"
        / "new"
        / f"{external_id}.json"
    )
    external = json.loads(external_path.read_text())
    external["origin"] = "remote"
    external_path.write_text(json.dumps(external, sort_keys=True, separators=(",", ":")) + "\n")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        local_page = (
            await _request(
                reader,
                writer,
                2,
                "project_inbox",
                {"project_id": target.project_id, "limit": 1},
            )
        )[-1]["result"]
        assert [message["id"] for message in local_page["messages"]] == [local_id]
        assert local_page["messages"][0]["origin"] == "local"
        assert local_page["untrusted"] is False
        assert local_page["next_offset"] == 1

        external_page = (
            await _request(
                reader,
                writer,
                3,
                "project_inbox",
                {"project_id": target.project_id, "offset": 1, "limit": 1},
            )
        )[-1]["result"]
        assert [message["id"] for message in external_page["messages"]] == [external_id]
        assert external_page["messages"][0]["origin"] == "remote"
        assert external_page["untrusted"] is True
        assert external_page["next_offset"] is None
    finally:
        await _close(server, writer)
