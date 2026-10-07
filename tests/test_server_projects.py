from __future__ import annotations

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

        result = (await _request(
            reader, writer, 6, "project_inbox", {"project_id": alpha.project_id, "status": "new"}
        ))[-1]["result"]
        assert result == {"status": "new", "messages": [result["messages"][0]], "untrusted": True}
        assert result["messages"][0]["id"] == message_id
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
        assert _mtimes(server.home) == before
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_requests_bound_large_views_and_report_unknown_project(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("large", "repo")
    for index, name in enumerate(("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")):
        content = ("\x01" if index == 0 else str(index)) * (128 * 1024)
        registry.update_memory(project.project_id, {name: content})

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
