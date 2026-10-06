from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.remote_sync import (
    LocalTransport,
    RemoteSyncError,
    pull_project_memory,
    pull_session,
    push_project_memory,
    push_session,
    resolve_transport,
)
from zeta.remote_sync.ssh import SshTransport


def _git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:example/project.git"],
        cwd=path,
        check=True,
    )


def _session(home: Path, repo: Path):
    projects = ProjectRegistry(home / "projects")
    project = projects.create_project("test", "test", str(repo))
    projects.update_memory(project.project_id, {"brief.md": "shared\n"})
    opened = SessionManager(home).create(
        provider="fake", model="fake", cwd=repo, project_id=project.project_id
    )
    opened.store.append_message(Message(MessageRole.USER, [TextContent("root transcript")]))
    child = ConversationStore(
        opened.store.session_dir / "agents", session_id="1", cwd=repo
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("child transcript")]))
    child.close()
    spill = opened.store.session_dir / "spill"
    spill.mkdir(mode=0o700)
    (spill / "history.txt").write_text("spill is history\n", encoding="utf-8")
    background = opened.store.session_dir / "background"
    background.mkdir(mode=0o700)
    (background / "task.json").write_text('{"status":"exited"}\n', encoding="utf-8")
    return project, opened


def test_session_push_copies_consistent_history_and_excludes_credentials(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    repo = tmp_path / "repo"
    _git_repo(repo)
    project, opened = _session(local, repo)
    (local / "oauth.json").write_text("secret", encoding="utf-8")
    (local / "settings.toml").write_text('token = "secret"\n', encoding="utf-8")
    (opened.store.session_dir / "oauth.json").write_text("secret", encoding="utf-8")
    (opened.store.session_dir / "spill" / "oauth.json").write_text(
        "spill secret", encoding="utf-8"
    )

    result = push_session(
        local, LocalTransport(remote), session_id=opened.metadata.session_id
    )
    opened.store.append_message(Message(MessageRole.USER, [TextContent("written after snapshot")]))
    opened.store.close()

    destination = remote / "sessions" / result.session_id
    assert (destination / "agents" / "1" / "conversation.jsonl").is_file()
    assert (destination / "spill" / "history.txt").read_text() == "spill is history\n"
    assert (destination / "spill" / "oauth.json").read_text() == "spill secret"
    assert not (destination / "oauth.json").exists()
    assert (destination / "background" / "task.json").is_file()
    copied = (destination / "conversation.jsonl").read_text(encoding="utf-8")
    assert "root transcript" in copied
    assert "written after snapshot" not in copied
    assert all(json.loads(line) for line in copied.splitlines())
    assert not (remote / "oauth.json").exists()
    assert not (remote / "settings.toml").exists()
    assert (remote / "projects" / project.project_id / "memory" / "brief.md").read_text() == "shared\n"

    manifest = json.loads((destination / "transfer.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "zeta.session-transfer.v1"
    assert manifest["git"] == {
        "remote_url": "git@github.com:example/project.git",
        "branch": "master",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    }
    assert manifest["includes_spill_files"] is True


def test_push_refuses_newer_remote_without_force(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    repo = tmp_path / "repo"
    _git_repo(repo)
    _, opened = _session(local, repo)
    session_id = opened.metadata.session_id
    opened.store.close()
    transport = LocalTransport(remote)
    push_session(local, transport, session_id=session_id)

    remote_opened = SessionManager(remote).open(session_id)
    remote_opened.store.append_message(Message(MessageRole.USER, [TextContent("newer remote entry")]))
    remote_opened.store.close()

    with pytest.raises(RemoteSyncError, match="newer remote"):
        push_session(local, transport, session_id=session_id)
    push_session(local, transport, session_id=session_id, force=True)


def test_pull_maps_missing_cwd_and_records_reclone_hint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    destination = tmp_path / "destination"
    repo = tmp_path / "repo"
    mapped = tmp_path / "mapped"
    mapped.mkdir()
    _git_repo(repo)
    _, opened = _session(source, repo)
    session_id = opened.metadata.session_id
    opened.store.close()
    transport = LocalTransport(remote)
    push_session(source, transport, session_id=session_id)

    result = pull_session(
        destination, transport, session_id=session_id, cwd=mapped
    )

    imported = SessionManager(destination).open(session_id)
    assert imported.metadata.cwd == str(mapped.resolve())
    assert imported.store.cwd == str(mapped.resolve())
    imported.store.close()
    assert dict(ProjectRegistry(destination / "projects").load_memory(imported.metadata.project_id))["brief.md"] == "shared\n"
    assert "git@github.com:example/project.git" in result.resume_notice
    assert "clone" in result.resume_notice.lower()


def test_ssh_transport_uses_configured_alias_and_atomic_remote_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote home"
    repo = tmp_path / "repo"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "ssh"
    shim.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = -- ] && shift\n"
        "shift\n"
        "exec /bin/sh -c \"$1\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    local.mkdir()
    (local / "settings.toml").write_text(
        f'[remotes]\ncloud = "ssh://fake{remote}"\n', encoding="utf-8"
    )
    _git_repo(repo)
    _, opened = _session(local, repo)
    session_id = opened.metadata.session_id
    opened.store.close()

    transport = resolve_transport(local, "cloud")
    assert isinstance(transport, SshTransport)
    pushed = push_session(local, transport, session_id=session_id)
    assert pushed.session_id == session_id
    assert (remote / "sessions" / session_id / "transfer.json").is_file()

    pulled_home = tmp_path / "pulled"
    pull_session(pulled_home, transport, session_id=session_id)
    imported = SessionManager(pulled_home).open(session_id)
    imported.store.close()


def test_memory_pull_updates_when_only_remote_changed(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    repo = tmp_path / "repo"
    _git_repo(repo)
    project, opened = _session(local, repo)
    opened.store.close()
    transport = LocalTransport(remote)
    push_session(local, transport, session_id=opened.metadata.session_id)
    push_project_memory(local, transport, project_id=project.project_id)
    ProjectRegistry(remote / "projects").update_memory(
        project.project_id, {"brief.md": "remote only\n"}
    )

    result = pull_project_memory(local, transport, project_id=project.project_id)

    assert result.updated == ("brief.md",)
    assert not result.conflicts
    assert dict(ProjectRegistry(local / "projects").load_memory(project.project_id))[
        "brief.md"
    ] == "remote only\n"


def test_memory_push_uses_per_file_cas_and_keeps_conflicts(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    repo = tmp_path / "repo"
    _git_repo(repo)
    project, opened = _session(local, repo)
    opened.store.close()
    transport = LocalTransport(remote)
    push_session(local, transport, session_id=opened.metadata.session_id)
    push_project_memory(local, transport, project_id=project.project_id)

    local_projects = ProjectRegistry(local / "projects")
    remote_projects = ProjectRegistry(remote / "projects")
    local_projects.update_memory(project.project_id, {"brief.md": "local edit\n"})
    remote_projects.update_memory(project.project_id, {"brief.md": "remote edit\n"})

    result = push_project_memory(local, transport, project_id=project.project_id)

    assert result.conflicts == ("brief.md",)
    assert dict(remote_projects.load_memory(project.project_id))["brief.md"] == "remote edit\n"
    conflicts = list(
        (remote / "projects" / project.project_id / "memory").glob(
            "brief.md.conflict-local-*"
        )
    )
    assert len(conflicts) == 1
    assert conflicts[0].read_text(encoding="utf-8") == "local edit\n"
