from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeta.context import ContextAssembler
from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.session import SessionError, SessionManager
from zeta.tui.app import build_parser, create_app
from zeta.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent


def _args(*values: str):
    return build_parser().parse_args([*values, "--provider", "fake"])


def test_fresh_cli_session_writes_versioned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    session_dir = app.loop.store.session_dir

    assert session_dir.parent == tmp_path / "zeta-home" / "sessions"
    assert (session_dir / "conversation.jsonl").exists()
    metadata = json.loads((session_dir / "meta.json").read_text())
    assert metadata["version"] == 1
    assert metadata["session_id"] == app.loop.store.session_id
    assert metadata["provider"] == "fake"
    assert metadata["cwd"] == str(tmp_path)


def test_continue_requires_a_prior_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SessionError, match="no prior zeta session"):
        create_app(build_parser().parse_args(["--continue"]))


def test_continue_reopens_the_most_recent_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    manager = SessionManager(home)
    older = manager.create(provider="fake", model="offline", cwd=tmp_path)
    newer = manager.create(provider="fake", model="offline", cwd=tmp_path)
    older.metadata.updated_at = "2020-01-01T00:00:00+00:00"
    manager._write(older.metadata)
    newer.metadata.updated_at = "2030-01-01T00:00:00+00:00"
    manager._write(newer.metadata)

    app = create_app(build_parser().parse_args(["--continue"]))

    assert app.loop.store.session_id == newer.store.session_id


def test_resume_reopens_an_explicit_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    session_id = first.loop.store.session_id

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )

    assert resumed.loop.store.session_id == session_id


def test_resume_rejects_an_unknown_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))

    with pytest.raises(SessionError, match="was not found"):
        create_app(build_parser().parse_args(["--resume", "missing"]))


def test_resume_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--continue", "--resume", "session"])


def test_resume_provider_override_requires_force_and_records_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    session_id = first.loop.store.session_id

    with pytest.raises(SessionError, match="override rejected"):
        create_app(
            build_parser().parse_args(
                ["--resume", session_id, "--provider", "claude"]
            )
        )

    create_app(
        build_parser().parse_args(
            ["--resume", session_id, "--provider", "claude", "--force-provider"]
        )
    )
    metadata = json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )
    assert metadata["provider"] == "claude"
    assert metadata["override_audit"][-1]["provider"] == {
        "from": "fake",
        "to": "claude",
    }


@pytest.mark.asyncio
async def test_resume_replays_the_same_context_branch(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("first")])])
    loop = AgentLoop(backend, opened.store)
    [event async for event in loop.run_turn("hello")]

    resumed = manager.open(opened.store.session_id)
    expected = await ContextAssembler(opened.store).assemble()
    actual = await ContextAssembler(resumed.store).assemble()

    assert actual == expected


def test_resume_reemits_pending_approval_state(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    call = ToolCall("approval-1", "exec", {"command": "danger"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    resumed = manager.open(opened.store.session_id)

    assert resumed.store.pending_approvals() == [(call.id, call)]


def test_future_metadata_version_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    metadata_path = home / "sessions" / first.loop.store.session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["version"] = 99
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(SessionError, match="unsupported session metadata version"):
        SessionManager(home).open(first.loop.store.session_id)
