from __future__ import annotations

import json
import threading
import uuid
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.approval import ApprovalPolicy
from zeta.context import ContextAssembler
from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.session import SessionError, SessionManager
from zeta.tui.app import build_parser, create_app, main
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

    resumed = create_app(
        build_parser().parse_args(
            [
                "--resume",
                session_id,
                "--provider",
                "claude",
                "--model",
                "claude-sonnet-4-6",
                "--force-provider",
            ]
        )
    )
    metadata = json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )
    assert metadata["provider"] == "fake"
    assert metadata["model"] == "offline"
    assert metadata["override_audit"] == []
    assert resumed._pending_override == ("claude", "claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_forced_override_commits_after_first_successful_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    session_id = first.loop.store.session_id
    backend = FakeBackend([ScriptedTurn(content=[TextContent("ok")])])

    def build_backend(*args: object, **kwargs: object) -> tuple[FakeBackend, str]:
        del args, kwargs
        return backend, "claude-sonnet-4-6"

    monkeypatch.setattr("zeta.tui.app.build_backend", build_backend)
    app = create_app(
        build_parser().parse_args(
            [
                "--resume",
                session_id,
                "--provider",
                "claude",
                "--model",
                "claude-sonnet-4-6",
                "--force-provider",
            ]
        )
    )
    app._invalidate_prompt = lambda: None
    await app._consume_turn("hello")

    metadata = json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )
    assert metadata["provider"] == "claude"
    assert metadata["model"] == "claude-sonnet-4-6"
    assert metadata["override_audit"]


@pytest.mark.asyncio
async def test_invalid_model_keeps_forced_override_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    session_id = first.loop.store.session_id
    backend = FakeBackend([])

    def build_backend(*args: object, **kwargs: object) -> tuple[FakeBackend, str]:
        del args, kwargs
        return backend, "definitely-not-a-claude-model"

    monkeypatch.setattr("zeta.tui.app.build_backend", build_backend)
    app = create_app(
        build_parser().parse_args(
            [
                "--resume",
                session_id,
                "--provider",
                "claude",
                "--model",
                "definitely-not-a-claude-model",
                "--force-provider",
            ]
        )
    )
    app._invalidate_prompt = lambda: None
    await app._consume_turn("hello")

    metadata = json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )
    assert metadata["provider"] == "fake"
    assert metadata["model"] == "offline"
    assert metadata["override_audit"] == []


def test_force_provider_requires_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    first = create_app(_args())

    with pytest.raises(SessionError, match="requires --model"):
        create_app(
            build_parser().parse_args(
                [
                    "--resume",
                    first.loop.store.session_id,
                    "--provider",
                    "claude",
                    "--force-provider",
                ]
            )
        )


def test_main_rejects_force_provider_without_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--resume", "session", "--provider", "claude", "--force-provider"])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "--force-provider requires --model" in captured.err
    assert "Traceback" not in captured.err


def test_forced_backend_failure_leaves_metadata_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    session_id = first.loop.store.session_id
    metadata_path = home / "sessions" / session_id / "meta.json"
    before = metadata_path.read_text()

    def fail_backend(*args: object, **kwargs: object) -> object:
        raise RuntimeError("backend construction failed")

    monkeypatch.setattr("zeta.tui.app.build_backend", fail_backend)
    with pytest.raises(RuntimeError, match="backend construction failed"):
        create_app(
            build_parser().parse_args(
                [
                    "--resume",
                    session_id,
                    "--provider",
                    "claude",
                    "--model",
                    "claude-sonnet-4-6",
                    "--force-provider",
                ]
            )
        )

    assert metadata_path.read_text() == before


@pytest.mark.asyncio
async def test_resumed_pending_approval_is_presented_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    opened = SessionManager(home).create(provider="fake", model="offline", cwd=tmp_path)
    call = ToolCall("approval-resume", "exec", {"command": "danger"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    app = create_app(
        build_parser().parse_args(
            ["--resume", opened.store.session_id, "--provider", "fake"]
        )
    )
    app.console = Console(file=StringIO(), force_terminal=False)
    app._present_pending_approvals()

    assert call.id in app.console.file.getvalue()
    assert await app._handle_approval_input(f"approve {call.id}")
    assert app.loop.store.pending_approvals() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny"])
async def test_resume_pending_tool_executes_and_persists_result(
    tmp_path: Path, decision: str
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    executed: list[str] = []

    async def echo(arguments: dict[str, str]) -> str:
        executed.append(arguments["value"])
        return arguments["value"]

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"echo": echo},
        approval_policy=policy,
    )
    call = ToolCall("approval-tool", "echo", {"value": "done"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    if decision == "allow":
        assert policy.approve(call.id)
    else:
        assert policy.deny(call.id)

    result = await loop.resume_pending_tool(call.id)

    assert result is not None
    assert opened.store.messages()[-1].tool_result == result
    assert executed == (["done"] if decision == "allow" else [])


def test_metadata_override_and_touch_are_serialized(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    session_id = opened.store.session_id
    override_manager = SessionManager(home)
    touch_manager = SessionManager(home)
    override_metadata = override_manager.open(session_id).metadata
    touch_metadata = touch_manager.open(session_id).metadata
    barrier = threading.Barrier(2)

    def override() -> None:
        barrier.wait()
        override_manager.record_override(
            override_metadata,
            provider="claude",
            model="claude-sonnet-4-6",
        )

    def touch() -> None:
        barrier.wait()
        touch_manager.touch(touch_metadata)

    first = threading.Thread(target=override)
    second = threading.Thread(target=touch)
    first.start()
    second.start()
    first.join()
    second.join()

    current = manager.open(session_id).metadata
    assert current.provider == "claude"
    assert current.model == "claude-sonnet-4-6"
    assert len(current.override_audit) == 1


def test_session_id_collision_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    collision = uuid.UUID("00000000000000000000000000000001")
    unique = uuid.UUID("00000000000000000000000000000002")
    calls = iter(
        [
            collision,
            uuid.UUID("00000000000000000000000000000003"),
            collision,
            unique,
            uuid.UUID("00000000000000000000000000000004"),
        ]
    )
    monkeypatch.setattr("zeta.session.uuid.uuid4", lambda: next(calls))
    manager.create(provider="fake", model="offline", cwd=tmp_path)
    created = manager.create(provider="fake", model="offline", cwd=tmp_path)

    assert created.store.session_id == unique.hex


def test_session_cwd_is_normalized_for_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=Path("."))

    assert manager.find_most_recent(cwd=Path.cwd()).session_id == opened.store.session_id


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
