from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.session import SessionError, SessionManager
from zeta.tui.app import TUIApp, create_app
from zeta.cli import build_parser, main
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


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


def test_fresh_session_injects_context_and_lists_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    (tmp_path / "AGENTS.md").write_text("repo rules", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    system_prompt = app.loop.context_assembler.system_prompt.content[0].text

    assert "You are zeta" in system_prompt
    assert "repo rules" in system_prompt
    assert str((tmp_path / "AGENTS.md").resolve()) in app.slash_status().context_files


def test_resume_restores_context_snapshot_across_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    original = tmp_path / "original"
    other = tmp_path / "other"
    original.mkdir()
    other.mkdir()
    original_context = original / "AGENTS.md"
    original_context.write_text("original rules", encoding="utf-8")
    (other / "AGENTS.md").write_text("new directory rules", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(original)

    first = create_app(_args())
    session_id = first.loop.store.session_id
    original_context.write_text("changed rules", encoding="utf-8")
    monkeypatch.chdir(other)

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )
    prompt = resumed.loop.context_assembler.system_prompt.content[0].text

    assert "original rules" in prompt
    assert "changed rules" not in prompt
    assert "new directory rules" not in prompt
    assert resumed.slash_status().context_files == (str(original_context.resolve()),)


def test_legacy_resume_persists_context_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("legacy rules", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    opened = SessionManager(home).create(provider="fake", model="offline", cwd=tmp_path)
    metadata_path = home / "sessions" / opened.store.session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("system_prompt")
    metadata.pop("context_files")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = create_app(
        build_parser().parse_args(["--resume", opened.store.session_id, "--provider", "fake"])
    )
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved["system_prompt"] == resumed.loop.context_assembler.system_prompt.content[0].text
    assert saved["context_files"] == [str(context_file.resolve())]


def test_legacy_resume_uses_persisted_snapshot_on_second_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("first rules", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    opened = SessionManager(home).create(provider="fake", model="offline", cwd=tmp_path)
    metadata_path = home / "sessions" / opened.store.session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("system_prompt")
    metadata.pop("context_files")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    create_app(
        build_parser().parse_args(["--resume", opened.store.session_id, "--provider", "fake"])
    )
    context_file.write_text("second rules", encoding="utf-8")

    resumed = create_app(
        build_parser().parse_args(["--resume", opened.store.session_id, "--provider", "fake"])
    )
    prompt = resumed.loop.context_assembler.system_prompt.content[0].text

    assert "first rules" in prompt
    assert "second rules" not in prompt


def test_partial_context_metadata_is_replaced_with_fallback_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("first rules", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(_args())
    session_id = first.loop.store.session_id
    metadata_path = home / "sessions" / session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("context_files")
    context_file.write_text("replacement rules", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompt = resumed.loop.context_assembler.system_prompt.content[0].text

    assert "replacement rules" in prompt
    assert saved["context_files"] == [str(context_file.resolve())]


def test_session_bash_cwd_round_trips_through_store_state(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)

    opened.store.set_bash_cwd("/tmp")
    resumed = manager.open(opened.store.session_id)

    assert resumed.store.bash_cwd == "/tmp"
    assert json.loads(resumed.store.state_path.read_text(encoding="utf-8")) == {
        "bash_cwd": "/tmp"
    }


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
    metadata_path = home / "sessions" / session_id / "meta.json"
    before_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
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

    metadata = json.loads(metadata_path.read_text())
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == before_digest
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


@pytest.mark.asyncio
async def test_resumed_tool_abort_active_persists_canceled_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    started = asyncio.Event()

    async def block(arguments: dict[str, str], abort_signal: object) -> str:
        del arguments
        started.set()
        await abort_signal.wait()  # type: ignore[attr-defined]
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": block},
        approval_policy=policy,
    )
    call = ToolCall("approval-abort", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    approval_task = asyncio.create_task(app._handle_approval_input(f"approve {call.id}"))
    await asyncio.wait_for(started.wait(), timeout=1)
    app.abort_active()
    assert await approval_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"


@pytest.mark.asyncio
async def test_resumed_tool_direct_cancel_persists_canceled_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    started = asyncio.Event()

    async def block(arguments: dict[str, str]) -> str:
        del arguments
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": block},
        approval_policy=policy,
    )
    call = ToolCall("approval-cancel", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy.approve(call.id)
    task = asyncio.create_task(loop.resume_pending_tool(call.id))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"


@pytest.mark.asyncio
async def test_resumed_tool_immediate_abort_persists_canceled_result(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)

    async def never_runs(arguments: dict[str, str]) -> str:
        del arguments
        raise AssertionError("the handler must not run")

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"never": never_runs},
        approval_policy=policy,
    )
    call = ToolCall("approval-immediate-abort", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy.approve(call.id)

    assert loop.prepare_resume_pending_tool(call.id)
    loop.abort()
    result = await loop.resume_pending_tool(call.id, prepared=True)

    assert result is not None
    assert result.content == "tool execution canceled"
    assert opened.store.messages()[-1].tool_result == result


def test_finalize_canceled_is_idempotent(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    loop = AgentLoop(FakeBackend([]), opened.store)
    call = ToolCall("approval-idempotent-cancel", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    first = loop.finalize_canceled(call.id)
    second = loop.finalize_canceled(call.id)

    results = [
        message.tool_result
        for message in opened.store.messages()
        if message.tool_result is not None
    ]
    assert first is not None
    assert second == first
    assert results == [first]


def test_completion_edge_idempotence_preserves_success(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    loop = AgentLoop(FakeBackend([]), opened.store)
    call = ToolCall("approval-completion-edge", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    success = ToolResult(call.id, "completed")
    opened.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(success.content)],
            tool_result=success,
        )
    )

    result = loop.finalize_canceled(call.id)
    results = [
        message.tool_result
        for message in opened.store.messages()
        if message.tool_result is not None
    ]

    assert result == success
    assert results == [success]


@pytest.mark.asyncio
async def test_strict_pre_start_parent_cancellation_persists_canceled_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)

    async def never_runs(arguments: dict[str, str]) -> str:
        del arguments
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"never": never_runs},
        approval_policy=policy,
    )
    call = ToolCall("approval-parent-cancel", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    original_prepare = loop.prepare_resume_pending_tool
    parent_task: asyncio.Task[bool]

    def cancel_create(coro: object) -> asyncio.Task[object]:
        close = getattr(coro, "close")
        close()
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        raise asyncio.CancelledError

    def prepare(request_id: str) -> bool:
        prepared = original_prepare(request_id)
        monkeypatch.setattr(asyncio, "create_task", cancel_create)
        return prepared

    monkeypatch.setattr(loop, "prepare_resume_pending_tool", prepare)
    parent_task = asyncio.create_task(
        app._handle_approval_input(f"approve {call.id}")
    )

    with pytest.raises(asyncio.CancelledError):
        await parent_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == (
        "tool execution canceled"
    )


@pytest.mark.asyncio
async def test_parent_cancellation_after_child_start_persists_result(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    handler_started = asyncio.Event()

    async def blocks(arguments: dict[str, str]) -> str:
        del arguments
        handler_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": blocks},
        approval_policy=policy,
    )
    call = ToolCall("approval-parent-after-start", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    parent_task = asyncio.create_task(
        app._handle_approval_input(f"approve {call.id}")
    )
    await handler_started.wait()
    parent_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await parent_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == (
        "tool execution canceled"
    )


@pytest.mark.asyncio
async def test_summary_success_commits_override_before_main_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(
        provider="fake",
        model="offline",
        cwd=tmp_path,
        retained_tail=35,
        compaction_budget=600,
    )
    for index in range(50):
        opened.store.append_message(
            Message(MessageRole.USER, [TextContent(f"message {index}")])
        )
    session_id = opened.store.session_id
    backend = FakeBackend([ScriptedTurn(content=[TextContent("summary")])])

    def build_backend(*args: object, **kwargs: object) -> tuple[FakeBackend, str]:
        del args, kwargs
        return backend, "claude-sonnet-4-6"

    monkeypatch.setenv("ZETA_HOME", str(home))
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
    await app._consume_turn("new message")

    metadata = manager.open(session_id).metadata
    assert metadata.provider == "claude"
    assert metadata.model == "claude-sonnet-4-6"
    assert metadata.override_audit


def test_concurrent_overrides_are_first_writer_wins(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    session_id = opened.store.session_id
    managers = [SessionManager(manager.home), SessionManager(manager.home)]
    metadata = [item.open(session_id).metadata for item in managers]
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def override(index: int) -> None:
        try:
            barrier.wait()
            managers[index].record_override(
                metadata[index],
                provider=f"provider-{index}",
                model=f"model-{index}",
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=override, args=(0,))
    second = threading.Thread(target=override, args=(1,))
    first.start()
    second.start()
    first.join()
    second.join()

    current = manager.open(session_id).metadata
    assert len(errors) == 1
    assert isinstance(errors[0], SessionError)
    assert len(current.override_audit) == 1
    assert current.provider in {"provider-0", "provider-1"}


def test_sequential_overrides_use_latest_snapshot(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)

    manager.record_override(
        opened.metadata,
        provider="claude",
        model="claude-sonnet-4-6",
    )
    latest = SessionManager(home).open(opened.store.session_id).metadata
    manager.record_override(
        latest,
        provider="codex",
        model="gpt-5.4",
    )

    current = manager.open(opened.store.session_id).metadata
    assert current.provider == "codex"
    assert current.model == "gpt-5.4"
    assert [item["provider"] for item in current.override_audit] == [
        {"from": "fake", "to": "claude"},
        {"from": "claude", "to": "codex"},
    ]


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
    monkeypatch.setattr("zeta.core.session.uuid.uuid4", lambda: next(calls))
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
