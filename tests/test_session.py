from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console

from zeta.core.abort import AbortGenerationRegistry
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionError, SessionManager
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.tui.app import TUIApp, create_app
from zeta.tools.agent import ChildApprovalPolicy
from zeta.cli import build_parser, main
from zeta.tui.layout import CONTENT_MARGIN, content_width
from zeta.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def _args(*values: str):
    return build_parser().parse_args([*values, "--provider", "fake"])


async def wait_until(check: Callable[[], bool]) -> None:
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


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


def test_concurrent_legacy_resumes_adopt_the_persisted_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    opened = SessionManager(home).create(provider="fake", model="offline", cwd=tmp_path)
    session_id = opened.store.session_id
    metadata_path = home / "sessions" / session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("system_prompt")
    metadata.pop("context_files")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    barrier = threading.Barrier(2)
    load_count = 0
    load_count_lock = threading.Lock()

    def load_context(*, repo_root: Path, zeta_home: Path) -> ProjectContext:
        del repo_root, zeta_home
        nonlocal load_count
        with load_count_lock:
            index = load_count
            load_count += 1
        barrier.wait()
        return ProjectContext(f"fallback {index}", (tmp_path / f"context-{index}.md",))

    monkeypatch.setattr("zeta.tui.app.load_project_context", load_context)
    apps: list[TUIApp] = []
    errors: list[Exception] = []

    def resume() -> None:
        try:
            apps.append(
                create_app(
                    build_parser().parse_args(
                        ["--resume", session_id, "--provider", "fake"]
                    )
                )
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=resume)
    second = threading.Thread(target=resume)
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert load_count == 2
    saved = SessionManager(home).open(session_id).metadata
    prompts = [app.loop.context_assembler.system_prompt.content[0].text for app in apps]
    context_files = [app.slash_status().context_files for app in apps]
    assert saved.system_prompt in {"fallback 0", "fallback 1"}
    assert prompts == [saved.system_prompt, saved.system_prompt]
    assert context_files == [tuple(saved.context_files), tuple(saved.context_files)]


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


def test_model_swap_persists_and_restores_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())

    output = create_slash_registry().dispatch(first, "/model faster")
    assert output == "model: faster"
    assert first.model == "faster"
    assert SessionManager(home).open(first.loop.store.session_id).metadata.model == "faster"

    resumed = create_app(
        build_parser().parse_args(
            ["--resume", first.loop.store.session_id, "--provider", "fake"]
        )
    )
    assert resumed.model == "faster"


def test_vim_mode_defaults_on_and_persists_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())

    assert first.vim_mode is True
    assert create_slash_registry().dispatch(first, "/vim off") == "vim mode: off"
    assert first.vim_mode is False
    assert (
        SessionManager(home).open(first.loop.store.session_id).metadata.vim_mode
        is False
    )

    resumed = create_app(
        build_parser().parse_args(
            ["--resume", first.loop.store.session_id, "--provider", "fake"]
        )
    )
    assert resumed.vim_mode is False


def test_legacy_session_metadata_defaults_vim_mode_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    first = create_app(_args())
    metadata_path = home / "sessions" / first.loop.store.session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("vim_mode")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = create_app(
        build_parser().parse_args(
            ["--resume", first.loop.store.session_id, "--provider", "fake"]
        )
    )
    assert resumed.vim_mode is True


@pytest.mark.asyncio
async def test_unknown_model_swap_warns_and_changes_the_model(tmp_path: Path) -> None:
    backend = FakeBackend([])
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=lambda provider: frozenset({"claude-sonnet-4-6"}),
    )

    output = create_slash_registry().dispatch(app, "/model offline")
    await wait_until(lambda: app._model_catalog_loaded)

    assert output == "model: offline (model catalog unavailable for claude — using anyway)"
    assert app.model == "offline"


@pytest.mark.asyncio
async def test_unknown_model_with_provider_prefix_warns_before_state_change(
    tmp_path: Path,
) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([]), opened.store),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=lambda provider: frozenset({"claude-sonnet-4-6"}),
    )

    output = create_slash_registry().dispatch(
        app, "/model claude-definitely-not-real"
    )
    await wait_until(lambda: app._model_catalog_loaded)

    assert output == (
        "model: claude-definitely-not-real "
        "(model catalog unavailable for claude — using anyway)"
    )
    assert app.model == "claude-definitely-not-real"
    assert manager.open(opened.store.session_id).metadata.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_model_catalog_hit_has_no_warning(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=lambda provider: frozenset({"claude-opus-4-7"}),
    )

    first = create_slash_registry().dispatch(app, "/model claude-opus-4-7")
    await wait_until(lambda: app._model_catalog_loaded)
    output = create_slash_registry().dispatch(app, "/model claude-opus-4-7")

    assert first == (
        "model: claude-opus-4-7 "
        "(model catalog unavailable for claude — using anyway)"
    )
    assert output == "model: claude-opus-4-7"


@pytest.mark.asyncio
async def test_model_catalog_miss_warns_and_is_cached(tmp_path: Path) -> None:
    calls: list[str] = []

    def load_catalog(provider: str) -> frozenset[str]:
        calls.append(provider)
        return frozenset({"claude-opus-4-7"})

    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=load_catalog,
    )

    first = create_slash_registry().dispatch(app, "/model claude-new")
    await wait_until(lambda: app._model_catalog_loaded)
    second = create_slash_registry().dispatch(app, "/model claude-other")

    assert first == (
        "model: claude-new "
        "(model catalog unavailable for claude — using anyway)"
    )
    assert second == (
        "model: claude-other (model not found in claude catalog — using anyway)"
    )
    assert calls == ["claude"]


@pytest.mark.asyncio
async def test_model_catalog_load_does_not_block_input(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def load_catalog(provider: str) -> frozenset[str]:
        del provider
        started.set()
        release.wait(timeout=1)
        return frozenset({"claude-opus-4-7"})

    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=load_catalog,
    )

    began = time.monotonic()
    first = create_slash_registry().dispatch(app, "/model claude-opus-4-7")
    elapsed = time.monotonic() - began

    try:
        assert elapsed < 0.25
        assert first == (
            "model: claude-opus-4-7 "
            "(model catalog unavailable for claude — using anyway)"
        )
        assert await asyncio.to_thread(started.wait, 1)
    finally:
        release.set()

    await wait_until(lambda: app._model_catalog_loaded)
    assert create_slash_registry().dispatch(app, "/model claude-opus-4-7") == (
        "model: claude-opus-4-7"
    )


@pytest.mark.asyncio
async def test_unavailable_model_catalog_warns(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="codex",
        model="gpt-5.4",
        model_catalog_loader=lambda provider: None,
    )

    output = create_slash_registry().dispatch(app, "/model gpt-5.6-sol")
    await wait_until(lambda: app._model_catalog_loaded)

    assert output == "model: gpt-5.6-sol (model catalog unavailable for codex — using anyway)"


def test_wrong_provider_model_prefix_is_rejected(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="claude",
        model="claude-sonnet-4-6",
        model_catalog_loader=lambda provider: frozenset(),
    )

    output = create_slash_registry().dispatch(app, "/model gpt-5.6-sol")

    assert output == "model unchanged: model 'gpt-5.6-sol' has a wrong-provider prefix for claude"


@pytest.mark.asyncio
async def test_model_swap_is_rejected_during_active_turn(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_task = asyncio.create_task(asyncio.sleep(1))

    try:
        output = create_slash_registry().dispatch(app, "/model faster")
    finally:
        app._active_task.cancel()
        await asyncio.gather(app._active_task, return_exceptions=True)

    assert output == "model unchanged: cannot change model while a turn or approval is active"
    assert app.model == "offline"


def test_model_swap_is_rejected_with_pending_approval(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(store=store)
    call = ToolCall("pending-model-swap", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy),
        provider="fake",
        model="offline",
        approval_policy=policy,
    )

    output = create_slash_registry().dispatch(app, "/model faster")

    assert output == "model unchanged: cannot change model while a turn or approval is active"
    assert app.model == "offline"


@pytest.mark.asyncio
async def test_compact_command_forces_the_existing_compaction_path(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn(content=[TextContent("summary")])])
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("first")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("answer")]))
    assembler = ContextAssembler(
        store,
        backend=backend,
        token_budget=1000,
        retained_tail=1,
        system_prompt="stable identity",
        token_counter=lambda message: 10,
    )
    app = TUIApp(
        AgentLoop(backend, store, context_assembler=assembler),
        provider="fake",
        model="offline",
    )

    output = await create_slash_registry().dispatch_async(app, "/compact")

    assert output is not None
    assert output.startswith("compacted entries ")
    assert store.compaction_marker_count() == 1
    assert backend.calls[0][0][0].content[0].text == "stable identity"


def test_session_previews_are_ordered_and_ansi_safe(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    older = manager.create(provider="fake", model="offline", cwd=tmp_path)
    newer = manager.create(provider="fake", model="offline", cwd=tmp_path)
    older.store.append_message(
        Message(MessageRole.USER, [TextContent("older message")])
    )
    newer.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("\x1b[31mnew\x1b[0m\nmessage with controls")],
        )
    )
    older.metadata.updated_at = "2020-01-01T00:00:00+00:00"
    newer.metadata.updated_at = "2030-01-01T00:00:00+00:00"
    manager._write(older.metadata)
    manager._write(newer.metadata)

    previews = manager.list_session_previews()

    assert [item.session_id for item in previews] == [
        newer.store.session_id,
        older.store.session_id,
    ]
    assert previews[0].preview == "new message with controls"
    assert "\x1b" not in previews[0].preview


def test_session_preview_strips_c1_controls_and_truncates_by_cell_width(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("wide \u009b31m" + "界" * 40 + "\u009b0m tail")],
        )
    )

    preview = manager.list_session_previews()[0].preview

    assert "\u009b" not in preview
    assert cell_len(preview) <= 80
    assert preview.endswith("...")


def test_session_preview_strips_zero_width_and_bidi_controls_before_capping(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [
                TextContent(
                    "start" + "\u0301" * 100_000 + "\u200d" * 100_000
                    + "\u202e end"
                )
            ],
        )
    )

    preview = manager.list_session_previews()[0].preview

    assert "\u200d" not in preview
    assert "\u202e" not in preview
    assert len(preview) <= 512


def test_session_preview_keeps_combining_and_emoji_text_safe(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("cafe\u0301 family 👩\u200d💻 \u2066safe\u2069")],
        )
    )

    preview = manager.list_session_previews()[0].preview

    assert "cafe\u0301" in preview
    assert "👩💻" in preview
    assert "\u2066" not in preview
    assert "\u2069" not in preview


def test_session_preview_picker_limits_recent_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    sessions = [create_app(_args()).loop.store.session_id for _ in range(21)]
    manager = SessionManager(home)
    for index, session_id in enumerate(sessions):
        metadata = manager.open(session_id).metadata
        metadata.updated_at = f"2030-01-01T00:00:{index:02d}+00:00"
        manager._write(metadata)

    previews = manager.list_session_previews()

    assert len(previews) == 20
    assert previews[0].session_id == sessions[-1]


def test_resume_picker_rejects_zero_and_negative_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    create_app(_args())

    for choice in ("0", "-1"):
        monkeypatch.setattr("builtins.input", lambda prompt, choice=choice: choice)
        with pytest.raises(SessionError, match="invalid resume session selection"):
            create_app(build_parser().parse_args(["--resume", "--provider", "fake"]))


def test_resume_picker_matches_direct_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(_args())
    second = create_app(_args())
    first.loop.store.append_message(
        Message(MessageRole.USER, [TextContent("first session")])
    )
    second.loop.store.append_message(
        Message(MessageRole.USER, [TextContent("second session")])
    )
    manager = SessionManager(home)
    first_metadata = manager.open(first.loop.store.session_id).metadata
    second_metadata = manager.open(second.loop.store.session_id).metadata
    first_metadata.updated_at = "2020-01-01T00:00:00+00:00"
    second_metadata.updated_at = "2030-01-01T00:00:00+00:00"
    manager._write(first_metadata)
    manager._write(second_metadata)
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    picked = create_app(
        build_parser().parse_args(["--resume", "--provider", "fake"])
    )
    direct = create_app(
        build_parser().parse_args(
            ["--resume", first.loop.store.session_id, "--provider", "fake"]
        )
    )

    assert picked.loop.store.session_id == direct.loop.store.session_id


@pytest.mark.parametrize("terminal_width", [200, 120, 80, 40])
def test_resume_picker_stays_within_shared_content_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminal_width: int,
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    session = create_app(_args())
    session.loop.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("a very long session preview " * 12)],
        )
    )
    monkeypatch.setattr(
        "zeta.tui.app.get_terminal_size",
        lambda fallback: SimpleNamespace(columns=terminal_width, lines=24),
    )
    prompts: list[str] = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or "1",
    )

    create_app(build_parser().parse_args(["--resume", "--provider", "fake"]))

    output = capsys.readouterr().out.splitlines()
    assert output
    assert len(prompts) == 1
    assert output[0].startswith(" " * CONTENT_MARGIN)
    assert prompts[0].startswith(" " * CONTENT_MARGIN)
    assert prompts[0].endswith(" ")
    assert all(cell_len(line) <= terminal_width for line in output)
    assert cell_len(prompts[0]) <= terminal_width
    assert all(
        cell_len(line[CONTENT_MARGIN:]) <= content_width(terminal_width)
        for line in output
    )
    assert cell_len(prompts[0][CONTENT_MARGIN:]) <= content_width(terminal_width)


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


def test_resume_honors_provider_from_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(_args())
    session_id = first.loop.store.session_id

    project = tmp_path / "project"
    dot_zeta = project / ".zeta"
    dot_zeta.mkdir(parents=True)
    (dot_zeta / "settings.toml").write_text(
        'provider = "claude"\n', encoding="utf-8"
    )
    monkeypatch.chdir(project)
    # No CLI --provider; settings.provider alone must trigger the mismatch banner.
    with pytest.raises(SessionError, match="override rejected"):
        create_app(build_parser().parse_args(["--resume", session_id]))

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
    monkeypatch.setattr("zeta.tui.app._load_model_catalog", lambda provider: None)
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
    assert create_slash_registry().dispatch(app, "/model claude-opus-4-1") == (
        "model: claude-opus-4-1 "
        "(model catalog unavailable for claude — using anyway; "
        "context budget 1,000,000 -> 200,000)"
    )
    assert json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )["model"] == "offline"
    app._invalidate_prompt = lambda: None
    await app._consume_turn("hello")

    metadata = json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )
    assert metadata["provider"] == "claude"
    assert metadata["model"] == "claude-opus-4-1"
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

    assert call.name in app.console.file.getvalue()  # card shows the tool name
    assert await app._handle_approval_input(f"approve {call.id}")
    assert app.loop.store.pending_approvals() == []


@pytest.mark.asyncio
async def test_tui_resolves_colliding_child_approvals_by_unique_key(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    loop = AgentLoop(FakeBackend([]), parent_store, approval_policy=policy)
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    app.console = Console(file=StringIO(), force_terminal=False)
    child_a = ConversationStore(tmp_path / "children", session_id="a")
    child_b = ConversationStore(tmp_path / "children", session_id="b")
    policy_a = ChildApprovalPolicy(policy, child_a, "child a", "child-a")
    policy_b = ChildApprovalPolicy(policy, child_b, "child b", "child-b")
    call_a = ToolCall("same-request", "bash", {"cmd": "a"})
    call_b = ToolCall("same-request", "bash", {"cmd": "b"})
    signal_a = AbortGenerationRegistry().new_generation()
    signal_b = AbortGenerationRegistry().new_generation()
    task_a = asyncio.create_task(policy_a.authorize(call_a, signal_a))
    task_b = asyncio.create_task(policy_b.authorize(call_b, signal_b))

    await wait_until(lambda: len(app.pending_approvals) == 2)
    app._present_pending_approvals()
    rendered = app.console.file.getvalue()
    assert "child a: bash" in rendered
    assert "child b: bash" in rendered
    # y/n only answer the first card, so the second one names its own key.
    assert "y approve · n deny" in rendered
    assert "approve ('child-b', 'same-request')" in rendered

    assert await app._handle_approval_input(
        "approve ('child-a', 'same-request')"
    )
    assert await app._handle_approval_input("deny ('child-b', 'same-request')")
    assert await task_a == ApprovalDecision.ALLOW
    assert await task_b == ApprovalDecision.DENY
    signal_a.abort()
    signal_b.abort()


@pytest.mark.asyncio
async def test_cancel_then_approve_does_not_resume_tool(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    executed: list[str] = []

    async def echo(arguments: dict[str, str]) -> str:
        executed.append(arguments["value"])
        return arguments["value"]

    call = ToolCall("approval-cancel-then-approve", "echo", {"value": "no"})
    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn(tool_calls=[call])]),
            opened.store,
            tools={"echo": echo},
            approval_policy=policy,
            max_turns=1,
        ),
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    turn = asyncio.create_task(app._consume_turn("start"))
    app._active_task = turn
    await wait_until(lambda: bool(app.pending_approvals))

    app.abort_active()
    await asyncio.gather(turn, return_exceptions=True)

    assert app.pending_approvals == ()
    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"
    assert await app._handle_approval_input(f"approve {call.id}")
    assert executed == []
    assert len(
        [message for message in opened.store.messages() if message.tool_result is not None]
    ) == 1


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

    events = []
    result = await loop.resume_pending_tool(call.id, event_sink=events.append)

    assert result is not None
    assert opened.store.messages()[-1].tool_result == result
    assert executed == (["done"] if decision == "allow" else [])
    assert [event.type for event in events] == (
        [StreamEventType.TOOL_EXECUTION_START, StreamEventType.TOOL_EXECUTION_END]
        if decision == "allow"
        else [StreamEventType.TOOL_EXECUTION_END]
    )


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
    events: list[StreamEvent] = []
    task = asyncio.create_task(loop.resume_pending_tool(call.id, event_sink=events.append))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"
    assert [event.type for event in events] == [
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_END,
    ]


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
async def test_resume_pending_tool_rejects_existing_result(tmp_path: Path) -> None:
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
    call = ToolCall("approval-existing-result", "echo", {"value": "done"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    assert policy.approve(call.id)
    success = ToolResult(call.id, "already completed")
    opened.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(success.content)],
            tool_result=success,
        )
    )

    events: list[StreamEvent] = []
    result = await loop.resume_pending_tool(call.id, event_sink=events.append)

    assert result == success
    assert executed == []
    assert events == []


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


def test_new_session_stores_the_model_derived_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setattr("zeta.tui.app._load_model_catalog", lambda provider: None)
    app = create_app(
        build_parser().parse_args(
            ["--provider", "claude", "--model", "claude-sonnet-4-6"]
        )
    )
    session_id = app.loop.store.session_id
    metadata = json.loads((home / "sessions" / session_id / "meta.json").read_text())
    assert metadata["compaction_budget"] == 1_000_000
    assert metadata["budget_pinned"] is False
    assert app.loop.context_assembler.token_budget == 1_000_000


def test_unknown_model_falls_back_to_the_default_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(_args())
    assert app.loop.context_assembler.token_budget == 200_000


def test_token_budget_override_pins_and_survives_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setattr("zeta.tui.app._load_model_catalog", lambda provider: None)
    first = create_app(
        build_parser().parse_args(
            [
                "--provider",
                "claude",
                "--model",
                "claude-sonnet-4-6",
                "--token-budget",
                "12345",
            ]
        )
    )
    session_id = first.loop.store.session_id
    assert first.loop.context_assembler.token_budget == 12345
    metadata = json.loads((home / "sessions" / session_id / "meta.json").read_text())
    assert metadata["compaction_budget"] == 12345
    assert metadata["budget_pinned"] is True

    resumed = create_app(build_parser().parse_args(["--resume", session_id]))
    assert resumed.loop.context_assembler.token_budget == 12345
    assert resumed._on_budget_change is None


def test_resume_retunes_an_unpinned_budget_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setattr("zeta.tui.app._load_model_catalog", lambda provider: None)
    session_id = create_app(
        build_parser().parse_args(
            ["--provider", "claude", "--model", "claude-opus-4-5"]
        )
    ).loop.store.session_id
    assert json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )["compaction_budget"] == 200_000

    resumed = create_app(
        build_parser().parse_args(
            [
                "--resume",
                session_id,
                "--model",
                "claude-sonnet-4-6",
                "--force-provider",
            ]
        )
    )
    assert resumed.loop.context_assembler.token_budget == 1_000_000
    assert json.loads(
        (home / "sessions" / session_id / "meta.json").read_text()
    )["compaction_budget"] == 1_000_000


def test_model_command_leaves_a_pinned_budget_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setattr("zeta.tui.app._load_model_catalog", lambda provider: None)
    app = create_app(
        build_parser().parse_args(
            [
                "--provider",
                "claude",
                "--model",
                "claude-opus-4-5",
                "--token-budget",
                "9000",
            ]
        )
    )
    assert create_slash_registry().dispatch(app, "/model claude-sonnet-4-6") == (
        "model: claude-sonnet-4-6 "
        "(model catalog unavailable for claude — using anyway)"
    )
    assert app.loop.context_assembler.token_budget == 9000
