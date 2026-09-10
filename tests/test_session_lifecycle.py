"""Tests for ZETA-79 session lifecycle ergonomics."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.cli import _print_exit_hint, build_parser, main
from zeta.core.session import (
    SESSION_NAME_MAX_LENGTH,
    SessionError,
    SessionManager,
    SessionPreview,
    format_relative_age,
    normalize_session_name,
)
from zeta.core.slash import create_slash_registry
from zeta.tui.app import create_app, format_picker_row


def _args(*values: str) -> argparse.Namespace:
    return build_parser().parse_args([*values, "--provider", "fake"])


def test_normalize_session_name_accepts_and_rejects() -> None:
    assert normalize_session_name("  hello world  ") == "hello world"
    with pytest.raises(SessionError):
        normalize_session_name("")
    with pytest.raises(SessionError):
        normalize_session_name("\x1b[31m\x1b[0m")
    with pytest.raises(SessionError):
        normalize_session_name("x" * (SESSION_NAME_MAX_LENGTH + 1))


def test_format_relative_age_covers_ranges() -> None:
    now = datetime(2030, 6, 15, 12, 0, 0, tzinfo=UTC)
    def at(delta: timedelta) -> str:
        return format_relative_age((now - delta).isoformat(), now=now)

    assert at(timedelta(seconds=1)) == "just now"
    assert at(timedelta(seconds=30)) == "30s ago"
    assert at(timedelta(minutes=5)) == "5m ago"
    assert at(timedelta(hours=3)) == "3h ago"
    assert at(timedelta(days=2)) == "2d ago"
    assert at(timedelta(days=45)) == "1mo ago"
    assert at(timedelta(days=400)) == "1y ago"
    assert format_relative_age("not-a-date", now=now) == "unknown"


def test_session_metadata_persists_name_field(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)

    manager.record_name(opened.metadata, name="planning")
    reopened = SessionManager(home).open(opened.store.session_id)
    assert reopened.metadata.name == "planning"


def test_slash_name_persists_label_and_shows_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(_args())

    assert create_slash_registry().dispatch(app, "/name") == "session name: (unnamed)"
    assert (
        create_slash_registry().dispatch(app, "/name  planning  ")
        == "session name: planning"
    )
    assert SessionManager(home).open(app.loop.store.session_id).metadata.name == (
        "planning"
    )
    assert create_slash_registry().dispatch(app, "/name") == "session name: planning"


def test_slash_name_rejects_empty_or_control_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(_args())

    output = create_slash_registry().dispatch(app, "/name \x1b[31m\x1b[0m")
    assert output.startswith("session name unchanged")


def test_slash_new_requests_restart_only_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(_args())

    assert create_slash_registry().dispatch(app, "/new arg") == (
        "new unchanged: /new does not accept arguments"
    )
    assert app.new_session_requested is False

    result = create_slash_registry().dispatch(app, "/new")
    assert result == "starting a fresh session..."
    assert app.new_session_requested is True
    assert app._exit_requested is True


def test_slash_new_rejected_during_active_turn(tmp_path: Path) -> None:
    async def rejected() -> None:
        from zeta.core.fake import FakeBackend
        from zeta.core.loop import AgentLoop
        from zeta.core.store import ConversationStore
        from zeta.tui.app import TUIApp

        app = TUIApp(
            AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
            provider="fake",
            model="offline",
        )
        app._active_task = asyncio.create_task(asyncio.sleep(1))
        try:
            output = create_slash_registry().dispatch(app, "/new")
        finally:
            app._active_task.cancel()
            await asyncio.gather(app._active_task, return_exceptions=True)
        assert output == (
            "new unchanged: cannot start a new session while a turn or "
            "approval is active"
        )
        assert app.new_session_requested is False

    asyncio.run(rejected())


def test_ephemeral_creates_and_cleans_up_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(build_parser().parse_args(["--provider", "fake", "--no-session"]))
    root = app.ephemeral_root
    assert root is not None and root.exists()
    assert not (home / "sessions").exists()
    assert "ephemeral session" in app._startup_warnings[0]

    from zeta.cli import _cleanup_ephemeral

    _cleanup_ephemeral(app)
    assert not root.exists()


def test_ephemeral_headless_mode_leaves_no_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--provider", "fake", "--no-session", "-p", "hi"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert not (home / "sessions").exists()
    assert "ephemeral session" in captured.err


def test_session_list_shows_id_name_age_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    from zeta.types import Message, MessageRole, TextContent

    first = create_app(_args())
    first.loop.store.append_message(
        Message(MessageRole.USER, [TextContent("first prompt")])
    )
    manager = SessionManager(home)
    manager.record_name(
        SessionManager(home).open(first.loop.store.session_id).metadata,
        name="planning",
    )

    exit_code = main(["session", "list"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "planning" in output
    assert first.loop.store.session_id[:8] in output
    assert "first prompt" in output


def test_session_delete_confirms_unless_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    session_id = app.loop.store.session_id

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    exit_code = main(["session", "delete", session_id])
    assert exit_code == 1
    assert (home / "sessions" / session_id).exists()

    asyncio.run(app.loop.close())
    app.loop.store.close()
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    exit_code = main(["session", "delete", session_id])
    assert exit_code == 0
    assert not (home / "sessions" / session_id).exists()

    other = create_app(_args())
    other_id = other.loop.store.session_id
    asyncio.run(other.loop.close())
    other.loop.store.close()
    assert main(["session", "delete", other_id, "--force"]) == 0
    assert not (home / "sessions" / other_id).exists()


def test_session_export_writes_portable_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    from zeta.types import Message, MessageRole, TextContent

    app = create_app(_args())
    session_id = app.loop.store.session_id
    app.loop.store.append_message(
        Message(MessageRole.USER, [TextContent("hi world")])
    )

    exit_code = main(["session", "export", session_id])
    output = capsys.readouterr().out
    assert exit_code == 0
    lines = [json.loads(line) for line in output.splitlines() if line.strip()]
    assert lines[0]["type"] == "session_export"
    assert lines[0]["metadata"]["session_id"] == session_id
    assert len(lines) >= 2

    target = tmp_path / "export.jsonl"
    exit_code = main(["session", "export", session_id, "--out", str(target)])
    assert exit_code == 0
    assert target.exists() and target.read_text().startswith("{")


def test_session_delete_unknown_session_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))

    exit_code = main(["session", "delete", "deadbeef", "--force"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "was not found" in captured.err


def test_picker_row_shows_name_and_age() -> None:
    preview = SessionPreview(
        session_id="abcdef0123456789",
        updated_at=datetime.now(UTC).isoformat(),
        preview="hi there",
        name="planning",
    )
    line = format_picker_row(1, preview)
    assert "abcdef01" in line
    assert "[planning]" in line
    assert "hi there" in line


def test_exit_hint_printed_for_persisted_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    app = create_app(_args())

    _print_exit_hint(app)
    captured = capsys.readouterr()
    assert f"resume with: zeta --resume {app.loop.store.session_id}" in captured.err


def test_exit_hint_suppressed_for_ephemeral_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    app = create_app(build_parser().parse_args(["--provider", "fake", "--no-session"]))

    _print_exit_hint(app)
    captured = capsys.readouterr()
    assert "resume with" not in captured.err


def test_no_session_mutually_exclusive_with_resume() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--no-session", "--resume", "abc"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--no-session", "--continue"])


def test_resume_picker_shows_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(_args())
    manager = SessionManager(home)
    manager.record_name(
        manager.open(first.loop.store.session_id).metadata, name="planning"
    )
    monkeypatch.setattr(
        "zeta.tui.app.get_terminal_size",
        lambda fallback: SimpleNamespace(columns=200, lines=24),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    create_app(build_parser().parse_args(["--resume", "--provider", "fake"]))
    output = capsys.readouterr().out

    assert "[planning]" in output


def test_session_metadata_rejects_bad_name_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    metadata_path = home / "sessions" / opened.store.session_id / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["name"] = 42
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(SessionError, match="context is invalid"):
        SessionManager(home).open(opened.store.session_id)


def test_ephemeral_history_does_not_touch_shared_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    home.mkdir()
    history = home / "history"
    history.write_text("persisted-line\n", encoding="utf-8")
    baseline_bytes = history.read_bytes()
    baseline_mtime_ns = history.stat().st_mtime_ns
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(build_parser().parse_args(["--provider", "fake", "--no-session"]))
    root = app.ephemeral_root
    assert root is not None
    assert app._history_path is not None
    assert Path(app._history_path).parent == root

    from zeta.cli import _cleanup_ephemeral

    _cleanup_ephemeral(app)

    assert history.read_bytes() == baseline_bytes
    assert history.stat().st_mtime_ns == baseline_mtime_ns


def test_session_delete_refuses_when_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import fcntl

    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    session_id = app.loop.store.session_id
    lock_path = home / "sessions" / session_id / ".lock"
    assert lock_path.exists()
    asyncio.run(app.loop.close())
    app.loop.store.close()

    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        exit_code = main(["session", "delete", session_id, "--force"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "currently open" in captured.err
    assert (home / "sessions" / session_id).exists()
    # Releasing only the append lock must make deletion possible.
    assert main(["session", "delete", session_id, "--force"]) == 0
    assert not (home / "sessions" / session_id).exists()


def test_session_delete_removes_corrupt_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    session_id = app.loop.store.session_id
    session_dir = home / "sessions" / session_id
    (session_dir / "conversation.jsonl").unlink()
    asyncio.run(app.loop.close())
    app.loop.store.close()

    exit_code = main(["session", "delete", session_id, "--force"])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert not session_dir.exists()


def test_session_delete_resolves_unique_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(_args())
    session_id = app.loop.store.session_id

    asyncio.run(app.loop.close())
    app.loop.store.close()
    exit_code = main(["session", "delete", session_id[:8], "--force"])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert not (home / "sessions" / session_id).exists()


def test_session_delete_rejects_ambiguous_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "zeta-home"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "abc111").mkdir()
    (sessions_dir / "abc222").mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["session", "delete", "abc", "--force"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ambiguous" in captured.err
    assert (sessions_dir / "abc111").exists()
    assert (sessions_dir / "abc222").exists()


def test_ephemeral_cleanup_when_create_app_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import glob
    import tempfile

    home = tmp_path / "zeta-home"
    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(fake_tmp))
    monkeypatch.chdir(tmp_path)
    tempfile.tempdir = None

    def _boom(*args_, **kwargs_):
        raise RuntimeError("simulated create_app failure")

    monkeypatch.setattr("zeta.tui.app.load_settings", _boom)

    with pytest.raises(RuntimeError, match="simulated create_app failure"):
        create_app(build_parser().parse_args(["--provider", "fake", "--no-session"]))

    leaks = glob.glob(str(fake_tmp / "zeta-ephemeral-*"))
    assert leaks == [], f"leaked ephemeral roots: {leaks}"
