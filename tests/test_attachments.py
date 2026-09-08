from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.tui import composer
from zeta.tui.app import TUIApp
from zeta.tui.composer import (
    ATTACHMENT_MAX_TEXT_BYTES,
    AttachmentError,
    attachment_refs,
    build_key_bindings,
    build_user_message,
)
from zeta.types import ImageContent, MessageRole, TextContent

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


async def run_ctrl_v(app: TUIApp, value: str) -> list[str]:
    submitted: list[str] = []
    session: PromptSession[str] | None = None

    def submit(text: str) -> None:
        submitted.append(text)
        assert session is not None
        session.app.exit()

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
                on_paste=app._paste_from_keybinding,
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        pipe.send_text(value + "\r")
        await task
    return submitted


def webp_data(chunk_type: bytes, chunk_data: bytes) -> bytes:
    chunk = chunk_type + len(chunk_data).to_bytes(4, "little") + chunk_data
    return b"RIFF" + (len(chunk) + 4).to_bytes(4, "little") + b"WEBP" + chunk


WEBP_FIXTURES = (
    webp_data(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00"),
    webp_data(b"VP8L", b"\x2f\x00\x00\x00\x00"),
    webp_data(b"VP8X", b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"),
    webp_data(
        b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00" + b"\x00" * 80
    ),
)


def test_attachment_refs_support_quoted_paths(tmp_path: Path) -> None:
    refs = attachment_refs('read @./notes.txt and @"a b.txt"', tmp_path)

    assert [ref.token for ref in refs] == ["@./notes.txt", '@"a b.txt"']
    assert refs[1].path == (tmp_path / "a b.txt").resolve()


def test_attachment_refs_leave_bare_at_words_as_text(tmp_path: Path) -> None:
    assert attachment_refs("mention @dataclass and @user", tmp_path) == ()
    with pytest.raises(AttachmentError, match="does not exist"):
        build_user_message("read @./missing.txt", tmp_path)


def test_build_user_message_inlines_text_and_preserves_prompt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    message = build_user_message("summarize @./notes.txt", tmp_path)

    assert message.role is MessageRole.USER
    assert message.content[0] == TextContent("summarize @./notes.txt")
    attachment = message.content[1]
    assert isinstance(attachment, TextContent)
    assert attachment.path == str(path.resolve())
    assert attachment.size == 5
    assert attachment.text.endswith("\nhello")


def test_build_user_message_rejects_missing_large_and_binary_files(tmp_path: Path) -> None:
    with pytest.raises(AttachmentError, match="does not exist"):
        build_user_message("@./missing.txt", tmp_path)

    large = tmp_path / "large.txt"
    large.write_bytes(b"a" * (ATTACHMENT_MAX_TEXT_BYTES + 1))
    with pytest.raises(AttachmentError, match="limit"):
        build_user_message("@./large.txt", tmp_path)

    binary = tmp_path / "data.bin"
    binary.write_bytes(b"header\x00payload")
    with pytest.raises(AttachmentError, match="binary"):
        build_user_message("@./data.bin", tmp_path)


def test_empty_text_attachment_persists_and_replays(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    message = build_user_message("read @./empty.txt", tmp_path)
    attachment = message.content[1]
    assert isinstance(attachment, TextContent)
    assert attachment.size == 0

    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.append_message(message)
    reopened = ConversationStore(
        tmp_path / "sessions", session_id=store.session_id, cwd=tmp_path
    )

    assert reopened.messages() == [message]
    assert reopened.replay()


@pytest.mark.parametrize("data", WEBP_FIXTURES)
def test_webp_attachment_variants_are_detected(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "image.data"
    path.write_bytes(data)

    message = build_user_message("inspect @./image.data", tmp_path)

    assert isinstance(message.content[1], ImageContent)
    assert message.content[1].mime_type == "image/webp"


def test_corrupt_webp_attachment_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.data"
    path.write_bytes(webp_data(b"VP8 ", b"\x00" * 9))

    with pytest.raises(AttachmentError, match="binary"):
        build_user_message("inspect @./corrupt.data", tmp_path)


def test_image_attachment_round_trips_and_uses_provider_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "renamed.data"
    path.write_bytes(PNG)
    message = build_user_message("inspect @./renamed.data", tmp_path)
    image = message.content[1]
    assert isinstance(image, ImageContent)
    assert image.mime_type == "image/png"
    assert image.path == str(path.resolve())

    persisted = message.from_dict(message.to_dict())
    assert persisted == message

    anthropic = build_messages_payload(
        [persisted], [], model="claude", max_tokens=4096, thinking_budget=1024
    )
    assert anthropic["messages"][0]["content"][1]["type"] == "image"

    codex = build_responses_payload([persisted], [], model="codex")
    assert codex["input"][0]["content"][1]["type"] == "input_text"
    assert "renamed.data" in codex["input"][0]["content"][1]["text"]


def test_paste_image_queues_a_session_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "/usr/bin/pngpaste")
    real_run = composer.subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] != "/usr/bin/pngpaste":
            return real_run(command, **kwargs)
        Path(command[1]).write_bytes(PNG)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", fake_run)
    path = composer.paste_image(tmp_path)

    assert path.parent == tmp_path
    assert path.read_bytes() == PNG


@pytest.mark.asyncio
async def test_ctrl_v_queues_one_image_and_preserves_composer_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "/usr/bin/pngpaste")
    real_run = composer.subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] != "/usr/bin/pngpaste":
            return real_run(command, **kwargs)
        Path(command[1]).write_bytes(PNG)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", fake_run)
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append

    submitted = await run_ctrl_v(app, "draft\x16more")

    assert submitted == ["draft[Image #1]more"]
    assert len(app._pending_attachments) == 1
    assert notices == []
    await app.loop.close()


@pytest.mark.asyncio
async def test_ctrl_v_numbers_multiple_images_without_renumbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "/usr/bin/pngpaste")
    real_run = composer.subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] != "/usr/bin/pngpaste":
            return real_run(command, **kwargs)
        Path(command[1]).write_bytes(PNG)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", fake_run)
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )

    submitted = await run_ctrl_v(app, "a\x16b\x16c")

    assert submitted == ["a[Image #1]b[Image #2]c"]
    assert len(app._pending_attachments) == 2
    assert list(app._pending_attachment_tokens) == ["[Image #1]", "[Image #2]"]
    await app.loop.close()


@pytest.mark.asyncio
async def test_slash_paste_inserts_the_same_token_without_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "/usr/bin/pngpaste")
    real_run = composer.subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] != "/usr/bin/pngpaste":
            return real_run(command, **kwargs)
        Path(command[1]).write_bytes(PNG)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", fake_run)
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=app.abort_active,
                on_exit=app.request_exit,
            ),
            multiline=True,
        )
        run_task = asyncio.create_task(app.run(session))
        await asyncio.sleep(0)
        pipe.send_text("/paste\r")
        for _ in range(100):
            if session.app.current_buffer.text == "[Image #1]":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("/paste did not insert an image token")
        pipe.send_text("\x04")
        await run_task

    assert app._pending_attachment_tokens == {"[Image #1]": app._pending_attachments[0]}
    assert notices == []


def test_image_tokens_resolve_in_text_order_and_deleted_tokens_cancel(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(PNG)
    second.write_bytes(PNG)
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._pending_attachments[:] = [first, second]
    app._pending_attachment_tokens.update(
        {"[Image #1]": first, "[Image #2]": second}
    )
    notices: list[str] = []
    app._print_system = notices.append

    message = app._prepare_user_message("look [Image #2], not [Image #1]")

    assert message is not None
    assert message.content[0] == TextContent("look [Image #2], not [Image #1]")
    assert [
        block.path for block in message.content[1:] if isinstance(block, ImageContent)
    ] == [str(second.resolve()), str(first.resolve())]

    message = app._prepare_user_message("look [Image #2]")

    assert message is not None
    assert [
        block.path for block in message.content[1:] if isinstance(block, ImageContent)
    ] == [str(second.resolve())]
    assert app._pending_attachments == [second]
    assert notices == []


def test_unknown_image_token_is_plain_text(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )

    message = app._prepare_user_message("look at [Image #7]")

    assert message is not None
    assert message.content == [TextContent("look at [Image #7]")]


@pytest.mark.asyncio
async def test_image_token_numbering_resets_after_send(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    app._pending_attachments.append(image)
    app._pending_attachment_tokens["[Image #1]"] = image

    await app._handle_prompt_value("inspect [Image #1]")
    assert app._active_task is not None
    await app._active_task

    assert app._next_image_token == 1
    assert app._pending_attachment_tokens == {}
    assert app._pending_attachments == []
    assert store.messages()[0].content[0] == TextContent("inspect [Image #1]")
    await app.loop.close()


@pytest.mark.asyncio
async def test_cancelled_image_token_removes_staged_file_on_send(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    staged = store.session_dir / "clipboard-cancelled.png"
    staged.write_bytes(PNG)
    app._pending_attachments.append(staged)
    app._pending_attachment_tokens["[Image #1]"] = staged

    await app._handle_prompt_value("send")
    assert app._active_task is not None
    await app._active_task

    assert not staged.exists()
    assert store.messages()[0].content == [TextContent("send")]
    await app.loop.close()


@pytest.mark.asyncio
async def test_cancelled_image_token_keeps_delivered_staged_file(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    cancelled = store.session_dir / "clipboard-cancelled.png"
    delivered = store.session_dir / "clipboard-delivered.png"
    cancelled.write_bytes(PNG)
    delivered.write_bytes(PNG)
    app._pending_attachments[:] = [cancelled, delivered]
    app._pending_attachment_tokens.update(
        {"[Image #1]": cancelled, "[Image #2]": delivered}
    )

    await app._handle_prompt_value("send [Image #2]")
    assert app._active_task is not None
    await app._active_task

    assert not cancelled.exists()
    assert delivered.exists()
    assert store.messages()[0].content[0] == TextContent("send [Image #2]")
    assert store.messages()[0].content[1].path == str(delivered)
    await app.loop.close()


@pytest.mark.asyncio
async def test_partial_image_token_cancellation_removes_staged_file(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    staged = store.session_dir / "clipboard-partial.png"
    staged.write_bytes(PNG)
    app._pending_attachments.append(staged)
    app._pending_attachment_tokens["[Image #1]"] = staged

    await app._handle_prompt_value("send [Image #1")
    assert app._active_task is not None
    await app._active_task

    assert not staged.exists()
    await app.loop.close()


@pytest.mark.asyncio
async def test_paste_has_no_notice_or_pending_footer(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    with create_pipe_input() as pipe:
        session = PromptSession(input=pipe, output=DummyOutput(), multiline=True)
        prompt_task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        app._active_session = session
        app._pending_attachments.append(tmp_path / "image.png")

        toolbar = app._status_toolbar()

        assert "pending attachment" not in "".join(text for _, text in toolbar)
        session.app.exit()
        await prompt_task
    app._active_session = None
    await app.loop.close()


@pytest.mark.asyncio
async def test_ctrl_v_empty_clipboard_preserves_input_without_queueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "/usr/bin/pngpaste")
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        lambda _command, **_: SimpleNamespace(returncode=1),
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append

    submitted = await run_ctrl_v(app, "draft\x16more")

    assert submitted == ["draftmore"]
    assert app._pending_attachments == []
    assert notices == []
    await app.loop.close()


@pytest.mark.asyncio
async def test_ctrl_v_non_macos_uses_paste_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setattr(composer.platform, "system", lambda: "Linux")
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append

    submitted = await run_ctrl_v(app, "draft\x16")

    assert submitted == ["draft"]
    assert notices == [
        "paste unavailable: image paste is only available on macOS"
    ]
    await app.loop.close()


@pytest.mark.asyncio
async def test_app_submits_attachment_on_the_same_user_message(tmp_path: Path) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("context", encoding="utf-8")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )

    await app._handle_prompt_value("use @./prompt.txt")
    assert app._active_task is not None
    await app._active_task

    message = store.messages()[0]
    assert message.content[0] == TextContent("use @./prompt.txt")
    assert isinstance(message.content[1], TextContent)
    assert message.content[1].text.endswith("\ncontext")
    await app.loop.close()


@pytest.mark.asyncio
async def test_missing_path_like_attachment_blocks_with_notice(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append

    await app._handle_prompt_value("read @./missing.txt")

    assert app._active_task is None
    assert store.messages() == []
    assert notices == ["attachment rejected: file does not exist: " + str(tmp_path / "missing.txt")]
    await app.loop.close()


@pytest.mark.asyncio
async def test_missing_path_preserves_pending_paste_for_retry(tmp_path: Path) -> None:
    pending = tmp_path / "clipboard.png"
    pending.write_bytes(PNG)
    fixed = tmp_path / "fixed.txt"
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append
    app._pending_attachments.append(pending)

    await app._handle_prompt_value("read @./missing.txt")

    assert app._active_task is None
    assert app._pending_attachments == [pending]
    assert notices == ["attachment rejected: file does not exist: " + str(tmp_path / "missing.txt")]

    fixed.write_text("context", encoding="utf-8")
    await app._handle_prompt_value("read @./fixed.txt")
    assert app._active_task is not None
    await app._active_task

    users = [
        message for message in store.messages() if message.role is MessageRole.USER
    ]
    assert len(users) == 1
    assert [
        block.path
        for block in users[0].content
        if isinstance(block, (ImageContent, TextContent))
    ] == [
        None,
        str(fixed.resolve()),
        str(pending.resolve()),
    ]
    assert app._pending_attachments == []
    await app.loop.close()


@pytest.mark.asyncio
async def test_deleted_pending_paste_is_dropped_once(tmp_path: Path) -> None:
    pending = tmp_path / "clipboard.png"
    pending.write_bytes(PNG)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    notices: list[str] = []
    app._print_system = notices.append
    app._pending_attachments.append(pending)
    pending.unlink()

    await app._handle_prompt_value("first")
    assert app._active_task is not None
    await app._active_task
    await app._handle_prompt_value("second")
    assert app._active_task is not None
    await app._active_task

    users = [
        message for message in store.messages() if message.role is MessageRole.USER
    ]
    assert [message.content for message in users] == [
        [TextContent("first")],
        [TextContent("second")],
    ]
    assert len(notices) == 1
    assert notices[0].startswith("pending attachment dropped:")
    await app.loop.close()
