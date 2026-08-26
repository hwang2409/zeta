from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
    build_user_message,
)
from zeta.types import ImageContent, MessageRole, TextContent

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


def test_attachment_refs_support_quoted_paths(tmp_path: Path) -> None:
    refs = attachment_refs('read @notes.txt and @"a b.txt"', tmp_path)

    assert [ref.token for ref in refs] == ["@notes.txt", '@"a b.txt"']
    assert refs[1].path == (tmp_path / "a b.txt").resolve()


def test_build_user_message_inlines_text_and_preserves_prompt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    message = build_user_message("summarize @notes.txt", tmp_path)

    assert message.role is MessageRole.USER
    assert message.content[0] == TextContent("summarize @notes.txt")
    attachment = message.content[1]
    assert isinstance(attachment, TextContent)
    assert attachment.path == str(path.resolve())
    assert attachment.size == 5
    assert attachment.text.endswith("\nhello")


def test_build_user_message_rejects_missing_large_and_binary_files(tmp_path: Path) -> None:
    with pytest.raises(AttachmentError, match="does not exist"):
        build_user_message("@missing.txt", tmp_path)

    large = tmp_path / "large.txt"
    large.write_bytes(b"a" * (ATTACHMENT_MAX_TEXT_BYTES + 1))
    with pytest.raises(AttachmentError, match="limit"):
        build_user_message("@large.txt", tmp_path)

    binary = tmp_path / "data.bin"
    binary.write_bytes(b"header\x00payload")
    with pytest.raises(AttachmentError, match="binary"):
        build_user_message("@data.bin", tmp_path)


def test_image_attachment_round_trips_and_uses_provider_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "renamed.data"
    path.write_bytes(PNG)
    message = build_user_message("inspect @renamed.data", tmp_path)
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

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        Path(command[1]).write_bytes(PNG)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", fake_run)
    path = composer.paste_image(tmp_path)

    assert path.parent == tmp_path
    assert path.read_bytes() == PNG


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

    await app._handle_prompt_value("use @prompt.txt")
    assert app._active_task is not None
    await app._active_task

    message = store.messages()[0]
    assert message.content[0] == TextContent("use @prompt.txt")
    assert isinstance(message.content[1], TextContent)
    assert message.content[1].text.endswith("\ncontext")
    await app.loop.close()
