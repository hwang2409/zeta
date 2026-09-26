"""Resolve composer file references and stage clipboard images."""

from __future__ import annotations

import base64
import os
import platform
import re
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ..core.process_env import subprocess_env
from ..core.session_files import (
    child_directory,
    open_session_file,
    session_root,
    write_session_file,
)
from ..core.store import ConversationStore
from ..media.images import image_signature_matches
from ..protocol.types import ImageContent, Message, MessageRole, TextContent

ATTACHMENT_MAX_TEXT_BYTES = 200 * 1024
ATTACHMENT_TOKEN_RE = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s]+))')


class AttachmentError(ValueError):
    """Raised when a composer attachment cannot be read or decoded."""


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    token: str
    path: Path


def attachment_refs(value: str, base_dir: str | Path) -> tuple[AttachmentRef, ...]:
    """Parse quoted or path-like local ``@`` references.

    Bare words such as ``@user`` and ``@dataclass`` remain prompt text.
    """

    base = Path(base_dir)
    refs: list[AttachmentRef] = []
    for match in ATTACHMENT_TOKEN_RE.finditer(value):
        raw_path = match.group(1) or match.group(2)
        if raw_path is None:
            continue
        quoted = match.group(1) is not None
        if not quoted and not (
            "/" in raw_path or raw_path.startswith(("./", "../", "~/"))
        ):
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base / path
        refs.append(AttachmentRef(match.group(0), _attachment_path(path)))
    return tuple(refs)


def _image_media_type(data: bytes) -> str | None:
    if (
        len(data) >= 16
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] in {b"VP8 ", b"VP8L", b"VP8X"}
    ):
        return "image/webp"
    candidates = (
        ("image/png", data.startswith(b"\x89PNG\r\n\x1a\n")),
        ("image/jpeg", data.startswith(b"\xff\xd8\xff")),
        ("image/gif", data.startswith((b"GIF87a", b"GIF89a"))),
    )
    for media_type, matches in candidates:
        if matches and image_signature_matches(media_type, data):
            return media_type
    return None


def _attachment_path(path: Path) -> Path:
    # Preserve session components so link validation happens at descriptor opens.
    return path.absolute() if "sessions" in path.parts else path.resolve()


def _read_attachment(
    path: Path, session_store: ConversationStore | None = None
) -> TextContent | ImageContent:
    try:
        with ExitStack() as cleanup:
            if session_store is not None and path.is_relative_to(
                session_store.session_dir
            ):
                directory_fd = session_store.directory_fd
                for component in path.parent.relative_to(
                    session_store.session_dir
                ).parts:
                    directory_fd = cleanup.enter_context(
                        child_directory(directory_fd, component)
                    )
                handle = cleanup.enter_context(
                    os.fdopen(
                        open_session_file(directory_fd, path.name, os.O_RDONLY), "rb"
                    )
                )
            elif "sessions" in path.parts:
                directory_fd = cleanup.enter_context(session_root(path.parent))
                handle = cleanup.enter_context(
                    os.fdopen(
                        open_session_file(directory_fd, path.name, os.O_RDONLY), "rb"
                    )
                )
            else:
                if not path.exists():
                    raise AttachmentError(f"file does not exist: {path}")
                if not path.is_file():
                    raise AttachmentError(
                        f"directory attachments are not supported: {path}"
                    )
                handle = cleanup.enter_context(path.open("rb"))
            size = os.fstat(handle.fileno()).st_size
            prefix = handle.read(64)
            if _image_media_type(prefix) is None and size > ATTACHMENT_MAX_TEXT_BYTES:
                raise AttachmentError(
                    f"text file is {size} bytes; limit is {ATTACHMENT_MAX_TEXT_BYTES} bytes: {path}"
                )
            data = prefix + handle.read()
    except FileNotFoundError as exc:
        raise AttachmentError(f"file does not exist: {path}") from exc
    except IsADirectoryError as exc:
        raise AttachmentError(
            f"directory attachments are not supported: {path}"
        ) from exc
    except AttachmentError:
        raise
    except (OSError, ValueError) as exc:
        raise AttachmentError(f"cannot read {path}: {exc}") from exc
    media_type = _image_media_type(prefix)
    if media_type is not None:
        if not image_signature_matches(media_type, data):
            raise AttachmentError(f"binary file is not an image: {path}")
        return ImageContent(
            base64.b64encode(data).decode("ascii"),
            media_type,
            str(path),
            size,
        )
    if b"\x00" in data:
        raise AttachmentError(f"binary file is not an image: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"binary file is not an image: {path}") from None
    labeled = f"[file: {path} · {size} bytes]\n{text}"
    return TextContent(labeled, str(path), size)


def build_user_message(
    value: str,
    base_dir: str | Path,
    pending_paths: tuple[Path, ...] = (),
    *,
    attachment_value: str | None = None,
    session_store: ConversationStore | None = None,
) -> Message:
    """Resolve references into one message, deduplicating resolved paths."""

    paths: list[Path] = []
    source = value if attachment_value is None else attachment_value
    for ref in attachment_refs(source, base_dir):
        if ref.path not in paths:
            paths.append(ref.path)
    for path in pending_paths:
        resolved = _attachment_path(path)
        if resolved not in paths:
            paths.append(resolved)
    blocks = [TextContent(value)]
    blocks.extend(_read_attachment(path, session_store) for path in paths)
    return Message(MessageRole.USER, blocks)


def paste_image(session_dir: str | Path, *, directory_fd: int | None = None) -> Path:
    """Save a macOS clipboard image in the session directory."""

    if platform.system() != "Darwin":
        raise AttachmentError("image paste is only available on macOS")
    with tempfile.TemporaryDirectory(prefix="zeta-clipboard-") as temporary:
        destination = Path(temporary) / "clipboard.png"
        pngpaste = shutil.which("pngpaste")
        if pngpaste is not None:
            result = subprocess.run(
                [pngpaste, str(destination)],
                capture_output=True,
                check=False,
                env=subprocess_env(),
            )
        else:
            script = """
use framework "AppKit"
on run argv
    set destination to item 1 of argv
    set imageData to current application's NSPasteboard's generalPasteboard()'s dataForType:(current application's NSPasteboardTypePNG)
    if imageData is missing value then return "empty"
    imageData's writeToFile:destination atomically:true
    return "ok"
end run
"""
            result = subprocess.run(
                ["osascript", "-e", script, str(destination)],
                capture_output=True,
                check=False,
                env=subprocess_env(),
            )
        if result.returncode != 0 or not destination.is_file():
            raise AttachmentError("clipboard does not contain an image")
        try:
            data = destination.read_bytes()
        except OSError as exc:
            raise AttachmentError(f"cannot read clipboard image: {exc}") from exc
        if not data:
            raise AttachmentError("clipboard does not contain an image")
    name = f"clipboard-{uuid4().hex}.png"
    directory = (
        nullcontext(directory_fd)
        if directory_fd is not None
        else session_root(Path(session_dir))
    )
    try:
        with directory as pinned_fd:
            write_session_file(pinned_fd, name, data)
    except (OSError, ValueError) as exc:
        raise AttachmentError(f"cannot save clipboard image: {exc}") from exc
    return Path(session_dir) / name
