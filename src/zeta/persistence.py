"""Persistent composer history and draft state."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

HISTORY_LIMIT = 1000
DRAFT_WRITE_DELAY = 0.2


class BoundedFileHistory(FileHistory):
    """Store recent composer entries without retaining attachment payloads."""

    def __init__(self, filename: str | Path, *, limit: int = HISTORY_LIMIT) -> None:
        if limit < 1:
            raise ValueError("history limit must be positive")
        self.limit = limit
        super().__init__(str(filename))

    def load_history_strings(self) -> list[str]:
        return list(super().load_history_strings())[: self.limit]

    def append_string(self, string: str) -> None:
        with self._locked():
            entries = list(self.load_history_strings())
            if entries and entries[0] == string:
                self._loaded_strings = entries
                self._loaded = True
                return
            entries.insert(0, string)
            del entries[self.limit :]
            self._write_entries(entries[::-1])
        self._loaded_strings = entries
        self._loaded = True

    def store_string(self, string: str) -> None:
        """Keep direct history writes bounded as well."""

        with self._locked():
            entries = [string, *self.load_history_strings()]
            entries = entries[: self.limit]
            self._write_entries(entries[::-1])
        self._loaded_strings = entries
        self._loaded = True

    @contextmanager
    def _locked(self) -> Iterator[None]:
        history_path = Path(self.filename)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = history_path.with_name(f".{history_path.name}.lock")
        lock = lock_path.open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _write_entries(self, entries: list[str]) -> None:
        history_path = Path(self.filename)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=history_path.parent,
                prefix=f".{history_path.name}.",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                for entry in entries:
                    handle.write(b"\n# zeta composer history\n")
                    for line in entry.split("\n"):
                        handle.write(f"+{line}\n".encode("utf-8", errors="replace"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, history_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)


class DraftPersistence:
    """Persist the current composer text after a short idle delay."""

    def __init__(self, path: str | Path, *, delay: float = DRAFT_WRITE_DELAY) -> None:
        self.path = Path(path)
        self.delay = delay
        self._pending_text: str | None = None
        self._pending_revision: int | None = None
        self._revision = 0
        self._persisted_revision: int | None = None
        self._scheduled: asyncio.TimerHandle | None = None
        self._state_provider: Callable[[], tuple[Mapping[str, Path], int]] | None = None
        self._pending_attachment_tokens: tuple[tuple[str, str], ...] = ()
        self._pending_next_image_token = 1

    def load(self) -> str:
        return self.load_state().text

    def load_state(self) -> DraftState:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return DraftState("")
        except OSError:
            return DraftState("")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return DraftState(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return DraftState("")
        raw_tokens = payload.get("attachment_tokens", {})
        tokens = (
            tuple(
                (token, Path(path))
                for token, path in raw_tokens.items()
                if isinstance(token, str) and isinstance(path, str)
            )
            if isinstance(raw_tokens, dict)
            else ()
        )
        next_image_token = payload.get("next_image_token", 1)
        if type(next_image_token) is not int or next_image_token < 1:
            next_image_token = 1
        return DraftState(payload["text"], tokens, next_image_token)

    def schedule(
        self,
        text: str,
        *,
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int | None = None,
    ) -> None:
        if attachment_tokens is None and self._state_provider is not None:
            attachment_tokens, next_image_token = self._state_provider()
        attachment_tokens = attachment_tokens or {}
        if next_image_token is None:
            next_image_token = 1
        self._revision += 1
        self._pending_text = text
        self._pending_revision = self._revision
        self._pending_attachment_tokens = tuple(
            (token, str(Path(path))) for token, path in attachment_tokens.items()
        )
        self._pending_next_image_token = next_image_token
        if self._scheduled is not None:
            self._scheduled.cancel()
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        self._scheduled = event_loop.call_later(self.delay, self.flush)

    def flush(self) -> None:
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        text = self._pending_text
        revision = self._pending_revision
        attachment_tokens = self._pending_attachment_tokens
        next_image_token = self._pending_next_image_token
        self._pending_text = None
        self._pending_revision = None
        self._pending_attachment_tokens = ()
        self._pending_next_image_token = 1
        if text is None:
            return
        if not text:
            self.path.unlink(missing_ok=True)
            self._persisted_revision = revision
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump(
                    {
                        "text": text,
                        "attachment_tokens": dict(attachment_tokens),
                        "next_image_token": next_image_token,
                    },
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._persisted_revision = revision
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    def clear(self) -> None:
        self._pending_text = None
        self._pending_revision = None
        self._pending_attachment_tokens = ()
        self._pending_next_image_token = 1
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        self.path.unlink(missing_ok=True)
        self._persisted_revision = None

    def mark_submitted(self) -> int:
        """Return the draft revision captured by the prompt callback."""

        return self._revision

    def clear_submitted(self, submitted_revision: int) -> bool:
        """Clear only the draft revision captured by one submission."""

        if (
            self._pending_revision is not None
            and self._pending_revision <= submitted_revision
        ):
            self._pending_text = None
            self._pending_revision = None
            if self._scheduled is not None:
                self._scheduled.cancel()
                self._scheduled = None
        if (
            self._persisted_revision is not None
            and self._persisted_revision <= submitted_revision
        ):
            self.path.unlink(missing_ok=True)
            self._persisted_revision = None
        return True

    def attach(
        self,
        buffer: Buffer,
        *,
        state_provider: Callable[[], tuple[Mapping[str, Path], int]] | None = None,
    ) -> None:
        self._state_provider = state_provider
        draft = self.load_state()
        if draft.text:
            self._persisted_revision = self._revision
        if draft.text and not buffer.text:
            buffer.set_document(Document(draft.text, len(draft.text)))

        def changed(_buffer: Buffer) -> None:
            self.schedule(buffer.text)

        buffer.on_text_changed += changed


@dataclass(frozen=True, slots=True)
class DraftState:
    text: str
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1


def history_for(path: str | Path) -> BoundedFileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return BoundedFileHistory(history_path)
