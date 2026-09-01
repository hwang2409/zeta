"""Persistent composer history and draft state."""

from __future__ import annotations

import asyncio
import os
import tempfile
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
        if not self._loaded:
            self._loaded_strings = list(self.load_history_strings())
            self._loaded = True
        if self._loaded_strings and self._loaded_strings[0] == string:
            return
        self._loaded_strings.insert(0, string)
        del self._loaded_strings[self.limit :]
        self._write_entries(self._loaded_strings[::-1])

    def store_string(self, string: str) -> None:
        """Keep direct history writes bounded as well."""

        if self._loaded:
            entries = [string, *self._loaded_strings]
        else:
            entries = [string, *self.load_history_strings()]
        self._write_entries(entries[: self.limit][::-1])

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
        self._submitted_revision: int | None = None
        self._scheduled: asyncio.TimerHandle | None = None

    def load(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def schedule(self, text: str) -> None:
        self._revision += 1
        self._pending_text = text
        self._pending_revision = self._revision
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
        self._pending_text = None
        self._pending_revision = None
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
                handle.write(text)
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
        self._submitted_revision = None
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        self.path.unlink(missing_ok=True)
        self._persisted_revision = None

    def mark_submitted(self) -> None:
        """Remember the draft revision submitted by the prompt callback."""

        self._submitted_revision = self._revision

    def clear_submitted(self) -> bool:
        """Clear only the revision captured by the most recent submission."""

        submitted_revision = self._submitted_revision
        self._submitted_revision = None
        if submitted_revision is None:
            return False
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

    def attach(self, buffer: Buffer) -> None:
        draft = self.load()
        if draft:
            self._persisted_revision = self._revision
        if draft and not buffer.text:
            buffer.set_document(Document(draft, len(draft)))

        def changed(_buffer: Buffer) -> None:
            self.schedule(buffer.text)

        buffer.on_text_changed += changed


def history_for(path: str | Path) -> BoundedFileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return BoundedFileHistory(history_path)
