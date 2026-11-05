"""Persistent composer history and draft state."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
import fcntl
import json
import logging
import os
import queue
import tempfile
import threading
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from ...core.checkpoints import ConversationIntegrityError, load_session_json
from ...core.session_files import read_session_file, session_root, write_session_json

DRAFT_WRITE_DELAY = 0.2
PERSISTENCE_FLUSH_TIMEOUT = 2.0
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FileAction:
    watermark: int
    action: Callable[[], object]
    completed: Callable[[object], None] | None
    stop: bool = False


class _OrderedFileWriter:
    """Own one ordered daemon thread and expose bounded completion waits."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._queue: queue.Queue[_FileAction] = queue.Queue()
        self._lock = threading.Lock()
        self._completed = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._applied = 0
        self._accepting = True
        self._waiters: list[tuple[int, asyncio.AbstractEventLoop, asyncio.Future[None]]] = []
        self._reported_failure = False

    def submit(
        self,
        action: Callable[[], object],
        completed: Callable[[object], None] | None = None,
        *,
        stop: bool = False,
    ) -> int:
        try:
            asyncio.get_running_loop()
            wait = False
        except RuntimeError:
            wait = True
        with self._lock:
            if not self._accepting:
                raise RuntimeError(f"{self._label} writer is closed")
            self._submitted += 1
            watermark = self._submitted
            if stop:
                self._accepting = False
            self._queue.put(_FileAction(watermark, action, completed, stop))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name=f"zeta-{self._label.replace(' ', '-')}",
                    daemon=True,
                )
                self._thread.start()
        if wait:
            with self._completed:
                self._completed.wait_for(lambda: self._applied >= watermark)
        return watermark

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            result: object | None = None
            failed = False
            try:
                result = item.action()
            except Exception as exc:  # noqa: BLE001 - persistence must not crash the TUI
                failed = True
                self._report_failure(exc)
            self._deliver(item, result, failed)
            with self._lock:
                if item.stop or self._queue.empty():
                    self._thread = None
                    return

    def _deliver(self, item: _FileAction, result: object | None, failed: bool) -> None:
        try:
            if not failed and item.completed is not None:
                item.completed(result)
        except Exception as exc:  # noqa: BLE001 - completion must not stop the writer
            self._report_failure(exc)
        finally:
            self._mark_applied(item.watermark)

    def _mark_applied(self, watermark: int) -> None:
        with self._completed:
            self._applied = max(self._applied, watermark)
            ready = [waiter for waiter in self._waiters if waiter[0] <= self._applied]
            self._waiters = [waiter for waiter in self._waiters if waiter[0] > self._applied]
            self._completed.notify_all()
        for _, loop, future in ready:
            try:
                loop.call_soon_threadsafe(self._finish_waiter, future)
            except RuntimeError:
                pass

    @staticmethod
    def _finish_waiter(future: asyncio.Future[None]) -> None:
        if not future.done():
            future.set_result(None)

    async def flush(self, *, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            watermark = self._submitted
            if self._applied >= watermark:
                return
            future = loop.create_future()
            self._waiters.append((watermark, loop, future))
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout)
        except TimeoutError as exc:
            with self._lock:
                self._waiters = [waiter for waiter in self._waiters if waiter[2] is not future]
            self._report_failure(exc)

    async def shutdown(
        self,
        finalizer: Callable[[], object] = lambda: None,
        *,
        timeout: float,
    ) -> None:
        with self._lock:
            accepting = self._accepting
        if accepting:
            self.submit(finalizer, stop=True)
        await self.flush(timeout=timeout)

    def _report_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._reported_failure:
                return
            self._reported_failure = True
        _LOGGER.warning("failed to persist %s: %s", self._label, exc)


class DraftPersistence:
    """Persist the current composer text after a short idle delay."""

    def __init__(self, path: str | Path, *, delay: float = DRAFT_WRITE_DELAY, directory_fd: int | None = None) -> None:
        self.path = Path(path)
        self.directory_fd = directory_fd
        self.delay = delay
        self._pending_text: str | None = None
        self._pending_revision: int | None = None
        self._revision = 0
        self._persisted_revision: int | None = None
        self._scheduled: asyncio.TimerHandle | None = None
        self._state_provider: Callable[[], tuple[Mapping[str, Path], int]] | None = None
        self._buffer: Buffer | None = None
        self._changed: Callable[[Buffer], None] | None = None
        self._pending_attachment_tokens: tuple[tuple[str, str], ...] = ()
        self._pending_next_image_token = 1
        self._writer = _OrderedFileWriter("composer draft")

    def load(self) -> str:
        return self.load_state().text

    def load_state(self) -> DraftState:
        try:
            with self._directory() as directory_fd:
                raw = read_session_file(directory_fd, self.path.name).decode("utf-8")
        except (OSError, ValueError):
            return DraftState("")
        try:
            payload = load_session_json(raw.encode("utf-8"))
        except ConversationIntegrityError as exc:
            # Legacy drafts are plain text; structural/encoding failures are not.
            if isinstance(exc.__cause__, json.JSONDecodeError):
                return DraftState(raw)
            return DraftState("")
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
        submitted = self._submit(
            lambda directory_fd: self._persist(
                text,
                attachment_tokens=attachment_tokens,
                next_image_token=next_image_token,
                directory_fd=directory_fd,
            )
        )
        if submitted:
            self._persisted_revision = revision
        else:
            self._pending_text = text
            self._pending_revision = revision
            self._pending_attachment_tokens = attachment_tokens
            self._pending_next_image_token = next_image_token

    def _submit(self, action: Callable[[int | None], None]) -> bool:
        try:
            owned_fd = os.dup(self.directory_fd) if self.directory_fd is not None else None
        except OSError as exc:
            self._writer._report_failure(exc)
            return False

        def run() -> None:
            try:
                action(owned_fd)
            finally:
                if owned_fd is not None:
                    os.close(owned_fd)

        try:
            self._writer.submit(run)
        except Exception:
            if owned_fd is not None:
                os.close(owned_fd)
            raise
        return True

    def _persist(
        self,
        text: str,
        *,
        attachment_tokens: tuple[tuple[str, str], ...],
        next_image_token: int,
        directory_fd: int | None = None,
    ) -> None:
        if not text:
            self._unlink(directory_fd)
            return
        with self._directory(create=True, directory_fd=directory_fd) as active_fd:
            write_session_json(active_fd, self.path.name, {
                "text": text,
                "attachment_tokens": dict(attachment_tokens),
                "next_image_token": next_image_token,
            })

    async def flush_pending(self) -> None:
        self.flush()
        await self._writer.flush(timeout=PERSISTENCE_FLUSH_TIMEOUT)

    async def shutdown_pending(self) -> None:
        try:
            self.flush()
        finally:
            await self._writer.shutdown(timeout=PERSISTENCE_FLUSH_TIMEOUT)

    def _directory(self, *, create: bool = False, directory_fd: int | None = None):
        active_fd = self.directory_fd if directory_fd is None else directory_fd
        if active_fd is not None:
            return nullcontext(active_fd)
        return session_root(self.path.parent, create=create)

    def _unlink(self, directory_fd: int | None = None) -> None:
        try:
            with self._directory(directory_fd=directory_fd) as active_fd:
                os.unlink(self.path.name, dir_fd=active_fd)
        except FileNotFoundError:
            pass

    def clear(self) -> None:
        self._pending_text = None
        self._pending_revision = None
        self._pending_attachment_tokens = ()
        self._pending_next_image_token = 1
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        self._submit(self._unlink)
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
            self._submit(self._unlink)
            self._persisted_revision = None
        return True

    def attach(
        self,
        buffer: Buffer,
        *,
        state_provider: Callable[[], tuple[Mapping[str, Path], int]] | None = None,
    ) -> None:
        self.detach()
        self._state_provider = state_provider
        draft = self.load_state()
        if draft.text:
            self._persisted_revision = self._revision
        if draft.text and not buffer.text:
            buffer.set_document(Document(draft.text, len(draft.text)))

        def changed(_buffer: Buffer) -> None:
            self.schedule(buffer.text)

        self._buffer = buffer
        self._changed = changed
        buffer.on_text_changed += changed

    def detach(self) -> None:
        """Stop observing the composer before its session storage closes."""
        try:
            self.flush()
        finally:
            if self._buffer is not None and self._changed is not None:
                self._buffer.on_text_changed -= self._changed
            self._buffer = None
            self._changed = None
            self._state_provider = None


@dataclass(frozen=True, slots=True)
class DraftState:
    text: str
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1
