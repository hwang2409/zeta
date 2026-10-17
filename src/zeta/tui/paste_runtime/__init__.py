"""Ordered, session-scoped clipboard image pastes."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.buffer import Buffer

from ...protocol.types import ImageContent
from .._attachments import AttachmentError, _read_attachment, paste_image


@dataclass(frozen=True, slots=True)
class StagedImage:
    path: Path
    content: ImageContent


class PasteRuntime:
    """Stage and publish clipboard images in invocation order."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self._generation = 0
        self._buffer_generation = 0
        self._tasks: set[asyncio.Task[str]] = set()
        self._tail: asyncio.Task[str] | None = None
        self._insertions: dict[int, list[tuple[int, int]]] = {}

    @property
    def tasks(self) -> set[asyncio.Task[str]]:
        return self._tasks

    def start(self, buffer: Buffer | None = None) -> asyncio.Task[str]:
        return self._start(buffer, notify=True)

    def _start(self, buffer: Buffer | None, *, notify: bool) -> asyncio.Task[str]:
        previous = self._tail
        generation = self._generation
        buffer_generation = self._buffer_generation
        session = self._host._active_session
        cursor_position = buffer.cursor_position if buffer is not None else None
        task = asyncio.create_task(
            self._run(
                previous,
                generation,
                buffer_generation,
                session,
                buffer,
                cursor_position,
                notify,
            )
        )
        self._tail = task
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    def _finished(self, task: asyncio.Task[str]) -> None:
        self._tasks.discard(task)
        if not self._tasks:
            self._tail = None
            self._insertions.clear()

    async def paste(self, buffer: Buffer | None = None) -> str:
        return await self._start(buffer, notify=False)

    def advance_buffer_generation(self) -> None:
        """Retire paste targets when the composer accepts a submission."""

        self._buffer_generation += 1
        self._insertions.clear()

    async def reset(self) -> None:
        """Cancel and await every paste from the replaced UI session."""

        self._generation += 1
        self.advance_buffer_generation()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tail = None
        self._insertions.clear()

    async def _run(
        self,
        previous: asyncio.Task[str] | None,
        generation: int,
        buffer_generation: int,
        session: Any,
        buffer: Buffer | None,
        cursor_position: int | None,
        notify: bool,
    ) -> str:
        host = self._host
        store = host.loop.store
        session_dir = store.session_dir
        directory_fd = store.directory_fd
        worker = asyncio.create_task(asyncio.to_thread(self._stage, store))
        staged: StagedImage | None = None
        try:
            if previous is not None:
                await asyncio.gather(asyncio.shield(previous), return_exceptions=True)
            staged = await asyncio.shield(worker)
        except asyncio.CancelledError:
            try:
                staged = await worker
            except (AttachmentError, OSError):
                pass
            if staged is not None:
                await asyncio.to_thread(
                    self._delete_staged, staged.path, session_dir, directory_fd
                )
            # Successors wait on this task to keep FIFO order, so it must not
            # finish before its own predecessor, even when it is canceled.
            if previous is not None:
                await asyncio.wait({previous})
            raise
        except AttachmentError as exc:
            result = f"paste unavailable: {exc}"
            if notify:
                self._publish_notice(result)
            return result

        if not self._is_current(generation, buffer_generation, session, buffer):
            await asyncio.to_thread(
                self._delete_staged, staged.path, session_dir, directory_fd
            )
            return ""

        result = host._finish_image_paste(staged)
        if buffer is not None and cursor_position is not None:
            self._insert(buffer, cursor_position, result)
            return ""
        return result

    @staticmethod
    def _stage(store: Any) -> StagedImage:
        path = paste_image(store.session_dir, directory_fd=store.directory_fd)
        try:
            content = _read_attachment(path, store)
            if not isinstance(content, ImageContent):
                raise AttachmentError("clipboard image could not be decoded")
            return StagedImage(path, content)
        except (AttachmentError, OSError):
            PasteRuntime._delete_staged(
                path, store.session_dir, store.directory_fd
            )
            raise

    def _is_current(
        self,
        generation: int,
        buffer_generation: int,
        session: Any,
        buffer: Buffer | None,
    ) -> bool:
        if (
            generation != self._generation
            or buffer_generation != self._buffer_generation
            or session is not self._host._active_session
        ):
            return False
        if buffer is None or session is None:
            return True
        return session.app.current_buffer is buffer

    def _insert(self, buffer: Buffer, position: int, token: str) -> None:
        insertions = self._insertions.setdefault(id(buffer), [])
        adjusted = position + sum(length for start, length in insertions if start <= position)
        current_position = buffer.cursor_position
        buffer.cursor_position = min(adjusted, len(buffer.text))
        buffer.insert_text(token)
        if current_position >= adjusted:
            buffer.cursor_position = current_position + len(token)
        insertions.append((position, len(token)))

    def _publish_notice(self, result: str) -> None:
        if result != "paste unavailable: clipboard does not contain an image":
            self._host._print_system(result)

    @staticmethod
    def _delete_staged(
        path: Path, session_dir: Path, directory_fd: int | None
    ) -> None:
        if (
            path.parent == session_dir
            and path.name.startswith("clipboard-")
            and path.suffix == ".png"
        ):
            try:
                if directory_fd is None:
                    path.unlink()
                else:
                    os.unlink(path.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


__all__ = ["PasteRuntime", "StagedImage"]
