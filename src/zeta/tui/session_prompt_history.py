"""In-memory prompt history scoped to the session currently shown."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from prompt_toolkit.document import Document


class SessionPromptHistory:
    """Keep editable prompts and navigation state for each viewed session."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[str, ...]] = {}
        self._session: Path | None = None
        self._index: int | None = None
        self._draft: Document | None = None

    def replace(self, session_path: Path, prompts: Sequence[str]) -> None:
        self._entries[session_path] = tuple(prompts)
        if session_path == self._session:
            self.reset_navigation()

    def append(self, session_path: Path, prompt: str) -> None:
        if not prompt:
            return
        self._entries[session_path] = (*self._entries.get(session_path, ()), prompt)
        if session_path == self._session:
            self.reset_navigation()

    def snapshot(self, session_path: Path) -> tuple[str, ...]:
        return self._entries.get(session_path, ())

    def older(self, session_path: Path, draft: Document) -> Document | None:
        self._select_session(session_path, draft)
        entries = self._entries.get(session_path, ())
        if not entries:
            return None
        if self._index is None:
            self._draft = draft
            self._index = len(entries) - 1
        elif self._index > 0:
            self._index -= 1
        return Document(entries[self._index], cursor_position=len(entries[self._index]))

    def newer(self, session_path: Path) -> Document | None:
        if self._session != session_path or self._index is None:
            return None
        entries = self._entries.get(session_path, ())
        if self._index < len(entries) - 1:
            self._index += 1
            value = entries[self._index]
            return Document(value, cursor_position=len(value))
        draft = self._draft
        self.reset_navigation()
        return draft

    def reset_navigation(self) -> None:
        self._session = None
        self._index = None
        self._draft = None

    def _select_session(self, session_path: Path, draft: Document) -> None:
        if self._session != session_path:
            self._session = session_path
            self._index = None
            self._draft = draft
