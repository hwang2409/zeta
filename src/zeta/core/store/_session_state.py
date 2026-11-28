"""Durable session-state publication."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from ...storage_format import WriterLease
from ..agent_state import _apply_agent_state, _session_state_mutations
from ..checkpoints import ConversationIntegrityError
from ..session_files import write_session_file
from ..todo import TodoItem, parse_todo_items


class SessionStatePersistenceMixin:
    """Publish session state through the stored-format writer."""

    _storage_lease: WriterLease

    def set_todo_change_sink(
        self, sink: Callable[[list[TodoItem]], None] | None
    ) -> None:
        """Attach a callback that runs after a todo replacement commits."""
        self._todo_change_sink = sink

    def set_todo_items(self, items: object) -> None:
        """Replace the session todo list in one atomic state-file update."""
        normalized = parse_todo_items(items)
        with self._append_lock():
            self._todo_items = [dict(item) for item in normalized]
            self._todo_dismissed = False
            self._write_session_state(self.bash_cwd, normalized)
        sink = getattr(self, "_todo_change_sink", None)
        if sink is not None:
            try:
                sink([dict(item) for item in normalized])
            except Exception:
                logging.getLogger(__name__).exception("todo change callback failed")

    def _write_session_state(
        self,
        bash_cwd: str,
        todo_items: Iterable[TodoItem],
        *,
        deleted_agent_children: Iterable[str] = (),
    ) -> None:
        state: dict[str, Any] = {"bash_cwd": bash_cwd}
        normalized_items = [dict(item) for item in todo_items]
        if normalized_items:
            state["todo_items"] = normalized_items
        if self._todo_dismissed:
            state["todo_dismissed"] = True
        deleted = tuple(deleted_agent_children)
        dirty, self._dirty_agent_children = self._dirty_agent_children, set()
        _apply_agent_state(
            state,
            agent_counter=self._agent_counter,
            agent_children={
                key: self._agent_children[key]
                for key in dirty
                if key in self._agent_children and key not in deleted
            },
            agent_parent=self._agent_parent,
            agent_canceled=self._agent_canceled,
        )
        mutations = _session_state_mutations(state, deleted)
        published: list[bytes] = []

        def publish(directory_fd: int, name: str, data: bytes) -> None:
            write_session_file(directory_fd, name, data)
            published.append(data)

        try:
            output = self._storage_lease.publish_json(
                self.directory_fd, "session_state.json", mutations, publish
            )
        except BaseException:
            self._session_state_cache = None
            self._dirty_agent_children |= dirty
            raise
        try:
            # Writes hold the session append lock, so these bytes are the file now.
            self._session_state_cache = (published[0], self._parse_session_state(output))
        except ConversationIntegrityError:
            self._session_state_cache = None
