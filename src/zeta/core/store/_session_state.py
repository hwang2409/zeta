"""Durable session-state publication."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...storage_format import WriterLease
from ..agent_state import _apply_agent_state, _session_state_mutations
from ..session_files import write_session_file
from ..todo import TodoItem


class SessionStatePersistenceMixin:
    """Publish session state through the stored-format writer."""

    _storage_lease: WriterLease

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
        _apply_agent_state(
            state,
            agent_counter=self._agent_counter,
            agent_children=self._agent_children,
            agent_parent=self._agent_parent,
            agent_canceled=self._agent_canceled,
        )
        mutations = _session_state_mutations(state, tuple(deleted_agent_children))
        self._storage_lease.publish_json(
            self.directory_fd,
            "session_state.json",
            mutations,
            write_session_file,
        )
