"""Persistence for per-session context preferences."""

from __future__ import annotations

from typing import Any


class SessionPreferenceMixin:
    """Persist context preferences with optimistic concurrency."""

    def record_budget(
        self,
        metadata: Any,
        *,
        budget: int,
        pinned: bool,
        touch: bool = True,
    ) -> None:
        """Persist the compaction budget without losing concurrent updates."""

        from ..session import SessionError

        expected = metadata.compaction_budget

        def update(item: Any) -> Any:
            if item.compaction_budget != expected:
                raise SessionError(
                    "session budget changed before commit; winner: "
                    f"compaction_budget={item.compaction_budget!r}"
                )
            item.compaction_budget = budget
            item.budget_pinned = pinned
            return self._touch(item) if touch else item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
