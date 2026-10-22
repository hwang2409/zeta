"""Persistence for per-session context preferences."""

from __future__ import annotations

from typing import Any


class SessionPreferenceMixin:
    """Persist context preferences with optimistic concurrency."""

    def record_account_selection(
        self,
        metadata: Any,
        account: Any,
        *,
        boundary: int,
        switched: bool = False,
        lock_timeout: float | None = None,
    ) -> None:
        """Persist one provider selection and its native-context boundary atomically."""

        def update(item: Any) -> Any:
            return self._apply_account_selection(
                item, account, boundary=boundary, switched=switched
            )

        current = self._mutate(
            metadata.session_id, update, lock_timeout=lock_timeout
        )
        self._copy_metadata(metadata, current)

    @classmethod
    def _apply_account_selection(
        cls,
        metadata: Any,
        account: Any,
        *,
        boundary: int,
        switched: bool,
    ) -> Any:
        provider = account.provider
        metadata.selected_accounts[provider] = str(account)
        metadata.native_context_boundaries[provider] = (
            max(boundary, metadata.native_context_boundaries.get(provider, 0) + 1)
            if switched
            else boundary
        )
        if switched:
            metadata.account_switch_sequences[provider] = (
                metadata.account_switch_sequences.get(provider, 0) + 1
            )
        return cls._touch(metadata)

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
