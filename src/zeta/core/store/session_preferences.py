"""Persistence for per-session context preferences."""

from __future__ import annotations

from datetime import UTC, datetime
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

    @classmethod
    def _apply_session_settings(
        cls,
        metadata: Any,
        *,
        model: str,
        approval_mode: str,
        budget: int,
        provider: str,
        model_fallback: tuple[str, str, int] | None = None,
    ) -> Any:
        from ..session import SessionError

        if (
            approval_mode not in {"ask", "allow", "deny"}
            or not model.strip()
            or budget <= 0
        ):
            raise SessionError("invalid session settings")
        if metadata.model != model or metadata.provider != provider:
            metadata.override_audit.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "provider": {"from": metadata.provider, "to": provider}
                    if metadata.provider != provider
                    else None,
                    "model": {"from": metadata.model, "to": model}
                    if metadata.model != model
                    else None,
                }
            )
        metadata.provider = provider
        metadata.model = model
        metadata.approval_mode = approval_mode
        metadata.model_fallback = model_fallback
        metadata.compaction_budget = budget
        return cls._touch(metadata)

    def record_session_settings(
        self,
        metadata: Any,
        *,
        model: str,
        approval_mode: str,
        budget: int,
        provider: str,
        model_fallback: tuple[str, str, int] | None = None,
    ) -> None:
        """Persist active-session settings together, without changing global config."""
        from ..session import SessionError

        expected = (
            metadata.provider,
            metadata.model,
            metadata.approval_mode,
            metadata.compaction_budget,
            metadata.budget_pinned,
            metadata.model_fallback,
        )

        def update(item: Any) -> Any:
            if (
                item.provider,
                item.model,
                item.approval_mode,
                item.compaction_budget,
                item.budget_pinned,
                item.model_fallback,
            ) != expected:
                raise SessionError("session settings changed before commit")
            return self._apply_session_settings(
                item,
                model=model,
                approval_mode=approval_mode,
                budget=budget,
                provider=provider,
                model_fallback=model_fallback,
            )

        self._copy_metadata(metadata, self._mutate(metadata.session_id, update))

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
