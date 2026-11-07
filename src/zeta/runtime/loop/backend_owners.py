"""Runtime ownership for composed completion backends."""
from __future__ import annotations

from typing import Any

from ...protocol.types import OwnedCompletionBackend, is_owned_completion_backend
from ...providers.transport import _schedule_cleanup_failure


class BackendOwnerMixin:
    _backend_owners: list[OwnedCompletionBackend]

    def register_backend_owner(self, backend: object) -> None:
        if is_owned_completion_backend(backend) and all(
            owner is not backend for owner in self._backend_owners
        ):
            self._backend_owners.append(backend)

    def connection_setup_counts(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "provider": backend.provider,
                "model": backend.model,
                "count": backend.connection_setup_count,
            }
            for backend in self._backend_owners
        )

    def _backend_close_reporter(self, backend: OwnedCompletionBackend, stage: str):
        def report(index: int, error: BaseException) -> None:
            _schedule_cleanup_failure(
                error,
                stage=f"completion_backend_{stage}",
                index=index,
                primary_exception=None,
                diagnostics_path=getattr(backend, "diagnostics_path", None),
                provider=backend.provider,
            )

        return report

    async def close_backend(
        self, backend: OwnedCompletionBackend, *, stage: str
    ) -> None:
        backend.retire()
        report = self._backend_close_reporter(backend, stage)
        outcome = await backend.aclose(report_failure=report)
        for failure in outcome.failures:
            report(failure.index, failure.error)
        self._backend_owners = [
            owner for owner in self._backend_owners if owner is not backend
        ]
