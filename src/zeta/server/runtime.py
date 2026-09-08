"""Server-owned session lifecycle around the shared runtime composer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.project_context import (
    ProjectContext,
    discover_repo_root,
    load_project_context,
)
from ..core.session import OpenedSession, SessionManager, SessionMetadata
from ..settings import load_settings
from ..settings import resolve as resolve_settings
from ..submission import RuntimeComposition, compose_runtime
from ..types import CompletionBackend, StreamEvent
from .fake_backend import ServerFakeBackend

BackendFactory = Callable[
    [str, str | None, Path], tuple[CompletionBackend, str]
]


def default_backend(
    provider: str,
    model: str | None,
    home: Path,
    *,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
) -> tuple[CompletionBackend, str]:
    if provider == "fake":
        selected = model or "offline"
        return ServerFakeBackend(model=selected), selected
    from ..providers.factory import build_backend

    return build_backend(
        provider,
        model,
        home=home,
        stall_seconds=stall_seconds,
        stall_retries=stall_retries,
    )


class ServerRuntime:
    """Own the current session and its provider-neutral agent loop."""

    def __init__(
        self,
        home: str | Path,
        *,
        cwd: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.home = Path(home).expanduser().resolve()
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        self.provider = provider
        self.model = model
        self.backend_factory = backend_factory
        self.manager = SessionManager(self.home)
        self.opened: OpenedSession | None = None
        self.loop = None
        self.policy = None
        self.usage: dict[str, Any] = {}
        self._background_event_sink: Callable[[StreamEvent], None] | None = None

    @property
    def metadata(self) -> SessionMetadata:
        if self.opened is None:
            raise RuntimeError("server has no active session")
        return self.opened.metadata

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    def list_sessions(self) -> list[SessionMetadata]:
        return self.manager.list_sessions()

    def set_background_event_sink(
        self, sink: Callable[[StreamEvent], None] | None
    ) -> None:
        """Attach the current frontend to child-agent progress events."""

        self._background_event_sink = sink
        if self.loop is not None:
            self.loop.set_background_event_sink(sink)

    async def create_session(
        self, *, provider: str | None = None, model: str | None = None
    ) -> SessionMetadata:
        config = self._config(provider, model)
        context = load_project_context(
            cwd=self.cwd,
            repo_root=discover_repo_root(self.cwd),
            zeta_home=self.home,
        )
        composition = self._compose(
            config=config,
            provider=config.provider,
            model=config.model,
            project_context=context,
        )
        await self._replace(composition)
        return self.metadata

    async def resume_session(self, session_id: str) -> SessionMetadata:
        opened = self.manager.open(session_id)
        context = ProjectContext(
            opened.metadata.system_prompt,
            tuple(Path(path) for path in opened.metadata.context_files),
        )
        config = self._config(opened.metadata.provider, opened.metadata.model)
        composition = self._compose(
            config=config,
            provider=opened.metadata.provider,
            model=opened.metadata.model,
            project_context=context,
            opened=opened,
        )
        await self._replace(composition)
        return self.metadata

    async def close(self) -> None:
        if self.loop is not None:
            await self.loop.close()
        self.loop = None
        self.policy = None

    async def _replace(self, composition: RuntimeComposition) -> None:
        old_loop = self.loop
        self.opened = composition.opened
        self.provider = composition.provider
        self.model = composition.model
        self.loop = composition.loop
        self.policy = composition.policy
        self.usage = {}
        if old_loop is not None:
            await old_loop.close()
        await self.loop.activate()

    def _compose(self, **kwargs: object) -> RuntimeComposition:
        return compose_runtime(
            home=self.home,
            cwd=self.cwd,
            manager=self.manager,
            backend_builder=self._build_backend,
            background_event_sink=self._background_event_sink,
            **kwargs,
        )

    def _build_backend(
        self,
        provider: str,
        model: str | None,
        home: Path,
        *,
        stall_seconds: float | None = None,
        stall_retries: int | None = None,
    ) -> tuple[CompletionBackend, str]:
        if self.backend_factory is not None:
            return self.backend_factory(provider, model, home)
        return default_backend(
            provider,
            model,
            home,
            stall_seconds=stall_seconds,
            stall_retries=stall_retries,
        )

    def _config(self, provider: str | None, model: str | None):
        project_dir = discover_repo_root(self.cwd) / ".zeta"
        settings = load_settings(home=self.home, project_dir=project_dir)
        return resolve_settings(
            settings.settings,
            cli_provider=provider or self.provider,
            cli_model=model or self.model,
            cli_yolo=None,
            cli_token_budget=None,
        )


__all__ = ["BackendFactory", "ServerRuntime", "default_backend"]
