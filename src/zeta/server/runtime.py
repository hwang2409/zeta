"""Server-owned session lifecycle around the shared runtime composer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.approval import ApprovalPolicy
from ..core.project_context import (
    ProjectContext,
    discover_repo_root,
    load_project_context,
)
from ..core.session import OpenedSession, SessionManager, SessionMetadata
from ..loop import AgentLoop
from ..runtime import RuntimeComposition, compose_runtime
from ..settings import load_settings
from ..settings import resolve as resolve_settings
from ..types import CompletionBackend, StreamEvent
from .fake_backend import ServerFakeBackend

BackendFactory = Callable[[str, str | None, Path], tuple[CompletionBackend, str]]
SessionEventSink = Callable[[str, StreamEvent], None]


def default_backend(
    provider: str,
    model: str | None,
    home: Path,
    *,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
    require_credentials: bool = False,
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
        require_credentials=require_credentials,
    )


@dataclass(slots=True)
class SessionState:
    """Own the active session identity, metadata, usage, and status."""

    opened: OpenedSession
    loop: AgentLoop
    policy: ApprovalPolicy
    usage: dict[str, Any] = field(default_factory=dict)
    status: str = "idle"

    @classmethod
    def from_composition(cls, composition: RuntimeComposition) -> SessionState:
        return cls(
            opened=composition.opened,
            loop=composition.loop,
            policy=composition.policy,
        )

    @property
    def metadata(self) -> SessionMetadata:
        return self.opened.metadata

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def model(self) -> str:
        return self.metadata.model

    def turn_started(self) -> None:
        self.status = "running"

    def tool_started(self) -> None:
        self.status = "tool"

    def tool_finished(self) -> None:
        self.status = "running"

    def turn_finished(self) -> None:
        self.status = "idle"


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
        self._server_provider = provider
        self._server_model = model
        self._server_provider = self._config(None, None).provider
        self.backend_factory = backend_factory
        self.manager = SessionManager(self.home)
        self._state: SessionState | None = None
        self._background_event_sink: SessionEventSink | None = None

    @property
    def fake_catalog(self) -> bool:
        return self._server_provider == "fake"

    def backend_for_model(self, provider: str, model: str) -> CompletionBackend:
        config = self._config(provider, model)
        backend, _ = self._build_backend(
            provider,
            model,
            self.home,
            stall_seconds=config.stream_stall_seconds,
            stall_retries=config.stream_stall_retries,
            require_credentials=True,
        )
        return backend

    @property
    def opened(self) -> OpenedSession | None:
        return self._state.opened if self._state is not None else None

    @property
    def loop(self) -> AgentLoop | None:
        return self._state.loop if self._state is not None else None

    @property
    def policy(self) -> ApprovalPolicy | None:
        return self._state.policy if self._state is not None else None

    @property
    def provider(self) -> str | None:
        return (
            self._state.provider if self._state is not None else self._server_provider
        )

    @property
    def model(self) -> str | None:
        return self._state.model if self._state is not None else self._server_model

    @property
    def usage(self) -> dict[str, Any]:
        return self._state.usage if self._state is not None else {}

    @property
    def state(self) -> SessionState | None:
        return self._state

    @property
    def metadata(self) -> SessionMetadata:
        if self._state is None:
            raise RuntimeError("server has no active session")
        return self._state.metadata

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    def list_sessions(self) -> list[SessionMetadata]:
        return [
            session
            for session in self.manager.list_sessions()
            if (session.provider == "fake") == self.fake_catalog
        ]

    def set_background_event_sink(self, sink: SessionEventSink | None) -> None:
        """Attach the current frontend to child-agent progress events."""

        self._background_event_sink = sink
        if self._state is not None:
            self._bind_background_event_sink(self._state)

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
        metadata = self.manager.read_metadata(session_id)
        if (metadata.provider == "fake") != self.fake_catalog:
            if metadata.provider == "fake":
                raise ValueError(
                    "session uses the offline test provider; open it with --provider fake"
                )
            raise ValueError(
                f"session uses a real provider; open it with --provider {metadata.provider}"
            )
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
        state = self._state
        if state is not None:
            await state.loop.close()
            state.opened.store.close()
            self._state = None

    async def _replace(self, composition: RuntimeComposition) -> None:
        old_state = self._state
        if old_state is not None:
            await old_state.loop.close()
            old_state.opened.store.close()
        state = SessionState.from_composition(composition)
        self._state = state
        self._bind_background_event_sink(state)
        await state.loop.activate()

    def _bind_background_event_sink(self, state: SessionState) -> None:
        sink = self._background_event_sink
        if sink is None:
            state.loop.set_background_event_sink(None)
            return
        session_id = state.session_id
        state.loop.set_background_event_sink(lambda event: sink(session_id, event))

    def _compose(self, **kwargs: object) -> RuntimeComposition:
        return compose_runtime(
            home=self.home,
            cwd=self.cwd,
            manager=self.manager,
            backend_builder=self._build_backend,
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
        require_credentials: bool = False,
    ) -> tuple[CompletionBackend, str]:
        if self.backend_factory is not None:
            return self.backend_factory(provider, model, home)
        return default_backend(
            provider,
            model,
            home,
            stall_seconds=stall_seconds,
            stall_retries=stall_retries,
            require_credentials=require_credentials,
        )

    def _config(self, provider: str | None, model: str | None):
        project_dir = discover_repo_root(self.cwd) / ".zeta"
        settings = load_settings(home=self.home, project_dir=project_dir)
        return resolve_settings(
            settings.settings,
            cli_provider=provider if provider is not None else self._server_provider,
            cli_model=model if model is not None else self._server_model,
            cli_yolo=None,
            cli_token_budget=None,
        )


__all__ = ["BackendFactory", "ServerRuntime", "SessionState", "default_backend"]
