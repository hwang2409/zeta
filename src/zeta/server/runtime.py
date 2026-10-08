"""Server-owned session lifecycle around the shared runtime composer."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.settings import load_settings
from ..config.settings import resolve as resolve_settings
from ..core.approval import ApprovalPolicy
from ..core.project_context import discover_repo_root, load_project_context
from ..core.session import OpenedSession, SessionManager, SessionMetadata
from ..core.slash import effective_budget_for_model, resolve_session_budget
from ..models.catalog import REMOVED_PROVIDER_ERROR
from ..project_registry import ProjectRegistryError
from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from ..runtime import RuntimeComposition, compose_runtime
from ..runtime.cleanup import close_session
from ..runtime.loop import AgentLoop
from ..runtime.prompt_resume import resume_prompt
from ..skills import discover_session_skills
from ..skills.agent_catalog import discover_session_agents

BackendFactory = Callable[[str, str | None, Path], tuple[CompletionBackend, str]]
SessionEventSink = Callable[[str, StreamEvent], None]
SessionWakeSink = Callable[[str], None]
MemoryNoticeSink = Callable[[str, str], None]
TEST_SCRIPTED_PROVIDER_ENV = "ZETA_TEST_SCRIPTED_PROVIDER"


class _TestScriptedBackend(CompletionBackend):
    """Minimal serve adapter enabled only by the external test-harness hook."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        prompt = next(
            (
                "".join(
                    block.text
                    for block in message.content
                    if isinstance(block, TextContent)
                )
                for message in reversed(messages)
                if message.role is MessageRole.USER
            ),
            "",
        )
        text = f"you said: {prompt}"
        yield StreamEvent(StreamEventType.MESSAGE_START)
        await asyncio.sleep(0.1)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=text)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(text)]),
            data={"usage": {"input_tokens": len(prompt), "output_tokens": len(text)}},
        )


def default_backend(
    provider: str,
    model: str | None,
    home: Path,
    *,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
    require_credentials: bool = False,
    ollama_base_url: str | None = None,
    token_budget: int | None = None,
) -> tuple[CompletionBackend, str]:
    if provider == "fake":
        raise ValueError(REMOVED_PROVIDER_ERROR)
    from ..providers.factory import build_backend

    kwargs: dict[str, object] = {
        "home": home,
        "stall_seconds": stall_seconds,
        "stall_retries": stall_retries,
        "require_credentials": require_credentials,
        "token_budget": token_budget,
    }
    if ollama_base_url is not None:
        kwargs["ollama_base_url"] = ollama_base_url
    return build_backend(provider, model, **kwargs)


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
        compaction: str | None = None,
        tools: str | None = None,
        disallowed_tools: str | None = None,
        require_tools: bool = False,
        allow_hooks: bool | None = None,
        auto_memory: bool | None = None,
        cli_yolo: bool | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.home = Path(home).expanduser().resolve()
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        self._server_provider = provider
        self._server_model = model
        self._server_compaction = compaction
        self._server_tools = tools
        self._server_disallowed_tools = disallowed_tools
        self._require_tools = require_tools
        self._allow_hooks = allow_hooks
        self._auto_memory = auto_memory
        self._cli_yolo = cli_yolo
        self._server_provider = self._config(None, None).provider
        self._test_scripted_provider = os.environ.get(TEST_SCRIPTED_PROVIDER_ENV) == "1"
        self.backend_factory = backend_factory
        self.manager = SessionManager(self.home)
        self._state: SessionState | None = None
        self._background_event_sink: SessionEventSink | None = None
        self._background_wake_sink: SessionWakeSink | None = None
        self._memory_notice_sink: MemoryNoticeSink | None = None
        self._post_stream_provider_retry = False

    def backend_for_model(
        self, provider: str, model: str, *, token_budget: int | None = None
    ) -> CompletionBackend:
        config = self._config(provider, model)
        if token_budget is None:
            stored_budget = self.metadata.compaction_budget if self._state else 0
            stored_pin = self.metadata.budget_pinned if self._state else False
            token_budget, _ = resolve_session_budget(
                stored_budget,
                stored_pin,
                provider,
                model,
                config.token_budget,
            )
        token_budget = effective_budget_for_model(provider, model, token_budget)
        backend, _ = self._build_backend(
            provider,
            model,
            self.home,
            stall_seconds=config.stream_stall_seconds,
            stall_retries=config.stream_stall_retries,
            require_credentials=True,
            token_budget=token_budget,
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
        return self.manager.list_sessions()

    def list_sessions_read_only(self) -> list[SessionMetadata]:
        return self.manager.list_sessions_read_only()

    def set_post_stream_provider_retry(self, enabled: bool) -> None:
        """Apply the serve client's negotiated retry display capability."""

        self._post_stream_provider_retry = enabled
        if self._state is not None:
            self._state.loop.post_stream_provider_retry = enabled

    def set_background_event_sink(self, sink: SessionEventSink | None) -> None:
        """Attach the current frontend to child-agent progress events."""

        self._background_event_sink = sink
        if self._state is not None:
            self._bind_background_event_sink(self._state)

    def set_memory_notice_sink(self, sink: MemoryNoticeSink | None) -> None:
        """Attach the frontend callback for automatic memory updates."""
        self._memory_notice_sink = sink
        if self._state is not None:
            self._bind_background_event_sink(self._state)

    def set_background_wake_sink(self, sink: SessionWakeSink | None) -> None:
        """Attach the frontend callback for idle child-agent wakeups."""

        self._background_wake_sink = sink
        if self._state is not None:
            self._bind_background_event_sink(self._state)

    async def create_session(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        cwd: Path | None = None,
    ) -> SessionMetadata:
        """Create a session in ``cwd``, or in the server launch directory.

        A per-session ``cwd`` gets the same treatment as ``zeta serve --cwd``:
        its repository supplies the restriction-only project settings layer,
        context files, skills, agents, and project association.
        """

        session_cwd = cwd if cwd is not None else self.cwd
        config = self._config(provider, model, cwd=session_cwd)
        repo_root = discover_repo_root(session_cwd)
        skill_catalog = discover_session_skills(home=self.home, project_dir=repo_root)
        agent_catalog = discover_session_agents(home=self.home, project_dir=repo_root)
        context = load_project_context(
            cwd=session_cwd,
            repo_root=repo_root,
            zeta_home=self.home,
            catalog=skill_catalog,
            inbox_enabled=config.inbox_enabled,
        )
        composition = self._compose(
            cwd=session_cwd,
            config=config,
            provider=config.provider,
            model=config.model,
            project_context=context,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
        )
        await self._replace(composition)
        return self.metadata

    async def resume_session(self, session_id: str) -> SessionMetadata:
        if self._state is not None and self._state.session_id == session_id:
            return self.metadata
        metadata = self.manager.read_metadata(session_id)
        if metadata.provider == "fake":
            raise ValueError(REMOVED_PROVIDER_ERROR)
        session_cwd = Path(metadata.cwd)
        if not session_cwd.is_dir():
            raise ValueError(
                f"session working directory no longer exists: {metadata.cwd}"
            )
        opened = await asyncio.to_thread(self.manager.open, session_id)
        try:
            repo_root = discover_repo_root(session_cwd)
            config = self._config(opened.metadata.provider, opened.metadata.model)
            resumed_prompt = resume_prompt(
                opened.metadata,
                manager=self.manager,
                store=opened.store,
                home=self.home,
                repo_root=repo_root,
                inbox_enabled=config.inbox_enabled,
            )
            context = resumed_prompt.context
            skill_catalog = resumed_prompt.skill_catalog
            agent_catalog = resumed_prompt.agent_catalog
            composition = self._compose(
                cwd=session_cwd,
                config=config,
                provider=opened.metadata.provider,
                model=opened.metadata.model,
                project_context=context,
                skill_catalog=skill_catalog,
                agent_catalog=agent_catalog,
                opened=opened,
            )
        except BaseException:
            opened.store.close()
            raise
        await self._replace(composition)
        return self.metadata

    @staticmethod
    async def _close_state(state: SessionState) -> None:
        await close_session(state.loop)

    async def close(self) -> None:
        state, self._state = self._state, None
        if state is not None:
            await self._close_state(state)

    async def _replace(self, composition: RuntimeComposition) -> None:
        state = SessionState.from_composition(composition)
        try:
            await self.close()
            self._state = state
            self._bind_background_event_sink(state)
            await state.loop.activate()
            if self._require_tools:
                await state.loop.ensure_mcp_servers()
                state.loop.require_allowed_tools()
        except BaseException:
            self._state = None
            await self._close_state(state)
            raise

    def _bind_background_event_sink(self, state: SessionState) -> None:
        sink = self._background_event_sink
        wake_sink = self._background_wake_sink
        if sink is None:
            state.loop.set_background_event_sink(None)
        else:
            session_id = state.session_id
            state.loop.set_background_event_sink(lambda event: sink(session_id, event))
        reconciler = state.loop.memory_reconciler
        if reconciler is not None:
            memory_sink = self._memory_notice_sink
            reconciler.notice = (
                None
                if memory_sink is None
                else lambda message: memory_sink(state.session_id, message)
            )
        if wake_sink is None:
            state.loop.set_background_wake_callback(None)
        else:
            session_id = state.session_id
            state.loop.set_background_wake_callback(lambda: wake_sink(session_id))

    def _compose(self, *, cwd: Path, **kwargs: object) -> RuntimeComposition:
        try:
            project = self.manager.project_registry.find_for_directory(cwd)
            project_id = project.project_id if project is not None else None
        except (ProjectRegistryError, OSError, ValueError) as exc:
            logging.getLogger(__name__).warning(
                "project discovery unavailable; continuing without project: %s", exc
            )
            project_id = None
        return compose_runtime(
            home=self.home,
            cwd=cwd,
            manager=self.manager,
            backend_builder=self._build_backend,
            auto_project=False,
            project_id=project_id,
            post_stream_provider_retry=self._post_stream_provider_retry,
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
        ollama_base_url: str | None = None,
        token_budget: int | None = None,
    ) -> tuple[CompletionBackend, str]:
        if self.backend_factory is not None:
            return self.backend_factory(provider, model, home)
        if self._test_scripted_provider:
            from ..models.catalog import default_model

            selected = model or default_model(provider)
            if selected is None:
                raise ValueError(f"no default model for provider {provider!r}")
            return _TestScriptedBackend(selected), selected
        kwargs: dict[str, object] = {
            "stall_seconds": stall_seconds,
            "stall_retries": stall_retries,
            "require_credentials": require_credentials,
            "token_budget": token_budget,
        }
        if ollama_base_url is not None:
            kwargs["ollama_base_url"] = ollama_base_url
        return default_backend(provider, model, home, **kwargs)

    def _config(
        self, provider: str | None, model: str | None, *, cwd: Path | None = None
    ):
        # Resume keeps the launch directory's settings layer, as an explicit
        # CLI ``--resume`` uses the invocation directory's settings.
        project_dir = discover_repo_root(cwd if cwd is not None else self.cwd) / ".zeta"
        settings = load_settings(home=self.home, project_dir=project_dir)
        return resolve_settings(
            settings.settings,
            cli_provider=provider if provider is not None else self._server_provider,
            cli_model=model if model is not None else self._server_model,
            cli_yolo=self._cli_yolo,
            cli_token_budget=None,
            cli_compaction=self._server_compaction,
            cli_tools=self._server_tools,
            cli_disallowed_tools=self._server_disallowed_tools,
            cli_allow_hooks=self._allow_hooks,
            cli_auto_memory=self._auto_memory,
        )


__all__ = ["BackendFactory", "ServerRuntime", "SessionState", "default_backend"]
