"""Server-owned session and loop composition."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.hooks import load_hooks_for_provider
from ..core.project_context import discover_repo_root, load_project_context
from ..core.session import OpenedSession, SessionManager, SessionMetadata
from ..loop import AgentLoop
from ..providers.factory import build_backend
from ..settings import load_settings
from ..settings import resolve as resolve_settings
from ..tools._user_discovery import apply_external_tools
from ..types import CompletionBackend
from .fake_backend import ServerFakeBackend

BackendFactory = Callable[
    [str, str | None, Path], tuple[CompletionBackend, str]
]


def default_backend(provider: str, model: str | None, home: Path) -> tuple[CompletionBackend, str]:
    if provider == "fake":
        selected = model or "offline"
        return ServerFakeBackend(model=selected), selected
    return build_backend(provider, model, home=home)


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
        self.backend_factory = backend_factory or default_backend
        self.manager = SessionManager(self.home)
        self.opened: OpenedSession | None = None
        self.loop: AgentLoop | None = None
        self.policy: ApprovalPolicy | None = None
        self.usage: dict[str, Any] = {}

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

    def create_session(self, *, provider: str | None = None, model: str | None = None) -> SessionMetadata:
        config = self._config(provider, model)
        selected_provider = config.provider
        backend, selected_model = self.backend_factory(selected_provider, config.model, self.home)
        context = load_project_context(
            cwd=self.cwd,
            repo_root=discover_repo_root(self.cwd),
            zeta_home=self.home,
        )
        budget = config.token_budget or 200_000
        self.opened = self.manager.create(
            provider=selected_provider,
            model=selected_model,
            cwd=self.cwd,
            compaction_budget=budget,
            system_prompt=context.system_prompt,
            context_files=[str(path) for path in context.files],
        )
        self.provider, self.model = selected_provider, selected_model
        self._install_loop(backend, config, context.system_prompt)
        return self.metadata

    def resume_session(self, session_id: str) -> SessionMetadata:
        opened = self.manager.open(session_id)
        backend, selected_model = self.backend_factory(
            opened.metadata.provider, opened.metadata.model, self.home
        )
        self.opened = opened
        self.provider, self.model = opened.metadata.provider, selected_model
        self._install_loop(backend, self._config(self.provider, self.model), opened.metadata.system_prompt)
        return self.metadata

    async def close(self) -> None:
        if self.loop is not None:
            await self.loop.close()
        self.loop = None
        self.policy = None

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

    def _install_loop(self, backend: CompletionBackend, config: Any, system_prompt: str) -> None:
        if self.opened is None:
            raise RuntimeError("cannot install a loop without a session")
        metadata = self.opened.metadata
        policy = ApprovalPolicy(
            store=self.opened.store,
            default=ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK,
            always_allow=config.approval_allow,
            always_deny=config.approval_deny,
            always_ask=config.approval_ask,
        )

        def completion_success() -> None:
            self.manager.touch(metadata)

        loop = AgentLoop(
            backend,
            self.opened.store,
            approval_policy=policy,
            token_budget=metadata.compaction_budget,
            retained_tail=metadata.retained_tail,
            system_prompt=system_prompt,
            on_completion_success=completion_success,
            hooks=load_hooks_for_provider(self.home, metadata.provider),
        )
        loop.set_mcp_scope(home=self.home, project_dir=discover_repo_root(Path(metadata.cwd)))
        apply_external_tools(
            loop.tool_registry,
            home=self.home,
            project_dir=discover_repo_root(Path(metadata.cwd)) / ".zeta",
        )
        loop.tool_schemas = list(loop.tool_registry.schemas)
        if metadata.plan_mode:
            loop.set_plan_mode(True)
        self.loop, self.policy = loop, policy


__all__ = ["BackendFactory", "ServerRuntime", "default_backend"]
