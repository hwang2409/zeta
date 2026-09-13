"""Restricted composition for sessions without a human approval interface."""

from pathlib import Path

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.session import OpenedSession, SessionManager
from ..loop import AgentLoop
from ..providers.factory import build_backend
from ..tools import ToolRegistry
from ..types import CompletionBackend


def build_unattended_loop(
    session: OpenedSession,
    *,
    home: Path,
    allow: tuple[str, ...],
    backend: CompletionBackend | None = None,
) -> AgentLoop:
    metadata, store = session.metadata, session.store
    if backend is None:
        backend, _model = build_backend(metadata.provider, metadata.model, home=home)
    policy = ApprovalPolicy(
        store=store, default=ApprovalDecision.DENY, always_allow=allow
    )
    registry = ToolRegistry(
        metadata.cwd,
        session_store=store,
        approval_store=store,
        approval_policy=policy,
        enforce_approvals=True,
    )
    return AgentLoop(
        backend,
        store,
        registry=registry,
        approval_policy=policy,
        max_turns=25,
        skip_mcp_mount=True,
        system_prompt=metadata.system_prompt,
        on_completion_success=lambda: SessionManager(home).touch(metadata),
    )
