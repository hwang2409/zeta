"""Restricted composition for sessions without a human approval interface."""

from pathlib import Path

from ..accounts import SessionAccountResolver, account_provider
from ..config.settings import load_settings
from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.session import OpenedSession, SessionManager
from ..media.image_policy import image_policy_for_provider
from ..protocol.types import CompletionBackend
from ..providers.factory import build_backend
from ..skills import SkillCatalog
from ..skills.agent_catalog import discover_packaged_agents
from ..tools import ToolRegistry
from .loop import AgentLoop


def build_unattended_loop(
    session: OpenedSession,
    *,
    home: Path,
    allow: tuple[str, ...],
    backend: CompletionBackend | None = None,
) -> AgentLoop:
    metadata, store = session.metadata, session.store
    manager = SessionManager(home)
    resolver = SessionAccountResolver.load(home, manager, metadata)
    selected_provider = account_provider(metadata.provider)
    account = resolver.resolve(metadata.provider) if selected_provider is not None else None
    if selected_provider not in metadata.selected_accounts:
        account = None
    selected_accounts = resolver.snapshot()
    if metadata.provider == "ollama":
        raise ValueError("Ollama automations are not yet supported")
    if backend is None:
        backend, _model = build_backend(
            metadata.provider,
            metadata.model,
            home=home,
            token_budget=metadata.compaction_budget,
            account=account,
        )
    policy = ApprovalPolicy(
        store=store, default=ApprovalDecision.DENY, always_allow=allow
    )
    if session.metadata.skill_catalog is None:
        raise ValueError("unattended sessions require a skill catalog")
    skill_catalog = SkillCatalog.from_snapshot(session.metadata.skill_catalog)
    # Automation sessions never mount user-defined agents, even if metadata
    # was modified outside the restricted runner.
    agent_catalog = discover_packaged_agents()
    registry = ToolRegistry(
        metadata.cwd,
        session_store=store,
        approval_store=store,
        approval_policy=policy,
        enforce_approvals=True,
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
        tool_allow=metadata.tool_allow,
        tool_deny=metadata.tool_deny,
        tool_allow_layers=metadata.tool_allow_layers,
        image_policy=image_policy_for_provider(metadata.provider),
        command_niceness=load_settings(home=home).settings.command_niceness,
        selected_accounts=selected_accounts,
        account_home=home,
        account_resolver=resolver.resolve,
    )
    loop = AgentLoop(
        backend,
        store,
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
        registry=registry,
        approval_policy=policy,
        max_turns=25,
        skip_mcp_mount=True,
        # Approved automation revisions keep their persisted skill snapshot.
        freeze_skill_catalog=True,
        system_prompt=metadata.system_prompt,
        on_completion_success=lambda: manager.touch(metadata),
    )
    loop.manager = manager
    loop.session_metadata = metadata
    loop.account_resolver = resolver
    loop.account_home = home
    loop.selected_accounts = selected_accounts
    loop.account_provider = account_provider(metadata.provider)
    loop.account_switch_sequence = metadata.account_switch_sequences.get(
        loop.account_provider, 0
    )
    return loop
