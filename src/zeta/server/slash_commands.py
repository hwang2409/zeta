"""Slash command surface for zeta serve.

Reuses the shared ``SlashCommandRegistry`` so the server never forks a second
slash implementation. A minimal ``ServerSlashSession`` shim satisfies the
``SlashSession`` protocol; commands that need a client-side surface report
themselves as such rather than half-executing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..accounts.store import rename_account_command
from ..accounts.usage import format_usage_lines
from ..core.project_context import discover_repo_root
from ..core.session import persist_metadata_change
from ..core.slash import (
    MODEL_CONTEXT_WINDOWS,
    SlashCommand,
    SlashCommandRegistry,
    SlashStatus,
    compaction_history,
    create_slash_registry,
)
from ..core.todo import todo_count_tuple
from ..mcp.prompt_commands import SlashPromptError
from ..memory.user_authorization import MemoryMutationAuthorization
from ..model_input import ModelInputEnvelope
from ..project_inbox import InboxError, ProjectInbox
from ..project_memory_commands import run_memory_command
from . import accounts, ergonomics
from .model_selection import apply as apply_settings
from .protocol import ProtocolError
from .runtime import ServerRuntime

_UNAVAILABLE_NOTICE = "unavailable over the serve protocol"
_UNAVAILABLE_REASON = "requires_tui"

# This allowlist is the capability seam shared by slash_list and slash_run.
# A new shared builtin stays unavailable until serve has an executor for it.
EXECUTABLE_BUILTINS: frozenset[str] = frozenset(
    {
        "accounts",
        "compact",
        "help",
        "inbox",
        "init",
        "memory",
        "mcp",
        "model",
        "project-instructions",
        "status",
        "vim",
    }
)


@dataclass(frozen=True, slots=True)
class SlashCommandInfo:
    """One command entry surfaced by ``slash_list``."""

    name: str
    description: str
    kind: str
    source: str
    client_only: bool
    unavailable: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "source": self.source,
            "client_only": self.client_only,
            "unavailable": self.unavailable,
        }


class _ApprovalProjection(Protocol):
    def pending_approval_labels(self) -> tuple[str, ...]: ...
    def has_pending_approvals(self) -> bool: ...


class ServerSlashSession:
    """Session shim over ``ServerRuntime`` for the shared slash dispatcher.

    Only methods backed by ``EXECUTABLE_BUILTINS`` live here. The catalog and
    dispatcher reject every other shared command before it reaches this shim.
    """

    def __init__(
        self,
        runtime: ServerRuntime,
        approval_control: _ApprovalProjection | None = None,
    ) -> None:
        self._runtime = runtime
        self._approval_control = approval_control

    # --- runnable server-side ---

    def slash_status(self) -> SlashStatus:
        runtime = self._runtime
        loop = runtime.loop
        opened = runtime.opened
        if loop is None or opened is None:
            raise ProtocolError(-32003, "no active session")
        assembler = loop.context_assembler
        child_usage = assembler.descendant_usage
        pending = (
            self._approval_control.pending_approval_labels()
            if self._approval_control is not None
            else ()
        )
        items = loop.store.todo_items()
        return SlashStatus(
            session_id=opened.metadata.session_id,
            provider=opened.metadata.provider,
            model=opened.metadata.model,
            retained_tail=assembler.retained_tail,
            tokens_used_this_session=assembler.tokens_used_this_session,
            tokens_in_current_context=assembler.token_count,
            compaction_marker_count=loop.store.compaction_marker_count(),
            pending_approvals=pending,
            checkpoint_count=loop.store.checkpoint_count(),
            cache_read_input_tokens=assembler.cache_read_input_tokens_this_session,
            cache_creation_input_tokens=assembler.cache_creation_input_tokens_this_session,
            uncached_input_tokens=assembler.uncached_input_tokens_this_session,
            output_tokens_this_session=assembler.output_tokens_this_session,
            child_cache_read_input_tokens=child_usage["cache_read_input_tokens"],
            child_cache_creation_input_tokens=child_usage[
                "cache_creation_input_tokens"
            ],
            child_uncached_input_tokens=child_usage["input_tokens"],
            child_output_tokens_this_session=child_usage["output_tokens"],
            context_files=tuple(opened.metadata.context_files),
            vim_mode=opened.metadata.vim_mode,
            plan_mode=opened.metadata.plan_mode,
            hooks=(),
            todo_counts=todo_count_tuple(items) if items else None,
            usage_history=(),
            usage_cost_by_model=(),
            compaction_history=compaction_history(
                loop.store.replay(), assembler.token_counter
            ),
            model_window=MODEL_CONTEXT_WINDOWS.get(opened.metadata.provider, {}).get(
                opened.metadata.model
            ),
            mcp_summary=loop.mcp_summary,
            explicit_finish=loop.explicit_finish,
            max_agent_depth=loop.max_agent_depth,
            automatic_memory_failure=(
                loop.memory_reconciler.automatic_memory_status
                if loop.memory_reconciler is not None
                else None
            ),
            transcript_index_failure=loop.transcript_index_failure(
                opened.metadata.session_id
            ),
            project_context_health=(
                loop.project_context_health.summary
                if loop.project_context_health is not None
                else None
            ),
        )

    async def slash_accounts(self, args: str) -> str:
        """List accounts or change one display label."""

        context = getattr(self._runtime.loop, "account_context", None)
        if context is None:
            return "accounts unavailable for this session"
        if args.split():
            return await asyncio.to_thread(rename_account_command, context.store, args)
        displays = await asyncio.to_thread(context.store.list_display)
        lines: list[str] = []
        for item in displays:
            lines.append(str(item) + (f"  {item.detail}" if item.detail else ""))
            lines.extend(f"  {line}" for line in format_usage_lines(item.usage))
        return "\n".join(lines) or "no accounts"

    async def slash_model(self, args: str) -> str:
        """Switch models directly, or report the current model to the client."""

        runtime = self._runtime
        requested = args.strip()
        if not requested:
            return f"model: {runtime.model} (open Settings to switch)"
        mode = runtime.policy.default.value if runtime.policy else "ask"
        try:
            await apply_settings(runtime, requested, mode)
        except ValueError as exc:
            return f"model unchanged: {exc}"
        except (OSError, RuntimeError) as exc:
            return f"model unchanged: {exc}"
        return f"model: {requested}"

    async def slash_compact(self) -> str:
        runtime = self._runtime
        loop = runtime.loop
        if loop is None:
            return f"compact {_UNAVAILABLE_NOTICE}"
        before = loop.store.compaction_marker_count()
        try:
            context = await loop.context_assembler.assemble_context(
                backend=loop.backend,
                force=True,
                bypass_eviction_hysteresis=True,
            )
        except Exception as exc:  # noqa: BLE001 - context assembler surface is broad
            return f"compact failed: {exc}"
        after = loop.store.compaction_marker_count()
        if after == before:
            return "compact: nothing to compact"
        marker = next(
            entry
            for entry in reversed(loop.store.replay())
            if entry.type == "compaction"
        )
        return (
            "compacted entries "
            f"{marker.data['source_seq_start']}–{marker.data['source_seq_end']}; "
            f"tokens after: {context.token_count}"
        )

    async def slash_vim(self, args: str) -> str:
        runtime = self._runtime
        requested = args.strip().lower()
        if requested not in {"", "on", "off", "toggle"}:
            return "vim mode unchanged: use /vim on, /vim off, or /vim toggle"
        current = runtime.metadata.vim_mode
        if not requested:
            return f"vim mode: {'on' if current else 'off'}"
        enabled = not current if requested == "toggle" else requested == "on"
        if enabled != current:
            await persist_metadata_change(
                runtime.manager,
                runtime.metadata,
                lambda candidate: runtime.manager.record_vim_mode(
                    candidate, enabled=enabled
                ),
            )
        return f"vim mode: {'on' if enabled else 'off'}"

    def slash_memory(self, args: str) -> str:
        runtime = self._runtime
        project_id = runtime.metadata.project_id
        if project_id is None:
            return "memory: no associated project"
        registry = runtime.manager.project_registry
        if registry is None:
            return "memory: project registry unavailable"
        loop = getattr(runtime, "loop", None)
        reconciler = loop.memory_reconciler if loop is not None else None
        return run_memory_command(
            registry,
            project_id,
            args,
            reconciler,
            MemoryMutationAuthorization.direct_slash(),
        )

    async def slash_mcp(self, args: str) -> str | ModelInputEnvelope:
        if self._runtime.loop is None:
            raise RuntimeError(_UNAVAILABLE_NOTICE)
        return await self._runtime.loop.slash_mcp(args)

    def slash_inbox(self, args: str) -> str:
        if args.strip():
            return "usage: /inbox"
        runtime = self._runtime
        if runtime.loop is None or runtime.opened is None:
            return "project inbox unavailable: no active session"
        registry = runtime.loop.project_registry
        project_id = runtime.opened.metadata.project_id
        if registry is None or project_id is None:
            return "project inbox unavailable: session is not associated with a project"
        try:
            state = ProjectInbox(
                registry, sessions_root=registry.root.parent / "sessions"
            ).list(project_id)
        except (InboxError, OSError) as exc:
            return f"project inbox unavailable: {exc}"
        return json.dumps(state, indent=2, sort_keys=True)


def build_registry(runtime: ServerRuntime) -> SlashCommandRegistry:
    """Build a per-request registry so custom commands and skills stay current."""

    opened = runtime.opened
    if opened is None:
        raise ProtocolError(-32003, "no active session")
    loop = runtime.loop
    if loop is None:
        raise ProtocolError(-32003, "no active session")
    catalog = loop.tool_registry.skill_catalog
    project_dir = discover_repo_root(Path(opened.metadata.cwd or runtime.cwd))
    return create_slash_registry(
        zeta_home=runtime.home,
        project_dir=project_dir,
        skill_catalog=catalog,
        project_registry_state=runtime.manager.project_registry_state,
        extra_commands=(
            SlashCommand(
                "accounts",
                lambda session, args: session.slash_accounts(args),
                "list or label accounts",
            ),
        ),
    )


READ_REQUESTS = {"slash_list", *accounts.REQUEST_METHODS}


async def read_request(
    method: str,
    runtime: ServerRuntime,
    require_idle: Callable[[], Awaitable[None]],
    params: dict[str, object],
    owner: object,
) -> dict[str, object]:
    if method == "slash_list":
        include_unavailable = params.get("include_unavailable", False)
        if not isinstance(include_unavailable, bool):
            raise ProtocolError(-32602, "include_unavailable must be a boolean")
        return await list_commands(runtime, include_unavailable=include_unavailable)
    operation = accounts.request(method, runtime, require_idle, params)
    if method == "account_switch":
        return await accounts.run_owned_switch(owner, operation)
    return await operation


async def list_commands(
    runtime: ServerRuntime, *, include_unavailable: bool = False
) -> dict[str, object]:
    """Enumerate runnable commands, with an optional complete catalog."""

    if runtime.loop is not None:
        await runtime.loop.refresh_skills()
    registry = await asyncio.to_thread(build_registry, runtime)
    entries = _entries_from_registry(registry, include_unavailable=include_unavailable)
    return {
        "commands": [entry.to_dict() for entry in entries],
        "notices": list(registry.notices),
    }


async def run_command(
    runtime: ServerRuntime,
    text: str,
    *,
    register_model_input: Callable[[ModelInputEnvelope], str] | None = None,
    approval_control: _ApprovalProjection | None = None,
) -> dict[str, object]:
    """Dispatch one slash invocation through the shared registry."""

    if not text.startswith("/") or text.startswith("//"):
        raise ProtocolError(-32602, "text must start with '/'")
    first_line = text.split("\n", 1)[0]
    parts = first_line[1:].split(maxsplit=1)
    if not parts or not parts[0]:
        raise ProtocolError(-32602, "text must name a command")
    name = parts[0]
    tail = parts[1] if len(parts) == 2 else ""
    if (
        (name == "vim" and tail.strip().lower() in {"on", "off", "toggle"})
        or name == "compact"
        or (name == "model" and tail.strip())
        or (name == "accounts" and tail.strip())
    ):
        # Guard mutation-capable dispatch on the same seam session-mutation
        # RPCs use. Without this, `/compact` or `/model <name>` could edit
        # the store or settings while background children are still running
        # or an approval is outstanding. Read-only calls (`/status`,
        # `/help`, argless `/model`) stay guard-free.
        ergonomics.require_mutable(
            runtime,
            approval_control is not None and approval_control.has_pending_approvals(),
        )
    if runtime.loop is not None:
        await runtime.loop.refresh_skills()
    registry = await asyncio.to_thread(build_registry, runtime)
    builtin_names = {command.name for command in registry.builtin_commands}
    if name in builtin_names and name not in EXECUTABLE_BUILTINS:
        return {"kind": "client_only", "name": name}
    if name == "mcp":
        return await _run_mcp_command(runtime, tail)
    # Prompt macros expand to user input rather than dispatch through a handler.
    # Inline shell spans need the client's approval loop; report those as
    # client-only rather than resolving with unapproved output here.
    custom = registry.custom_command_for(name)
    if custom is not None:
        if custom.kind != "prompt" or registry.needs_inline_shell_resolution(text):
            return {"kind": "client_only", "name": name}
        envelope = registry.input_for_model(text)
        return _model_input_result(envelope, register_model_input)
    if name in {entry[0] for entry in registry.mcp_prompt_entries}:
        return {"kind": "client_only", "name": name}
    session = ServerSlashSession(runtime, approval_control)
    result = await registry.dispatch_async(session, text)
    if result is None:
        return {"kind": "unknown", "name": name}
    if isinstance(result, SlashPromptError):
        return {"kind": "error", "text": result.message}
    if isinstance(result, ModelInputEnvelope):
        return _model_input_result(result, register_model_input)
    if isinstance(result, str):
        return {"kind": "output", "text": result}
    raise ProtocolError(-32000, "unexpected slash result type")


async def _run_mcp_command(runtime: ServerRuntime, tail: str) -> dict[str, object]:
    stripped = tail.strip()
    verb = stripped.partition(" ")[0]
    if verb not in {"status", "auth", "logout"}:
        return {"kind": "client_only", "name": "mcp"}
    if runtime.loop is None:
        raise ProtocolError(-32003, "no active session")
    result = await runtime.loop.slash_mcp(stripped)
    response: dict[str, object] = {
        "kind": "text" if stripped == "status" else "output",
        "text": result,
    }
    credential_error = getattr(result, "credential_store_error", None)
    if credential_error is not None:
        response["credential_store_error"] = credential_error
    return response


def _model_input_result(
    envelope: ModelInputEnvelope,
    register: Callable[[ModelInputEnvelope], str] | None,
) -> dict[str, object]:
    if register is None:
        return {
            "kind": "model_input",
            "text": envelope.text,
            "display_text": envelope.display_text,
            "origin": envelope.origin.value,
        }
    return {
        "kind": "model_input",
        "input_id": register(envelope),
        "display_text": envelope.display_text,
    }


def _entries_from_registry(
    registry: SlashCommandRegistry, *, include_unavailable: bool
) -> list[SlashCommandInfo]:
    """Project the shared registry through serve's executable capabilities."""

    entries: list[SlashCommandInfo] = []
    for command in registry.builtin_commands:
        unavailable = (
            None if command.name in EXECUTABLE_BUILTINS else _UNAVAILABLE_REASON
        )
        if unavailable is None or include_unavailable:
            entries.append(
                SlashCommandInfo(
                    name=command.name,
                    description=command.description,
                    kind="builtin",
                    source="builtin",
                    client_only=unavailable is not None,
                    unavailable=unavailable,
                )
            )
    for command in registry.custom_commands:
        unavailable = (
            None
            if command.kind == "prompt"
            and not registry.needs_inline_shell_resolution(f"/{command.name}")
            else _UNAVAILABLE_REASON
        )
        if unavailable is None or include_unavailable:
            entries.append(
                SlashCommandInfo(
                    name=command.name,
                    description=command.description,
                    kind=f"macro-{command.kind}",
                    source=command.source,
                    client_only=unavailable is not None,
                    unavailable=unavailable,
                )
            )
    for skill in registry.skill_entries:
        entries.append(
            SlashCommandInfo(
                name=skill.name,
                description=skill.description,
                kind="skill",
                source=skill.source,
                client_only=False,
                unavailable=None,
            )
        )
    if include_unavailable:
        for name, description, source in registry.mcp_prompt_entries:
            entries.append(
                SlashCommandInfo(
                    name=name,
                    description=description,
                    kind="mcp-prompt",
                    source=f"mcp:{source}",
                    client_only=True,
                    unavailable=_UNAVAILABLE_REASON,
                )
            )
    return entries


__all__ = [
    "EXECUTABLE_BUILTINS",
    "ServerSlashSession",
    "SlashCommandInfo",
    "build_registry",
    "list_commands",
    "run_command",
]
