"""Slash command surface for zeta serve.

Reuses the shared ``SlashCommandRegistry`` so the server never forks a second
slash implementation. A minimal ``ServerSlashSession`` shim satisfies the
``SlashSession`` protocol; commands that need a client-side surface report
themselves as such rather than half-executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.commands.custom_commands import CustomCommand
from ..core.project_context import discover_repo_root
from ..core.slash import (
    MODEL_CONTEXT_WINDOWS,
    SlashCommandRegistry,
    SlashStatus,
    compaction_history,
    create_slash_registry,
)
from ..core.todo import todo_count_tuple
from ..mcp.prompt_commands import SlashModelInput, SlashPromptError
from ..skills import SkillCatalog
from . import ergonomics
from .model_selection import apply as apply_settings
from .protocol import ProtocolError
from .runtime import ServerRuntime

# Composed once at the model layer so the wire text is not literal in the GUI.
_CLIENT_ONLY_NOTICE = "runs client-side; open the composer on the desktop app"
_UNAVAILABLE_NOTICE = "unavailable over the serve protocol"

# Commands that need a client-side surface (picker, workspace mutation,
# transcript navigation, or a shell macro loop). They still appear in the
# list response so a GUI can render the menu; ``slash_run`` reports them as
# client-only rather than half-executing them here.
CLIENT_ONLY_BUILTINS: frozenset[str] = frozenset(
    {
        "plan",
        "implement",
        "paste",
        "checkpoint",
        "fork",
        "tree",
        "tools",
        "undo",
        "redo",
        "new",
        "name",
        "theme",
        "runs",
        "send",
        "mcp",
        "automations",
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
    unavailable: str  # empty when the command is fine

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "source": self.source,
            "client_only": self.client_only,
            "unavailable": self.unavailable or None,
        }


class ServerSlashSession:
    """Session shim over ``ServerRuntime`` for the shared slash dispatcher.

    Methods that the server can drive return real output. The rest keep the
    ``SlashSession`` structural type intact but return a client-only notice
    so the shared dispatcher never leaks half-executed state.
    """

    def __init__(self, runtime: ServerRuntime) -> None:
        self._runtime = runtime

    # --- runnable server-side ---

    def slash_status(self) -> SlashStatus:
        runtime = self._runtime
        loop = runtime.loop
        opened = runtime.opened
        if loop is None or opened is None:
            raise ProtocolError(-32003, "no active session")
        assembler = loop.context_assembler
        pending = tuple(
            f"{request.key} ({request.label or request.tool_call.name})"
            for request in (runtime.policy.pending_requests() if runtime.policy else ())
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
        )

    def slash_model(self, args: str) -> str:
        """Switch models directly, or report the current model to the client."""

        runtime = self._runtime
        requested = args.strip()
        if not requested:
            return f"model: {runtime.model} (open Settings to switch)"
        mode = runtime.policy.default.value if runtime.policy else "ask"
        try:
            apply_settings(runtime, requested, mode)
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
                backend=loop.backend, force=True
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

    # --- client-only stubs ---

    def _client_only(self, name: str) -> str:
        return f"/{name}: {_CLIENT_ONLY_NOTICE}"

    def slash_vim(self, args: str) -> str:
        runtime = self._runtime
        requested = args.strip().lower()
        if requested not in {"", "on", "off", "toggle"}:
            return "vim mode unchanged: use /vim on, /vim off, or /vim toggle"
        current = runtime.metadata.vim_mode
        if not requested:
            return f"vim mode: {'on' if current else 'off'}"
        enabled = not current if requested == "toggle" else requested == "on"
        if enabled != current:
            runtime.manager.record_vim_mode(runtime.metadata, enabled=enabled)
        return f"vim mode: {'on' if enabled else 'off'}"

    def slash_plan(self, args: str) -> str | SlashModelInput:
        del args
        return self._client_only("plan")

    def slash_implement(self, args: str) -> str | SlashModelInput:
        del args
        return self._client_only("implement")

    def slash_paste(self, args: str) -> str:
        del args
        return self._client_only("paste")

    def slash_checkpoint(self, args: str) -> str:
        del args
        return self._client_only("checkpoint")

    def slash_fork(self, args: str) -> str:
        del args
        return self._client_only("fork")

    def slash_tree(self, args: str) -> str:
        del args
        return self._client_only("tree")

    def slash_tools(self, args: str) -> str:
        del args
        return self._client_only("tools")

    def slash_undo(self, args: str) -> str:
        del args
        return self._client_only("undo")

    def slash_redo(self, args: str) -> str:
        del args
        return self._client_only("redo")

    def slash_new(self, args: str) -> str:
        del args
        return self._client_only("new")

    def slash_name(self, args: str) -> str:
        del args
        return self._client_only("name")

    def slash_theme(self, args: str) -> str:
        del args
        return self._client_only("theme")

    def slash_runs(self, args: str) -> str:
        del args
        return self._client_only("runs")

    def slash_send(self, args: str) -> str:
        del args
        return self._client_only("send")

    async def slash_mcp(self, args: str) -> str | SlashModelInput:
        del args
        return self._client_only("mcp")

    async def slash_automations(self, args: str) -> str:
        del args
        return self._client_only("automations")

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        del name, arguments
        raise RuntimeError(_UNAVAILABLE_NOTICE)

    async def slash_exec_macro(self, command: CustomCommand, args: str) -> str:
        del command, args
        return _UNAVAILABLE_NOTICE


def build_registry(runtime: ServerRuntime) -> SlashCommandRegistry:
    """Build a per-request registry so custom commands and skills stay current."""

    opened = runtime.opened
    if opened is None:
        raise ProtocolError(-32003, "no active session")
    if opened.metadata.skill_catalog is not None:
        catalog = SkillCatalog.from_snapshot(opened.metadata.skill_catalog)
    else:
        catalog = SkillCatalog.empty()
    project_dir = discover_repo_root(Path(opened.metadata.cwd or runtime.cwd))
    return create_slash_registry(
        zeta_home=runtime.home,
        project_dir=project_dir,
        skill_catalog=catalog,
    )


def list_commands(runtime: ServerRuntime) -> dict[str, object]:
    """Enumerate every command the shared registry knows about."""

    registry = build_registry(runtime)
    entries = _entries_from_registry(registry)
    return {
        "commands": [entry.to_dict() for entry in entries],
        "notices": list(registry.notices),
    }


async def run_command(runtime: ServerRuntime, text: str) -> dict[str, object]:
    """Dispatch one slash invocation through the shared registry."""

    if not text.startswith("/") or text.startswith("//"):
        raise ProtocolError(-32602, "text must start with '/'")
    first_line = text.split("\n", 1)[0]
    parts = first_line[1:].split(maxsplit=1)
    if not parts or not parts[0]:
        raise ProtocolError(-32602, "text must name a command")
    name = parts[0]
    tail = parts[1] if len(parts) == 2 else ""
    if (name == "vim" and tail.strip().lower() in {"on", "off", "toggle"}) or name == "compact" or (
        name == "model" and tail.strip()
    ):
        # Guard mutation-capable dispatch on the same seam session-mutation
        # RPCs use. Without this, `/compact` or `/model <name>` could edit
        # the store or settings while background children are still running
        # or an approval is outstanding. Read-only calls (`/status`,
        # `/help`, argless `/model`) stay guard-free.
        ergonomics.require_mutable(runtime)
    if name == "model" and not tail.strip():
        # Argless `/model` opens the client's picker surface (Settings on
        # the GUI) rather than returning a bare text notice — the doc
        # contract at docs/serve-protocol.md. `/model <name>` still
        # dispatches server-side through the shared apply-settings path.
        return {"kind": "client_only", "name": name}
    registry = build_registry(runtime)
    if name in CLIENT_ONLY_BUILTINS:
        return {"kind": "client_only", "name": name}
    # Prompt macros expand to user input rather than dispatch through a handler.
    # Inline shell spans need the client's approval loop; report those as
    # client-only rather than resolving with unapproved output here.
    custom = registry.custom_command_for(name)
    if custom is not None and custom.kind == "prompt":
        if registry.needs_inline_shell_resolution(text):
            return {"kind": "client_only", "name": name}
        return {"kind": "model_input", "text": registry.input_for_model(text)}
    session = ServerSlashSession(runtime)
    result = await registry.dispatch_async(session, text)
    if result is None:
        return {"kind": "unknown", "name": name}
    if isinstance(result, SlashPromptError):
        return {"kind": "error", "text": result.message}
    if isinstance(result, SlashModelInput):
        return {"kind": "model_input", "text": result.text}
    if isinstance(result, str):
        return {"kind": "output", "text": result}
    raise ProtocolError(-32000, "unexpected slash result type")


def _entries_from_registry(
    registry: SlashCommandRegistry,
) -> list[SlashCommandInfo]:
    """Walk the registry once into a stable, wire-safe list."""

    entries: list[SlashCommandInfo] = []
    for command in registry.builtin_commands:
        name = command.name
        client_only = name in CLIENT_ONLY_BUILTINS
        entries.append(
            SlashCommandInfo(
                name=name,
                description=command.description,
                kind="builtin",
                source="builtin",
                client_only=client_only,
                unavailable="",
            )
        )
    for command in registry.custom_commands:
        entries.append(
            SlashCommandInfo(
                name=command.name,
                description=command.description,
                kind=f"macro-{command.kind}",
                source=command.source,
                # Exec macros need the tool-approval loop the server does not
                # drive over the RPC surface; list them so the menu shows them
                # and let the GUI decide how to surface the limitation.
                client_only=command.kind == "exec",
                unavailable=(
                    "" if command.kind == "prompt" else _UNAVAILABLE_NOTICE
                ),
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
                unavailable="",
            )
        )
    for name, description, source in registry.mcp_prompt_entries:
        entries.append(
            SlashCommandInfo(
                name=name,
                description=description,
                kind="mcp-prompt",
                source=f"mcp:{source}",
                client_only=False,
                unavailable="",
            )
        )
    return entries


__all__ = [
    "CLIENT_ONLY_BUILTINS",
    "ServerSlashSession",
    "SlashCommandInfo",
    "build_registry",
    "list_commands",
    "run_command",
]
