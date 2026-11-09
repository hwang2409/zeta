"""Slash command surface for zeta serve.

Reuses the shared ``SlashCommandRegistry`` so the server never forks a second
slash implementation. A minimal ``ServerSlashSession`` shim satisfies the
``SlashSession`` protocol; commands that need a client-side surface report
themselves as such rather than half-executing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..accounts.store import rename_account_command
from ..accounts.usage import format_usage_lines
from ..attention_decisions import DecisionItem, open_decisions
from ..attention_records import Broken
from ..core.commands.plan_commands import run_implement_command, run_plan_command
from ..core.commands.slash_registry import SlashSurface
from ..core.project_context import discover_repo_root, project_command_view
from ..core.session import SessionError, persist_metadata_change
from ..core.slash import (
    SlashCommandRegistry,
    SlashStatus,
    _format_status,
    create_slash_registry,
)
from ..core.transcript_recovery.runtime import (
    repair_transcript_for_runtime,
    transcript_repair_available,
)
from ..mcp.prompt_commands import SlashPromptError
from ..memory.user_authorization import MemoryMutationAuthorization
from ..model_input import ModelInputEnvelope
from ..project_inbox import InboxError, ProjectInbox
from ..project_memory_commands import run_memory_command
from ..project_registry import ProjectRegistryError
from ..runtime.status import (
    SlashStatusSnapshot,
    build_slash_status,
    capture_slash_status,
)
from . import accounts, ergonomics
from .model_selection import apply as apply_settings
from .protocol import (
    ErrorDescriptor,
    ProtocolError,
    SlashErrorCodeDescriptor,
    SlashResultKindDescriptor,
)
from .runtime import ServerRuntime

_UNAVAILABLE_NOTICE = "unavailable over the serve protocol"
_UNAVAILABLE_REASON = "requires_tui"
_SERVE_MCP_VERBS = (
    "status",
    "auth",
    "logout",
    "add",
    "remove",
    "enable",
    "disable",
    "trust",
    "untrust",
)


@dataclass(frozen=True, slots=True)
class _ClientOnlySlashResult:
    name: str


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
    """Session adapter for shared slash commands available over serve."""

    def __init__(
        self,
        runtime: ServerRuntime,
        approval_control: _ApprovalProjection | None = None,
        run_blocking: Callable[[Callable[[], object]], Awaitable[object]] | None = None,
    ) -> None:
        self._runtime = runtime
        self._approval_control = approval_control
        self._run_blocking = run_blocking

    # --- runnable server-side ---

    async def slash_decisions(self, args: str) -> str:
        _loop, opened = self._active_loop_and_session()
        command, _, remainder = args.strip().partition(" ")
        registry_state = self._runtime.manager.project_registry_state
        if command == "batch":
            return await self._slash_decisions_batch(
                remainder, opened.metadata.session_id
            )
        if command in {"dismiss", "reply"}:
            items = await asyncio.to_thread(
                open_decisions,
                self._runtime.home,
                opened.metadata.session_id,
                registry_state,
            )
            return await self._slash_decisions_action(
                command, remainder, items, opened.metadata.session_id
            )
        await self._run_attention_operation(
            self._attention_replies(opened.metadata.session_id).recover
        )
        items = await asyncio.to_thread(
            open_decisions,
            self._runtime.home,
            opened.metadata.session_id,
            registry_state,
        )
        if not items:
            return "no open attention items"
        return "open attention items:\n" + "\n".join(
            (
                f"  {item.record.id} [broken] {item.record.path}: {item.record.reason}"
                if isinstance(item.record, Broken)
                else f"  {item.record.id} [{item.record.kind}] {item.record.title} [{item.project_name}]"
            )
            for item in items
        )

    def _active_loop_and_session(self):
        loop = self._runtime.loop
        opened = self._runtime.opened
        if loop is None or opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        return loop, opened

    def _attention_replies(self, session_id: str):
        from ..attention_replies import AttentionReplies

        loop, _opened = self._active_loop_and_session()
        on_visible = (
            loop.notify_project_inbox_publication
            if session_id == loop.store.session_id
            else None
        )
        return AttentionReplies(
            self._runtime.home,
            session_id,
            registry_state=self._runtime.manager.project_registry_state,
            on_inbox_visible=on_visible,
        )

    async def _run_attention_operation(self, operation, *args):
        return await asyncio.to_thread(operation, *args)

    async def _slash_decisions_batch(self, payload: str, from_session: str) -> str:
        if not payload:
            return "usage: /decisions batch <json array>"
        try:
            submitted = json.loads(payload)
            if not isinstance(submitted, list):
                raise TypeError
            batch = [(item["id"], item["reply"]) for item in submitted]
            if any(
                not isinstance(item_id, str) or not isinstance(reply, str)
                for item_id, reply in batch
            ) or len({item_id for item_id, _ in batch}) != len(batch):
                raise TypeError
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return "batch must be a JSON array of objects with id and reply"
        return await self._run_decisions_batch(batch, from_session)

    async def _run_decisions_batch(
        self, batch: list[tuple[str, str]], from_session: str
    ) -> str:
        outcome = await self._run_attention_operation(
            self._attention_replies(from_session).stage_and_flush,
            batch,
            from_session,
        )
        refusals = [item.detail for item in outcome.items if item.status == "broken"]
        replied = len(outcome.items) - len(refusals)
        lines = [f"replied to {replied} attention item(s)"] if replied else []
        return "\n".join([*lines, *refusals])

    async def _slash_decisions_action(
        self,
        command: str,
        remainder: str,
        items: tuple[DecisionItem, ...],
        from_session: str,
    ) -> str:
        attention_id, separator, reply = remainder.partition(" ")
        if not attention_id:
            return "usage: /decisions reply <id> <text> or /decisions dismiss <id>"
        item = next((item for item in items if item.record.id == attention_id), None)
        if item is None:
            return "attention item not found"
        if isinstance(item.record, Broken):
            return f"attention record {item.record.id} is broken ({item.record.reason}); repair {item.record.path} before replying"
        replies = self._attention_replies(item.session_id)
        if command == "dismiss":
            await self._run_attention_operation(replies.dismiss, item.record.id)
            text = f"dismissed {item.record.id}"
        else:
            if not separator or not reply.strip():
                return "reply text must be nonempty"
            await self._run_attention_operation(
                replies.reply, item.record.id, reply, from_session
            )
            text = f"replied to {item.record.id}"
        return text
    @property
    def plan_mode(self) -> bool:
        return self._runtime.loop.plan_mode if self._runtime.loop is not None else False

    @property
    def active(self) -> bool:
        return False

    @property
    def pending_approvals(self) -> tuple[str, ...]:
        if self._approval_control is None:
            return ()
        return self._approval_control.pending_approval_labels()

    def set_plan_mode(self, enabled: bool):
        if self._runtime.loop is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        return self._runtime.loop.set_plan_mode(enabled)

    def invalidate_prompt(self) -> None:
        return None

    def plan_mode_background_blocker(self) -> str | None:
        work = tuple(getattr(self._runtime.loop, "background_work_descriptions", ()))
        if not work:
            return None
        return (
            "plan mode unchanged: background work is active: "
            f"{', '.join(work)}; stop it or wait"
        )

    async def slash_plan(self, args: str) -> str | ModelInputEnvelope:
        return await run_plan_command(self, args)

    async def slash_implement(self, args: str) -> str | ModelInputEnvelope:
        return await run_implement_command(self, args)

    def slash_status_command(self, args: str) -> str | Awaitable[str]:
        del args
        if self._run_blocking is None:
            return _format_status(self.slash_status())
        snapshot = self.status_snapshot()

        async def build() -> str:
            status = await self._run_blocking(lambda: build_slash_status(snapshot))
            if not isinstance(status, SlashStatus):
                raise TypeError("status worker returned an invalid projection")
            return _format_status(status)

        return build()

    def status_snapshot(self) -> SlashStatusSnapshot:
        runtime = self._runtime
        loop = runtime.loop
        opened = runtime.opened
        if loop is None or opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        state = runtime.state
        if state is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        pending = (
            self._approval_control.pending_approval_labels()
            if self._approval_control is not None
            else ()
        )
        return capture_slash_status(
            loop,
            session_id=opened.metadata.session_id,
            provider=opened.metadata.provider,
            model=opened.metadata.model,
            context_files=tuple(opened.metadata.context_files),
            vim_mode=opened.metadata.vim_mode,
            pending_approvals=pending,
            usage_tracker=state.usage_tracker,
            project_context_health=(
                loop.project_context_health.summary
                if loop.project_context_health is not None
                else None
            ),
        )

    def slash_status(self) -> SlashStatus:
        return build_slash_status(self.status_snapshot())

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
            availability = (
                "  unavailable: credentials missing" if not item.available else ""
            )
            lines.append(
                str(item) + (f"  {item.detail}" if item.detail else "") + availability
            )
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
        state = runtime.state
        if state is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        state.usage_tracker.record_compaction(state.model)
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

    async def slash_project(self, args: str) -> str:
        runtime = self._runtime
        opened = runtime.opened
        registry = runtime.manager.project_registry
        if opened is None:
            return "project registry unavailable: no active session"
        if registry is None:
            return "project registry unavailable"
        initialize = args.strip() == "init"
        try:
            view = await asyncio.to_thread(
                project_command_view,
                registry,
                opened.store.cwd,
                runtime.metadata.project_id,
                initialize=initialize,
            )
        except (ProjectRegistryError, OSError, SessionError) as exc:
            if initialize:
                return f"could not initialize project: {exc}"
            raise
        if initialize:
            assert view.project is not None
            await persist_metadata_change(
                runtime.manager,
                runtime.metadata,
                lambda candidate: runtime.manager.associate_project(
                    candidate, view.project.project_id
                ),
            )
            if runtime.loop is not None:
                runtime.loop._set_runtime_project(runtime.metadata)
        return view.text

    async def slash_repair_transcript(self, args: str) -> str:
        parts = args.split()
        if len(parts) not in {2, 3} or parts[-1] != "confirm":
            return "usage: /repair-transcript <session-id> [<child-run-id>] confirm"
        child_run_id = parts[1] if len(parts) == 3 else None
        result = await repair_transcript_for_runtime(
            self._runtime.manager,
            parts[0],
            child_run_id,
            parent_store=(
                self._runtime.opened.store if self._runtime.opened is not None else None
            ),
            memory_reconciler=(
                self._runtime.loop.memory_reconciler
                if self._runtime.loop is not None
                else None
            ),
        )
        return (
            f"repaired transcript {parts[0]}"
            + (f" child {child_run_id}" if child_run_id is not None else "")
            + "; quarantine "
            f"{result.quarantine_id}; retained through sequence "
            f"{result.retained_last_sequence}"
        )

    async def slash_memory(self, args: str) -> str:
        runtime = self._runtime
        project_id = runtime.metadata.project_id
        if project_id is None:
            return "memory: no associated project"
        registry = runtime.manager.project_registry
        if registry is None:
            return "memory: project registry unavailable"
        loop = getattr(runtime, "loop", None)
        reconciler = loop.memory_reconciler if loop is not None else None
        result = await asyncio.to_thread(
            run_memory_command,
            registry,
            project_id,
            args,
            reconciler,
            MemoryMutationAuthorization.direct_slash(),
        )
        return await result if inspect.isawaitable(result) else result

    async def slash_mcp(
        self, args: str
    ) -> str | ModelInputEnvelope | _ClientOnlySlashResult:
        stripped = args.strip()
        if stripped and stripped.partition(" ")[0] not in _SERVE_MCP_VERBS:
            return _ClientOnlySlashResult("mcp")
        if self._runtime.loop is None:
            raise RuntimeError(_UNAVAILABLE_NOTICE)
        return await self._runtime.loop.slash_mcp(stripped)

    async def slash_inbox(self, args: str) -> str:
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
            state = await asyncio.to_thread(
                ProjectInbox(
                    registry, sessions_root=registry.root.parent / "sessions"
                ).list,
                project_id,
            )
        except (InboxError, OSError) as exc:
            return f"project inbox unavailable: {exc}"
        return json.dumps(state, indent=2, sort_keys=True)


def build_registry(runtime: ServerRuntime) -> SlashCommandRegistry:
    """Build a per-request registry so custom commands and skills stay current."""

    opened = runtime.opened
    if opened is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
    loop = runtime.loop
    if loop is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
    catalog = loop.tool_registry.skill_catalog
    project_dir = discover_repo_root(Path(opened.metadata.cwd or runtime.cwd))
    return create_slash_registry(
        zeta_home=runtime.home,
        project_dir=project_dir,
        skill_catalog=catalog,
        project_registry_state=runtime.manager.project_registry_state,
        surface=SlashSurface.SERVE,
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
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "include_unavailable must be a boolean",
            )
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
        await runtime.loop.refresh_live_config()
    registry = await asyncio.to_thread(build_registry, runtime)
    repair_available = transcript_repair_available(runtime.manager)
    entries = [
        entry
        for entry in _entries_from_registry(
            registry, include_unavailable=include_unavailable
        )
        if repair_available or entry.name != "repair-transcript"
    ]
    return {
        "commands": [entry.to_dict() for entry in entries],
        "notices": list(registry.notices),
    }


async def can_run_while_busy(runtime: ServerRuntime, text: str) -> bool:
    """Return whether slash_run can resolve without starting model work."""

    first_line = text.split("\n", 1)[0]
    parts = first_line[1:].split(maxsplit=1)
    if not parts:
        return False
    name = parts[0]
    tail = parts[1] if len(parts) == 2 else ""
    if runtime.loop is not None:
        await runtime.loop.refresh_live_config()
    registry = await asyncio.to_thread(build_registry, runtime)
    command = registry.command_for(name)
    if command is not None:
        if not registry.available(command):
            return True
        return command.serve is not None and command.serve.can_run_while_busy(tail)
    custom = registry.custom_command_for(name)
    if custom is not None:
        return custom.kind == "exec" or registry.needs_inline_shell_resolution(text)
    if any(skill.name == name for skill in registry.skill_entries):
        return False
    return any(
        prompt_name == name
        for prompt_name, _description, _source in registry.mcp_prompt_entries
    )


async def run_command(
    runtime: ServerRuntime,
    text: str,
    *,
    register_model_input: Callable[[ModelInputEnvelope], str] | None = None,
    approval_control: _ApprovalProjection | None = None,
    run_blocking: Callable[[Callable[[], object]], Awaitable[object]] | None = None,
    busy: bool = False,
) -> dict[str, object]:
    """Dispatch one slash invocation through the shared registry."""

    if not text.startswith("/"):
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "text must start with '/'")
    first_line = text.split("\n", 1)[0]
    parts = first_line[1:].split(maxsplit=1)
    if not parts or not parts[0]:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "text must name a command")
    name = parts[0]
    tail = parts[1] if len(parts) == 2 else ""
    if runtime.loop is not None:
        await runtime.loop.refresh_live_config()
    registry = await asyncio.to_thread(build_registry, runtime)
    command = registry.command_for(name)
    if command is not None and not registry.available(command):
        return {"kind": SlashResultKindDescriptor.CLIENT_ONLY, "name": name}
    if (
        command is not None
        and command.serve is not None
        and command.serve.mutates(tail)
    ):
        ergonomics.require_mutable(
            runtime,
            approval_control is not None and approval_control.has_pending_approvals(),
        )
    # Prompt macros expand to user input rather than dispatch through a handler.
    # Inline shell spans need the client's approval loop; report those as
    # client-only rather than resolving with unapproved output here.
    custom = registry.custom_command_for(name)
    if custom is not None:
        if custom.kind != "prompt" or registry.needs_inline_shell_resolution(text):
            return {"kind": SlashResultKindDescriptor.CLIENT_ONLY, "name": name}
        envelope = registry.input_for_model(text)
        if busy:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )
        return _model_input_result(envelope, register_model_input)
    if name in {entry[0] for entry in registry.mcp_prompt_entries}:
        return {"kind": SlashResultKindDescriptor.CLIENT_ONLY, "name": name}
    session = ServerSlashSession(runtime, approval_control, run_blocking=run_blocking)
    result = await registry.dispatch_async(session, text)
    if result is None:
        return {"kind": SlashResultKindDescriptor.UNKNOWN, "name": name}
    if isinstance(result, SlashPromptError):
        return {"kind": SlashResultKindDescriptor.ERROR, "text": result.message}
    if isinstance(result, ModelInputEnvelope):
        if busy:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )
        return _model_input_result(result, register_model_input)
    if isinstance(result, _ClientOnlySlashResult):
        return {"kind": SlashResultKindDescriptor.CLIENT_ONLY, "name": result.name}
    if isinstance(result, str):
        response: dict[str, object] = {
            "kind": SlashResultKindDescriptor.OUTPUT,
            "text": result,
        }
        credential_error = getattr(result, "credential_store_error", None)
        if credential_error is not None:
            public_error = dict(credential_error)
            public_error["code"] = SlashErrorCodeDescriptor.MCP_CREDENTIAL_STORE
            response["credential_store_error"] = public_error
        return response
    raise ProtocolError(ErrorDescriptor.INTERNAL_ERROR, "unexpected slash result type")


def _model_input_result(
    envelope: ModelInputEnvelope,
    register: Callable[[ModelInputEnvelope], str] | None,
) -> dict[str, object]:
    if register is None:
        return {
            "kind": SlashResultKindDescriptor.MODEL_INPUT,
            "text": envelope.text,
            "display_text": envelope.display_text,
            "origin": envelope.origin.value,
        }
    return {
        "kind": SlashResultKindDescriptor.MODEL_INPUT,
        "input_id": register(envelope),
        "display_text": envelope.display_text,
    }


def _entries_from_registry(
    registry: SlashCommandRegistry, *, include_unavailable: bool
) -> list[SlashCommandInfo]:
    """Project the shared registry through serve's executable capabilities."""

    entries: list[SlashCommandInfo] = []
    for command in registry.builtin_commands:
        unavailable = None if registry.available(command) else _UNAVAILABLE_REASON
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
    "ServerSlashSession",
    "SlashCommandInfo",
    "build_registry",
    "can_run_while_busy",
    "list_commands",
    "run_command",
]
