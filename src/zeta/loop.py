"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import os
import shlex
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx

from .agent_background import (
    BackgroundAgentOwner,
    adopt_agent_children,
    recover_agent_children,
)
from .agent_budget import (
    MAX_AGENT_DEPTH,
    AgentTree,
    agent_tree_context,
    consume_turn,
)
from .agent_runner import run_agent_tool
from .core.abort import AbortSignal as ToolAbortSignal
from .core.approval import ApprovalPolicy
from .core.context import ContextAssembler
from .core.hooks import HookManager
from .core.store import ConversationStore
from .core.tool_dispatch import dispatch_tool_calls
from .mcp import (
    MCPConfigError,
    MCPMount,
    home_config_path,
    load_mcp_config_overlay,
    mount_mcp_servers,
    project_config_path,
)
from .mcp.commands import (
    MCP_USAGE,
    MCPCommandError,
    add_and_mount,
    parse_add_command,
    remove_and_unshadow,
)
from .prompts import load_identity
from .tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from .tools.agent import agent_result
from .tools.agent_presets import (
    compose_system_prompt,
    get_agent_preset,
)
from .tools.plan_mode import (
    EXIT_PLAN_MODE,
    PLAN_MODE_PREAMBLE,
    PLAN_MODE_TOOLS,
)
from .tools.registry import (
    ToolExecutionContext,
    _validate_unique_tool_call_ids,
    validate_tool_result,
)
from .types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    flatten_tool_content,
)

TaskResult = TypeVar("TaskResult")
MAX_ERROR_MESSAGE = 400


async def _close_completion(
    completion: AsyncIterator[StreamEvent] | None,
) -> BaseException | None:
    if completion is None:
        return None
    close = getattr(completion, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except BaseException as exc:
        return exc
    return None


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _error_info(error: BaseException) -> ErrorInfo:
    """Normalize provider and transport failures for the transcript."""

    code = getattr(error, "code", None)
    if type(code) is not str or not code:
        if isinstance(error, TimeoutError):
            code = "timeout"
        elif isinstance(error, httpx.TransportError):
            code = "transport_error"
        else:
            cause = error.__cause__
            while cause is not None:
                if isinstance(cause, httpx.TransportError):
                    code = "transport_error"
                    break
                cause = cause.__cause__
            else:
                code = "backend_error"
    try:
        message = str(error).strip()
    except Exception:
        message = ""
    if not message:
        message = type(error).__name__
    if len(message) > MAX_ERROR_MESSAGE:
        message = f"{message[: MAX_ERROR_MESSAGE - 3]}..."
    return ErrorInfo(code, message)


def _validated_tool_result(result: object, expected_id: str) -> ToolResult:
    if isinstance(result, ToolResult):
        if type(result.tool_call_id) is not str or not result.tool_call_id:
            return ToolResult(expected_id, "invalid tool result: call id", True)
        if type(result.content) is not str:
            return ToolResult(expected_id, "invalid tool result: content", True)
        if type(result.is_error) is not bool:
            return ToolResult(expected_id, "invalid tool result: is_error", True)
        if result.tool_call_id != expected_id:
            return ToolResult(
                expected_id,
                f"tool result id mismatch: expected {expected_id}, got {result.tool_call_id}",
                is_error=True,
            )
        return result
    if not isinstance(result, Mapping):
        return ToolResult(
            expected_id,
            "invalid tool result: expected structured result",
            True,
        )
    try:
        structured_result = validate_tool_result(result)
    except ValueError as exc:
        return ToolResult(expected_id, f"invalid tool result: {exc}", True)
    return ToolResult(
        expected_id,
        flatten_tool_content(structured_result["content"]),
        structured_result["isError"],
        content_blocks=structured_result["content"],
        structured_content=structured_result["structuredContent"],
    )


class AgentLoop:
    def __init__(
        self,
        backend: CompletionBackend,
        store: ConversationStore,
        *,
        tools: Mapping[str, ToolHandler] | ToolRegistry | None = None,
        registry: ToolRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int = 150,
        context_assembler: ContextAssembler | None = None,
        system_prompt: str | Message | None = None,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        on_completion_success: Callable[[], None] | None = None,
        hooks: HookManager | None = None,
        skip_mcp_mount: bool = False,
        agent_depth: int = 0,
        agent_instance_id: str | None = None,
        agent_turn_budget: int | None = None,
        agent_tree: AgentTree | None = None,
        background_owner: BackgroundAgentOwner | None = None,
    ) -> None:
        if type(agent_depth) is not int or not 0 <= agent_depth <= MAX_AGENT_DEPTH:
            raise ValueError(f"agent depth must be between 0 and {MAX_AGENT_DEPTH}")
        self.backend = backend
        self.store = store
        self.agent_depth = agent_depth
        self.agent_instance_id = agent_instance_id
        if agent_turn_budget is not None and (
            type(agent_turn_budget) is not int or agent_turn_budget < 1
        ):
            raise ValueError("agent turn budget must be a positive integer")
        self._agent_turn_budget = agent_turn_budget
        self._agent_tree = agent_tree
        self._background_owner = background_owner or BackgroundAgentOwner(store)
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_child_stores: dict[str, ConversationStore] = {}
        self._agent_child_turns: dict[str, int] = {}
        self._agent_child_types: dict[str, str] = {}
        self._background_child_cancellers: dict[str, Callable[[], None]] = {}
        self._background_child_watchers: dict[str, asyncio.Task[Any]] = {}
        self._background_event_sink: Callable[[StreamEvent], None] | None = None
        self._mcp_notice_sink: Callable[[str], None] | None = None
        self._mcp_prompt_refresh: Callable[[MCPMount], None] | None = None
        recover_agent_children(self)
        if registry is not None and tools is not None:
            raise ValueError("pass only one tool registry")
        if registry is not None:
            self.tool_registry = registry
        elif isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, Mapping):
            self.tool_registry = ToolRegistry(store.cwd, register_builtin=False)
            schemas_by_name = {
                schema.get("name"): schema
                for schema in (tool_schemas or [])
                if isinstance(schema.get("name"), str)
            }
            for name, handler in tools.items():
                schema = schemas_by_name.get(name, {})
                parameters = schema.get("parameters", schema.get("input_schema"))
                if parameters is None:
                    parameters = {
                        key: value
                        for key, value in schema.items()
                        if key not in {"name", "description", "cache_control"}
                    }
                self.tool_registry.register(
                    name,
                    handler,
                    description=(
                        schema.get("description", "")
                        if isinstance(schema.get("description", ""), str)
                        else ""
                    ),
                    parameters=parameters,
                )
        elif tools is None:
            self.tool_registry = ToolRegistry(store.cwd)
        else:
            raise TypeError("tools must be a mapping or ToolRegistry")
        self._mcp_mount: MCPMount | None = None
        self._mcp_mount_attempted = skip_mcp_mount
        self._mcp_mount_task: asyncio.Task[None] | None = None
        self._mcp_home_hint: str | None = None
        self._mcp_project_dir_value: Path | None = (
            Path(self.store.cwd).expanduser().resolve()
        )
        self._mcp_config_error: str | None = None
        self._mcp_schema_names: set[str] = set()
        self._provided_tool_schemas = tool_schemas is not None
        self.tool_registry.bind_session_store(store)
        if (
            approval_policy is not None
            and self.tool_registry.approval_policy is not None
            and self.tool_registry.approval_policy is not approval_policy
        ):
            raise ValueError("pass only one approval policy")
        if approval_policy is not None:
            self.tool_registry.set_approval_policy(approval_policy)
        if self.tool_registry.approval_policy is not None:
            self.tool_registry.bind_approval_store(store)
        self.tool_schemas = list(
            tool_schemas if tool_schemas is not None else self.tool_registry.schemas
        )
        self.max_turns = max_turns
        if system_prompt is None:
            system_prompt = load_identity()
        self.context_assembler = context_assembler or ContextAssembler(
            store,
            token_budget=token_budget,
            retained_tail=retained_tail,
            system_prompt=system_prompt,
            backend=backend,
            on_completion_success=on_completion_success,
        )
        self.on_completion_success = on_completion_success
        self.hooks = hooks
        if self.hooks is not None:
            self.hooks.bind_session(store.session_id)
            if self.tool_registry.pre_execute_hook is None:
                self.tool_registry.set_pre_execute_hook(self.hooks.pre_tool)
        self._plan_mode = False
        self._plan_mode_prior_prompt: Message | None = None
        self._plan_mode_prior_deny: frozenset[str] | None = None
        if "agent" in self.tool_registry.definitions_by_name:
            self.tool_registry.set_agent_runner(self._run_agent_tool)

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        """Restrict the assistant to read-only tools, or lift the restriction.

        Both edges rewrite the system prompt and the advertised tools, which
        Anthropic caches as one prefix, so each toggle costs a cache miss. That
        is fine for an occasional mode change and is why nothing flips this
        per turn.
        """

        if enabled == self._plan_mode:
            return
        assembler = self.context_assembler
        if enabled:
            self._plan_mode_prior_prompt = assembler.system_prompt
            composed = compose_system_prompt(
                assembler.system_prompt, PLAN_MODE_PREAMBLE
            )
            assert isinstance(composed, Message)
            assembler.system_prompt = composed
        elif self._plan_mode_prior_prompt is not None:
            assembler.system_prompt = self._plan_mode_prior_prompt
            self._plan_mode_prior_prompt = None
        self._apply_plan_mode_denials(enabled)
        self._plan_mode = enabled

    def _apply_plan_mode_denials(self, enabled: bool) -> None:
        """Deny the mutating tools outright, not just hide their schemas.

        Withholding a schema stops a well-behaved model, not a determined one
        replaying an older tool name, so plan mode also refuses the calls.
        """

        policy = self.tool_registry.approval_policy
        if policy is None:
            return
        if enabled:
            self._plan_mode_prior_deny = policy.always_deny
            allowed = PLAN_MODE_TOOLS | {EXIT_PLAN_MODE}
            policy.always_deny = policy.always_deny | {
                name
                for name in self.tool_registry.definitions_by_name
                if name not in allowed
            }
        elif self._plan_mode_prior_deny is not None:
            policy.always_deny = self._plan_mode_prior_deny
            self._plan_mode_prior_deny = None

    def _active_tool_schemas(self) -> list[ToolSchema]:
        """Return the schemas this turn advertises, honoring plan mode."""

        if not self._plan_mode:
            return [
                schema
                for schema in self.tool_schemas
                if schema.get("name") != EXIT_PLAN_MODE
            ]
        allowed = PLAN_MODE_TOOLS | {EXIT_PLAN_MODE}
        return [
            schema for schema in self.tool_schemas if schema.get("name") in allowed
        ]

    def _approved_plan_exit(self, calls: Sequence[ToolCall]) -> bool:
        """Report whether this batch carried an approved exit_plan_mode call."""

        for call in calls:
            if call.name != EXIT_PLAN_MODE:
                continue
            result = self._existing_tool_result(call.id)
            if result is not None and not result.is_error:
                return True
        return False

    def set_model(self, model: str) -> None:
        """Set the model used by subsequent provider completions."""

        if not model.strip():
            raise ValueError("model must be a nonempty name")
        if hasattr(self.backend, "model"):
            self.backend.model = model
        else:
            self._model = model

    def abort(self) -> None:
        """Signal the active tool batch before the caller cancels the turn."""

        self.tool_registry.abort()
        self._background_owner.cancel_all()

    def set_background_event_sink(
        self, sink: Callable[[StreamEvent], None] | None
    ) -> None:
        """Set the sink for progress from children that outlive their turn."""

        self._background_event_sink = sink

    def set_mcp_notice_sink(self, sink: Callable[[str], None] | None) -> None:
        """Set the sink for MCP mount notices."""

        self._mcp_notice_sink = sink

    def set_mcp_prompt_refresh(
        self, callback: Callable[[MCPMount], None] | None
    ) -> None:
        """Set the owner callback for live MCP prompt commands."""

        self._mcp_prompt_refresh = callback
        if callback is not None and self._mcp_mount is not None:
            callback(self._mcp_mount)

    @property
    def mcp_summary(self) -> str:
        if self._mcp_mount is None:
            return "mcp: 0 mounted, 0 failed"
        return self._mcp_mount.summary

    async def slash_mcp(self, args: str) -> str:
        """Show MCP state, reconnect, add, or remove one configured server."""

        await self._ensure_mcp_servers()
        mount = self._mcp_mount
        try:
            parts = shlex.split(args)
        except ValueError as exc:
            return f"mcp error: {exc}"
        if self._mcp_config_error is not None:
            return f"mcp error: {self._mcp_config_error}"
        if mount is None:
            return "mcp: no configured servers"
        if not parts:
            return mount.render()
        verb = parts[0]
        try:
            if verb == "reconnect":
                if len(parts) != 2:
                    return MCP_USAGE
                await mount.reconnect(parts[1], notice_sink=self._mcp_notice_sink)
                return mount.render()
            if verb == "add":
                await add_and_mount(
                    mount,
                    parse_add_command(parts[1:]),
                    target=self._mcp_add_target(),
                    notice_sink=self._mcp_notice_sink,
                )
                return mount.render()
            if verb == "remove":
                if len(parts) != 2:
                    return MCP_USAGE
                await remove_and_unshadow(
                    mount,
                    parts[1],
                    home_path=self._mcp_home_path(),
                    load_home=lambda: load_mcp_config_overlay(
                        home=self._mcp_home_hint, project_dir=None
                    ).configured_servers,
                    notice_sink=self._mcp_notice_sink,
                )
                return mount.render()
        except (MCPCommandError, ValueError) as exc:
            return f"mcp error: {exc}"
        return MCP_USAGE

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        """Resolve one mounted MCP prompt for the next model turn."""

        await self._ensure_mcp_servers()
        if self._mcp_mount is None:
            raise RuntimeError("MCP mount is unavailable")
        return await self._mcp_mount.get_prompt(name, arguments)

    def _mcp_add_target(self) -> Path:
        project_dir = self._mcp_project_dir_value
        if project_dir is not None:
            return project_config_path(project_dir)
        return home_config_path(self._mcp_home_hint)

    def _mcp_home_path(self) -> Path:
        override = os.environ.get("ZETA_MCP_CONFIG")
        if override:
            return Path(override).expanduser().resolve()
        return home_config_path(self._mcp_home_hint).resolve()

    @property
    def background_children_running(self) -> bool:
        return self._background_owner.running or bool(self._background_child_cancellers)

    def _publish_background_event(self, event: StreamEvent) -> None:
        if self._background_event_sink is not None:
            self._background_event_sink(event)

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, TaskResult],
    ) -> asyncio.Task[TaskResult]:
        task = asyncio.create_task(coroutine)
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

    def _child_result_payload(
        self,
        tool_call_id: str,
        content: str,
        *,
        error: bool,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        status: str | None = None,
        child_instance_id: str | None = None,
        description: str | None = None,
        depth: int | None = None,
        budget_exhausted: bool = False,
    ) -> dict[str, object]:
        child_store = self._agent_child_stores.get(tool_call_id)
        path = (
            child_session_path
            if child_session_path is not None
            else str(child_store.session_dir)
            if child_store is not None
            else ""
        )
        turns = (
            turns_used
            if turns_used is not None
            else self._agent_child_turns.get(tool_call_id, 0)
        )
        return agent_result(
            content,
            error=error,
            turns_used=turns,
            child_session_path=path,
            agent_type=(
                preset.name
                if (preset := get_agent_preset(agent_type)) is not None
                else None
            ),
            status=status,
            child_instance_id=child_instance_id,
            description=description,
            depth=depth,
            budget_exhausted=budget_exhausted,
        )

    def _canceled_agent_result(
        self,
        tool_call_id: str,
        *,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
    ) -> ToolResult:
        return _validated_tool_result(
            self._child_result_payload(
                tool_call_id,
                "tool execution canceled",
                error=True,
                child_session_path=child_session_path,
                turns_used=turns_used,
                agent_type=agent_type,
            ),
            tool_call_id,
        )

    async def _run_agent_tool(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal,
        publisher: ToolStreamPublisher | None,
        execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, object]:
        return await run_agent_tool(
            self,
            tool_call,
            arguments,
            abort_signal,
            publisher,
            validate_result=_validated_tool_result,
            error_message=lambda exc: _error_info(exc).message,
            execution_context=execution_context,
        )

    def prepare_resume_pending_tool(self, request_id: str) -> bool:
        """Reserve the abort generation before resuming an approved tool."""

        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return False
        if self._existing_tool_result(state[0].id) is not None:
            return False
        self.tool_registry.start_batch()
        return True

    def run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._run_turn(
            user_text,
            user_message=user_message,
            persist_user_message=persist_user_message,
            abort_signal=abort_signal,
        )

    async def close(self, *, cancel_background: bool = True) -> None:
        """Close session-owned transports and background processes."""

        if cancel_background and self.agent_depth == 0:
            self._background_owner.cancel_all()
        elif cancel_background:
            for cancel in tuple(self._background_child_cancellers.values()):
                cancel()
        watchers = tuple(self._background_child_watchers.values())
        if cancel_background and watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        if cancel_background and self.agent_depth == 0:
            await self._background_owner.wait()
        tracked_tasks = tuple(
            task
            for task in self._tracked_tasks
            if cancel_background or task not in self._background_child_watchers.values()
        )
        for task in tracked_tasks:
            task.cancel()
        await asyncio.gather(*tracked_tasks, return_exceptions=True)
        if self.hooks is not None:
            self.hooks.stop()
            await self.hooks.close()
        if self._mcp_mount is not None:
            await self._mcp_mount.close()
            self._mcp_mount = None
        if self._mcp_mount_task is not None and not self._mcp_mount_task.done():
            self._mcp_mount_task.cancel()
            await asyncio.gather(self._mcp_mount_task, return_exceptions=True)
        await self.tool_registry.background_tasks.close()

    def session_start(self) -> None:
        if self.hooks is not None:
            self.hooks.session_start()

    async def _ensure_mcp_servers(self) -> None:
        if self._mcp_mount_attempted:
            return
        if self._mcp_mount_task is None:
            self._mcp_mount_task = asyncio.create_task(self._mount_mcp_servers())
        await asyncio.shield(self._mcp_mount_task)

    async def _mount_mcp_servers(self) -> None:
        try:
            config = load_mcp_config_overlay(
                home=self._mcp_home_hint,
                project_dir=self._mcp_project_dir_value,
            )
            self._mcp_mount = await mount_mcp_servers(
                self.tool_registry, config, notice_sink=self._mcp_notice_sink
            )
        except MCPConfigError as exc:
            self._mcp_config_error = str(exc)
            self._mcp_mount = MCPMount(self.tool_registry, {}, {})
        self._mcp_mount.set_schema_refresh(self._refresh_mcp_tool_schemas)
        if self._mcp_prompt_refresh is not None:
            self._mcp_mount.set_prompt_refresh(self._mcp_prompt_refresh)
        self._mcp_mount_attempted = True

    def set_mcp_scope(
        self,
        *,
        home: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        """Set the home + project scope this loop uses for MCP config files."""

        self._mcp_home_hint = None if home is None else str(home)
        self._mcp_project_dir_value = (
            None if project_dir is None else Path(project_dir).expanduser().resolve()
        )

    def _refresh_mcp_tool_schemas(self, mount: MCPMount | None = None) -> None:
        mount = mount or self._mcp_mount
        if mount is None:
            return
        mcp_prefixes = tuple(f"{name}:" for name in mount.configs)
        current_mcp = [
            schema
            for schema in self.tool_registry.schemas
            if isinstance(schema.get("name"), str)
            and schema["name"].startswith(mcp_prefixes)
        ]
        current_names = {
            schema["name"]
            for schema in current_mcp
            if isinstance(schema.get("name"), str)
        }
        if not self._provided_tool_schemas:
            self.tool_schemas = list(self.tool_registry.schemas)
            self._mcp_schema_names = current_names
            return
        names_to_replace = self._mcp_schema_names | current_names
        self.tool_schemas = [
            schema
            for schema in self.tool_schemas
            if not (
                isinstance(schema.get("name"), str)
                and schema["name"] in names_to_replace
            )
        ] + current_mcp
        self._mcp_schema_names = current_names

    async def ensure_mcp_servers(self) -> None:
        """Connect MCP servers before a direct tool resume."""

        await self._ensure_mcp_servers()

    async def resume_pending_tool(
        self,
        request_id: str,
        *,
        prepared: bool = False,
        event_sink: Callable[[StreamEvent], None] | None = None,
    ) -> ToolResult | None:
        """Finish a durable approval request before starting another turn."""

        await self._ensure_mcp_servers()
        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        if not prepared:
            self.tool_registry.start_batch()
        abort_signal = self.tool_registry.abort_signal

        def lifecycle(kind: str) -> None:
            if event_sink is None:
                return
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is not None:
                event_sink(StreamEvent(event_type, tool_call=tool_call))

        try:
            result = await self.tool_registry.execute(
                tool_call,
                abort_signal=abort_signal,
                _scope_signal=abort_signal,
                _lifecycle_sink=lifecycle,
            )
        except asyncio.CancelledError:
            result = self.finalize_canceled(request_id)
            if event_sink is not None and result is not None:
                event_sink(
                    StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                )
            raise
        except Exception as exc:
            result = ToolResult(tool_call.id, str(exc), is_error=True)
        result = _validated_tool_result(result, tool_call.id)
        if result.content == "tool execution canceled" and result.is_error:
            result = self.finalize_canceled(request_id)
        else:
            result = self._finalize_tool_results([tool_call], [result])[0]
        if event_sink is not None and result is not None:
            event_sink(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=tool_call,
                    tool_result=result,
                )
            )
        return result

    def finalize_canceled(self, request_id: str) -> ToolResult | None:
        """Persist one canceled result for a durable approval request."""

        state = self.store.approval_states().get(request_id)
        if state is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        return self._finalize_tool_results([tool_call], [None])[0]

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None:
        for message in reversed(self.store.messages()):
            result = message.tool_result
            if result is not None and result.tool_call_id == tool_call_id:
                return result
        return None

    async def _run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if self.hooks is not None:
            self.hooks.user_prompt_submit(user_text)
        if user_message is None:
            user_message = Message(MessageRole.USER, [TextContent(user_text)])
        elif user_message.role is not MessageRole.USER:
            raise ValueError("user_message must have the user role")
        if persist_user_message:
            self.store.append_message(user_message)
        elif user_message not in self.store.messages():
            raise ValueError("cannot reuse a user message that is not persisted")
        agent_tree = AgentTree() if self.agent_depth == 0 else None
        setup_error: ErrorInfo | None = None
        try:
            await self._ensure_mcp_servers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            setup_error = _error_info(exc)

        for notification in self.store.agent_notifications():
            yield StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"notification_id": notification.id, **notification.data},
            )
            self.store.acknowledge_agent_notification(notification.id)

        if setup_error is not None:
            self._persist_partial_with_cancelled_tools([], None, failure=setup_error)
            yield StreamEvent(StreamEventType.AGENT_START)
            yield StreamEvent(StreamEventType.ERROR, error=setup_error)
            yield StreamEvent(StreamEventType.AGENT_END)
            return
        yield StreamEvent(StreamEventType.AGENT_START)

        for turn_number in range(1, self.max_turns + 1):
            if (
                self.agent_depth
                and (
                    error := consume_turn(
                        self._agent_tree.budget
                        if self._agent_tree is not None
                        else None
                    )
                )
                is not None
            ):
                yield StreamEvent(StreamEventType.ERROR, error=error)
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            self.tool_registry.start_batch()
            turn_abort_signal = abort_signal or self.tool_registry.abort_signal
            yield StreamEvent(
                StreamEventType.TURN_START,
                data={"turn": turn_number},
            )
            partial_blocks: list[ContentBlock] = []
            assistant_message: Message | None = None
            completion: AsyncIterator[StreamEvent] | None = None
            completion_succeeded = False
            provider_error: ErrorInfo | None = None
            try:
                if self.context_assembler.needs_compaction():
                    yield StreamEvent(
                        StreamEventType.COMPACTION_START,
                        data={"turn": turn_number},
                    )
                context_messages = await self.context_assembler.assemble(
                    backend=self.backend
                )
                context = self.context_assembler.last_context
                if context is not None and context.compacted:
                    yield StreamEvent(
                        StreamEventType.COMPACTION_END,
                        data={
                            "turn": turn_number,
                            "token_count": context.token_count,
                        },
                    )
                completion = self.backend.complete(
                    context_messages, self._active_tool_schemas()
                )
                async for event in completion:
                    self.context_assembler.observe_event(event)
                    if event.type is StreamEventType.ERROR:
                        provider_error = (
                            event.error
                            if isinstance(event.error, ErrorInfo)
                            else ErrorInfo(
                                "backend_error",
                                "provider emitted an invalid error event",
                            )
                        )
                        yield StreamEvent(
                            StreamEventType.ERROR,
                            error=provider_error,
                            data=dict(event.data),
                        )
                        break
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial_blocks.append(event.content)
                        if event.delta is not None:
                            partial_blocks.append(TextContent(event.delta))
                    if (
                        event.message is not None
                        and event.type is StreamEventType.MESSAGE_END
                    ):
                        assistant_message = event.message
                    if event.type is StreamEventType.MESSAGE_END:
                        if not event.data.get("truncated"):
                            completion_succeeded = True
                    yield event
                    if provider_error is not None:
                        yield StreamEvent(
                            StreamEventType.ERROR,
                            error=provider_error,
                        )
                        break
                if provider_error is None and not completion_succeeded:
                    provider_error = ErrorInfo(
                        "stream_error",
                        "provider stream ended before completion",
                    )
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        error=provider_error,
                    )
            except asyncio.CancelledError:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except GeneratorExit:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except Exception as exc:
                await _close_completion(completion)
                if _task_is_cancelling():
                    self._persist_partial_for_control(partial_blocks, assistant_message)
                    raise asyncio.CancelledError() from exc
                error = _error_info(exc)
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message, failure=error
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=error,
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            cleanup_error = await _close_completion(completion)
            if _task_is_cancelling():
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise asyncio.CancelledError()
            if cleanup_error is not None:
                self._persist_partial_with_cancelled_tools(
                    partial_blocks,
                    assistant_message,
                    failure=_error_info(cleanup_error),
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=_error_info(cleanup_error),
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            if provider_error is not None:
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message, failure=provider_error
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            if completion_succeeded and self.on_completion_success is not None:
                self.on_completion_success()

            if assistant_message is None and partial_blocks:
                assistant_message = Message(MessageRole.ASSISTANT, partial_blocks)
            if assistant_message is None:
                assistant_message = Message(MessageRole.ASSISTANT)
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            _validate_unique_tool_call_ids(calls)
            approval_requests: list[tuple[str, ToolCall]] = []
            for tool_call in calls:
                request = self.tool_registry.prepare_approval(tool_call)
                if request is not None:
                    approval_requests.append((request.request_id, request.tool_call))
            self.store.append_message_with_approval_requests(
                _durable_message(assistant_message),
                approval_requests,
            )
            if not calls:
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return

            dispatch = dispatch_tool_calls(
                self,
                calls,
                _validated_tool_result,
                abort_signal=turn_abort_signal,
            )
            with agent_tree_context(agent_tree):
                try:
                    async for event in dispatch:
                        yield event
                finally:
                    await dispatch.aclose()
            if self._plan_mode and self._approved_plan_exit(calls):
                self.set_plan_mode(False)
            yield StreamEvent(
                StreamEventType.TURN_END,
                message=assistant_message,
                data={"turn": turn_number, "tool_calls": len(calls)},
            )

        yield StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("max_turns", f"maximum turns reached: {self.max_turns}"),
        )
        yield StreamEvent(StreamEventType.AGENT_END)

    def _finalize_tool_results(
        self,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        new_results: list[tuple[ToolCall, ToolResult]] = []
        for call, slot in zip(calls, slots, strict=True):
            result = self._existing_tool_result(call.id)
            stored_result = result
            child_store = self._agent_child_stores.get(call.id)
            candidate = result if result is not None else slot
            if child_store is not None and (
                candidate is None or candidate.content == "tool execution canceled"
            ):
                result = self._canceled_agent_result(
                    call.id,
                    child_session_path=str(child_store.session_dir),
                    agent_type=self._agent_child_types.get(call.id),
                )
            if result is None:
                result = slot
                if result is None and call.name == "agent":
                    result = self._canceled_agent_result(call.id)
                if result is None:
                    result = ToolResult(
                        call.id,
                        "tool execution canceled",
                        is_error=True,
                    )
            if stored_result is None:
                new_results.append((call, result))
            results.append(result)
        for call, result in new_results:
            self.store.append_message(
                Message(
                    MessageRole.TOOL_RESULT,
                    [TextContent(result.content)],
                    tool_result=result,
                )
            )
            if self.hooks is not None:
                self.hooks.post_tool(call.name, result.content)
            if call.name == "agent":
                is_background = (
                    result.structured_content is not None
                    and result.structured_content.get("status") == "running"
                )
                if is_background:
                    continue
                child_store = self._agent_child_stores.pop(call.id, None)
                if child_store is not None:
                    if result.content == "tool execution canceled":
                        child_store.mark_agent_canceled(call.id)
                    else:
                        adopt_agent_children(
                            child_store,
                            self.store,
                            background_owner=self._background_owner,
                        )
                        child_store.finish_agent_parent()
                self.store.finish_agent_child(
                    f"{self.agent_instance_id}:{child_store.session_id}"
                    if child_store is not None and self.agent_instance_id is not None
                    else call.id
                )
                self._agent_child_turns.pop(call.id, None)
                self._agent_child_types.pop(call.id, None)
        return results

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        if assistant_message is None:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent) or block.signature
            ]
            if not durable_blocks and failure is None:
                return
            assistant_message = Message(MessageRole.ASSISTANT, durable_blocks)
        if failure is not None:
            metadata = dict(assistant_message.metadata)
            metadata[FAILED_TURN_MARKER] = True
            metadata[FAILED_TURN_ERROR] = failure.to_dict()
            assistant_message = Message(
                assistant_message.role,
                assistant_message.content,
                tool_result=assistant_message.tool_result,
                metadata=metadata,
            )
        self.store.append_message(_durable_message(assistant_message))

    def _persist_partial_for_control(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        try:
            self._persist_partial_with_cancelled_tools(
                partial_blocks, assistant_message
            )
        except Exception as exc:
            try:
                warnings.warn(
                    f"failed to persist partial state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except:
                pass

    def _persist_partial_with_cancelled_tools(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message, failure=failure)
        blocks = (
            assistant_message.content
            if assistant_message is not None
            else partial_blocks
        )
        calls = [
            block.tool_call for block in blocks if isinstance(block, ToolUseContent)
        ]
        self._finalize_tool_results(calls, [None] * len(calls))


def _durable_message(message: Message) -> Message:
    content = [
        block
        for block in message.content
        if not isinstance(block, ThinkingContent) or block.signature
    ]
    if len(content) == len(message.content):
        return message
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=dict(message.metadata),
    )
