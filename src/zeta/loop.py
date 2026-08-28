"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx

from .core.abort import AbortSignal as ToolAbortSignal
from .core.approval import ApprovalPolicy
from .core.context import ContextAssembler
from .core.hooks import HookManager
from .core.store import ConversationStore
from .core.tool_dispatch import dispatch_tool_calls
from .mcp import MCPMount, mount_mcp_servers
from .prompts import load_identity
from .tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from .tools.agent import ChildApprovalPolicy, agent_result
from .tools.agent_presets import (
    GENERAL_PRESET,
    agent_type_names,
    compose_system_prompt,
    get_agent_preset,
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
    assistant_text,
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
        message = f"{message[:MAX_ERROR_MESSAGE - 3]}..."
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
        max_turns: int = 50,
        context_assembler: ContextAssembler | None = None,
        system_prompt: str | Message | None = None,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        on_completion_success: Callable[[], None] | None = None,
        hooks: HookManager | None = None,
        skip_mcp_mount: bool = False,
    ) -> None:
        self.backend = backend
        self.store = store
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_child_stores: dict[str, ConversationStore] = {}
        self._agent_child_turns: dict[str, int] = {}
        self._agent_child_types: dict[str, str] = {}
        self._recover_agent_children()
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
            tool_schemas
            if tool_schemas is not None else self.tool_registry.schemas
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
        if "agent" in self.tool_registry.definitions_by_name:
            self.tool_registry.set_agent_runner(self._run_agent_tool)

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
    ) -> dict[str, object]:
        child_store = self._agent_child_stores.get(tool_call_id)
        path = (
            child_session_path
            if child_session_path is not None
            else str(child_store.session_dir) if child_store is not None else ""
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

    def _recover_agent_children(self) -> None:
        """Resolve child markers left by a process exit before resuming."""

        for tool_call_id, marker in self.store.agent_children().items():
            tool_call = ToolCall.from_dict(marker["tool_call"])
            existing_result = self._existing_tool_result(tool_call_id)
            if existing_result is None:
                self.store.append_message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        [TextContent("tool execution canceled")],
                        tool_result=self._canceled_agent_result(
                            tool_call_id,
                            child_session_path=marker["child_session_path"],
                            turns_used=marker.get("turns_used", 0),
                            agent_type=marker.get("agent_type"),
                        ),
                    )
                )
            child_path = Path(marker["child_session_path"])
            agents_root = self.store.session_dir / "agents"
            if child_path.parent == agents_root and child_path.name.isdigit():
                child_store = ConversationStore(
                    agents_root,
                    session_id=child_path.name,
                    cwd=self.store.cwd,
                )
                if existing_result is None:
                    child_store.mark_agent_canceled(tool_call.id)
                else:
                    child_store.finish_agent_parent()
            self.store.finish_agent_child(tool_call.id)

    async def _run_agent_tool(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal,
        publisher: ToolStreamPublisher | None,
        execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, object]:
        prompt = arguments.get("prompt")
        description = arguments.get("description")
        agent_type = arguments.get("agent_type", GENERAL_PRESET.name)
        if type(prompt) is not str or not prompt.strip():
            return self._child_result_payload(
                tool_call.id,
                "agent error: prompt must be a nonempty string",
                error=True,
            )
        if type(description) is not str or not description.strip():
            return self._child_result_payload(
                tool_call.id,
                "agent error: description must be a nonempty string",
                error=True,
            )
        preset = get_agent_preset(agent_type)
        if preset is None:
            return self._child_result_payload(
                tool_call.id,
                "agent error: unknown agent_type "
                f"{agent_type!r}; expected one of: {', '.join(agent_type_names())}",
                error=True,
            )
        await self._ensure_mcp_servers()
        stored_agent_type = (
            None if preset.name == GENERAL_PRESET.name else preset.name
        )
        child_number = self.store.allocate_agent_index()
        agents_root = self.store.session_dir / "agents"
        child_store = ConversationStore(
            agents_root,
            session_id=str(child_number),
            cwd=self.store.cwd,
        )
        child_store.mark_agent_parent(tool_call.id, agent_type=stored_agent_type)
        child_path = str(child_store.session_dir)
        child_instance_id = f"{self.store.session_id}:{child_number}"
        self._agent_child_stores[tool_call.id] = child_store
        self._agent_child_turns[tool_call.id] = 0
        self._agent_child_types[tool_call.id] = preset.name
        if publisher is not None:
            publisher.set_metadata({"child_session_path": child_path})
        self.store.register_agent_child(
            tool_call,
            child_session_path=child_path,
            description=description,
            agent_type=stored_agent_type,
        )
        excluded_names = {"agent"}
        if preset.tool_names is not None:
            excluded_names.update(
                set(self.tool_registry.definitions_by_name) - preset.tool_names
            )
        child_registry = self.tool_registry.clone_for_session(
            child_store,
            exclude_names=excluded_names,
        )
        parent_policy = self.tool_registry.approval_policy
        child_policy: ChildApprovalPolicy | None = None
        if parent_policy is not None:
            child_policy = ChildApprovalPolicy(
                parent_policy,
                child_store,
                description,
                child_instance_id,
            )
            child_registry.set_approval_policy(child_policy)
        child_loop = AgentLoop(
            self.backend,
            child_store,
            registry=child_registry,
            max_turns=preset.turn_cap,
            token_budget=self.context_assembler.token_budget,
            retained_tail=self.context_assembler.retained_tail,
            system_prompt=compose_system_prompt(
                self.context_assembler.system_prompt,
                preset.preamble,
            ),
            skip_mcp_mount=True,
        )

        lifecycle_sink = (
            execution_context.lifecycle_sink
            if execution_context is not None
            else None
        )

        def publish(status: str) -> None:
            if publisher is not None:
                publisher.publish(f"{description}: {status}\n", "stdout")

        def child_turns() -> int:
            return self._agent_child_turns.get(tool_call.id, 0)

        def child_result(text: str, *, error: bool) -> dict[str, object]:
            return self._child_result_payload(
                tool_call.id,
                text,
                error=error,
                child_session_path=child_path,
                agent_type=preset.name,
            )

        async def consume() -> dict[str, object]:
            final_message: Message | None = None
            last_assistant_text = ""
            cap_hit = False
            error_message: str | None = None
            try:
                async for event in child_loop.run_turn(prompt):
                    if event.type is StreamEventType.TURN_START:
                        publish(f"turn {child_turns() + 1}: thinking")
                    elif event.type is StreamEventType.TOOL_APPROVAL_START:
                        name = event.tool_call.name if event.tool_call is not None else "tool"
                        publish(f"turn {child_turns() + 1}: approval pending: {name}")
                        if lifecycle_sink is not None:
                            lifecycle_sink("approval_start", event.tool_call)
                    elif event.type is StreamEventType.TOOL_APPROVAL_END:
                        if lifecycle_sink is not None:
                            lifecycle_sink("approval_end", event.tool_call)
                    elif event.type is StreamEventType.TOOL_EXECUTION_START:
                        name = event.tool_call.name if event.tool_call is not None else "tool"
                        arguments = (
                            event.tool_call.arguments
                            if event.tool_call is not None
                            else {}
                        )
                        summary = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                        publish(f"turn {child_turns() + 1}: tool: {name} {summary}")
                    elif event.type is StreamEventType.TURN_END:
                        turns = child_turns() + 1
                        self._agent_child_turns[tool_call.id] = turns
                        self.store.update_agent_child_turns(tool_call.id, turns)
                        if event.message is not None:
                            last_assistant_text = _assistant_text_snippet(event.message)
                        if event.data.get("tool_calls") == 0 and event.message is not None:
                            final_message = event.message
                    elif event.type is StreamEventType.ERROR and event.error is not None:
                        if event.error.code == "max_turns":
                            cap_hit = True
                        else:
                            error_message = event.error.message
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_message = _error_info(exc).message
            if cap_hit:
                return child_result(
                    f"agent error: child reached the {preset.turn_cap}-turn cap; "
                    f"partial state is saved at {child_path}; "
                    f"last assistant text: {last_assistant_text or '[none]'}; "
                    f"turns used: {child_turns()}",
                    error=True,
                )
            if error_message is not None:
                return child_result(f"agent error: {error_message}", error=True)
            if final_message is None:
                return child_result(
                    "agent error: child ended without a final response", error=True
                )
            final_text = assistant_text(final_message)
            if not final_text.strip():
                return child_result(
                    "agent error: child returned an empty final assistant message",
                    error=True,
                )
            return child_result(final_text, error=False)

        child_task = self._create_task(consume())
        abort_task = self._create_task(abort_signal.wait())
        child_canceled = False

        async def cancel_child() -> None:
            nonlocal child_canceled
            if child_canceled:
                return
            child_canceled = True
            child_loop.abort()
            if not child_task.done():
                child_task.cancel()
            await asyncio.gather(child_task, return_exceptions=True)
            child_store.mark_agent_canceled(tool_call.id)

        try:
            done, _ = await asyncio.wait(
                (child_task, abort_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_task in done:
                await cancel_child()
                raise asyncio.CancelledError()
            result = child_task.result()
            return result
        except asyncio.CancelledError:
            await cancel_child()
            raise
        finally:
            if child_policy is not None:
                child_policy.cleanup()
            if not abort_task.done():
                abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)
            await child_loop.close()

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
    ) -> AsyncIterator[StreamEvent]:
        return self._run_turn(
            user_text,
            user_message=user_message,
            persist_user_message=persist_user_message,
        )

    async def close(self) -> None:
        """Close session-owned transports and background processes."""

        tracked_tasks = tuple(self._tracked_tasks)
        for task in tracked_tasks:
            task.cancel()
        await asyncio.gather(*tracked_tasks, return_exceptions=True)
        if self.hooks is not None:
            self.hooks.stop()
            await self.hooks.close()
        if self._mcp_mount is not None:
            await self._mcp_mount.close()
            self._mcp_mount = None
        await self.tool_registry.background_tasks.close()

    def session_start(self) -> None:
        if self.hooks is not None:
            self.hooks.session_start()

    async def _ensure_mcp_servers(self) -> None:
        if self._mcp_mount_attempted:
            return
        self._mcp_mount_attempted = True
        self._mcp_mount = await mount_mcp_servers(self.tool_registry)
        if self._provided_tool_schemas:
            existing = {schema.get("name") for schema in self.tool_schemas}
            self.tool_schemas.extend(
                schema for schema in self.tool_registry.schemas if schema.get("name") not in existing
            )
        else:
            self.tool_schemas = list(self.tool_registry.schemas)

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
        try:
            await self._ensure_mcp_servers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = _error_info(exc)
            self._persist_partial_with_cancelled_tools([], None, failure=error)
            yield StreamEvent(StreamEventType.AGENT_START)
            yield StreamEvent(StreamEventType.ERROR, error=error)
            yield StreamEvent(StreamEventType.AGENT_END)
            return
        yield StreamEvent(StreamEventType.AGENT_START)

        for turn_number in range(1, self.max_turns + 1):
            self.tool_registry.start_batch()
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
                    context_messages, self.tool_schemas
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
                    if event.message is not None and event.type is StreamEventType.MESSAGE_END:
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

            dispatch = dispatch_tool_calls(self, calls, _validated_tool_result)
            try:
                async for event in dispatch:
                    yield event
            finally:
                await dispatch.aclose()
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
            if (
                child_store is not None
                and (candidate is None or candidate.content == "tool execution canceled")
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
                child_store = self._agent_child_stores.pop(call.id, None)
                if child_store is not None:
                    child_store.finish_agent_parent()
                self.store.finish_agent_child(call.id)
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
            block.tool_call
            for block in blocks
            if isinstance(block, ToolUseContent)
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


def _assistant_text_snippet(message: Message) -> str:
    text = assistant_text(message).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= 160 else f"{text[:157]}..."
