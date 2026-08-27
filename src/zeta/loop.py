"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from .core.abort import AbortSignal as ToolAbortSignal
from .core.approval import ApprovalPolicy
from .core.context import ContextAssembler
from .core.hooks import HookManager
from .core.store import ConversationStore
from .mcp import MCPMount, mount_mcp_servers
from .prompts import load_identity
from .tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from .tools.agent import CHILD_TURN_CAP, ChildApprovalPolicy, agent_result
from .tools.registry import validate_tool_result
from .types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    StructuredToolResult,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    flatten_tool_content,
)


TaskResult = TypeVar("TaskResult")


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
    ) -> None:
        self.backend = backend
        self.store = store
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_child_stores: dict[str, ConversationStore] = {}
        self._agent_child_turns: dict[str, int] = {}
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
        self._mcp_mount_attempted = False
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
                        tool_result=ToolResult(
                            tool_call_id,
                            "tool execution canceled",
                            is_error=True,
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
    ) -> dict[str, object]:
        prompt = arguments.get("prompt")
        description = arguments.get("description")
        if type(prompt) is not str or not prompt.strip():
            return agent_result(
                "agent error: prompt must be a nonempty string",
                error=True,
                turns_used=0,
                child_session_path="",
            )
        if type(description) is not str or not description.strip():
            return agent_result(
                "agent error: description must be a nonempty string",
                error=True,
                turns_used=0,
                child_session_path="",
            )
        child_number = self.store.allocate_agent_index()
        agents_root = self.store.session_dir / "agents"
        child_store = ConversationStore(
            agents_root,
            session_id=str(child_number),
            cwd=self.store.cwd,
        )
        child_store.mark_agent_parent(tool_call.id)
        child_path = str(child_store.session_dir)
        child_instance_id = f"{self.store.session_id}:{child_number}"
        self._agent_child_stores[tool_call.id] = child_store
        self._agent_child_turns[tool_call.id] = 0
        if publisher is not None:
            publisher.set_metadata({"child_session_path": child_path})
        self.store.register_agent_child(
            tool_call,
            child_session_path=child_path,
            description=description,
        )
        child_registry = self.tool_registry.clone_for_session(
            child_store,
            exclude_names={"agent"},
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
            max_turns=CHILD_TURN_CAP,
            token_budget=self.context_assembler.token_budget,
            retained_tail=self.context_assembler.retained_tail,
            system_prompt=self.context_assembler.system_prompt,
        )

        def publish(status: str) -> None:
            if publisher is not None:
                publisher.publish(f"{description}: {status}\n", "stdout")

        turns_used = 0

        async def consume() -> dict[str, object]:
            nonlocal turns_used
            final_message: Message | None = None
            last_assistant_text = ""
            cap_hit = False
            error_message: str | None = None
            async for event in child_loop.run_turn(prompt):
                if event.type is StreamEventType.TURN_START:
                    turns_used = max(turns_used, int(event.data.get("turn", 0)))
                    publish(f"turn {turns_used}: thinking")
                elif event.type is StreamEventType.TOOL_APPROVAL_START:
                    name = event.tool_call.name if event.tool_call is not None else "tool"
                    publish(f"turn {turns_used}: approval pending: {name}")
                    if self.tool_registry._active_lifecycle_sink is not None:
                        self.tool_registry._active_lifecycle_sink(
                            "approval_start", event.tool_call
                        )
                elif event.type is StreamEventType.TOOL_APPROVAL_END:
                    if self.tool_registry._active_lifecycle_sink is not None:
                        self.tool_registry._active_lifecycle_sink(
                            "approval_end", event.tool_call
                        )
                elif event.type is StreamEventType.TOOL_EXECUTION_START:
                    name = event.tool_call.name if event.tool_call is not None else "tool"
                    arguments = (
                        event.tool_call.arguments
                        if event.tool_call is not None
                        else {}
                    )
                    summary = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                    publish(f"turn {turns_used}: tool: {name} {summary}")
                elif event.type is StreamEventType.TURN_END:
                    if event.message is not None:
                        last_assistant_text = _assistant_text_snippet(event.message)
                    if event.data.get("tool_calls") == 0 and event.message is not None:
                        final_message = event.message
                elif event.type is StreamEventType.ERROR and event.error is not None:
                    if event.error.code == "max_turns":
                        cap_hit = True
                    else:
                        error_message = event.error.message
            if cap_hit:
                return agent_result(
                    f"agent error: child reached the {CHILD_TURN_CAP}-turn cap; "
                    f"partial state is saved at {child_path}; "
                    f"last assistant text: {last_assistant_text or '[none]'}; "
                    f"turns used: {turns_used}",
                    error=True,
                    turns_used=turns_used,
                    child_session_path=child_path,
                )
            if error_message is not None:
                return agent_result(
                    f"agent error: {error_message}",
                    error=True,
                    turns_used=turns_used,
                    child_session_path=child_path,
                )
            if final_message is None:
                return agent_result(
                    "agent error: child ended without a final response",
                    error=True,
                    turns_used=turns_used,
                    child_session_path=child_path,
                )
            final_text = _assistant_text(final_message)
            if not final_text.strip():
                return agent_result(
                    "agent error: child returned an empty final assistant message",
                    error=True,
                    turns_used=turns_used,
                    child_session_path=child_path,
                )
            return agent_result(
                final_text,
                error=False,
                turns_used=turns_used,
                child_session_path=child_path,
            )

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
    ) -> AsyncIterator[StreamEvent]:
        return self._run_turn(user_text, user_message=user_message)

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
    ) -> AsyncIterator[StreamEvent]:
        if self.hooks is not None:
            self.hooks.user_prompt_submit(user_text)
        await self._ensure_mcp_servers()
        if user_message is None:
            user_message = Message(MessageRole.USER, [TextContent(user_text)])
        elif user_message.role is not MessageRole.USER:
            raise ValueError("user_message must have the user role")
        self.store.append_message(user_message)
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
            if self.context_assembler.needs_compaction():
                yield StreamEvent(
                    StreamEventType.COMPACTION_START,
                    data={"turn": turn_number},
                )
            context_messages = await self.context_assembler.assemble(backend=self.backend)
            context = self.context_assembler.last_context
            if context is not None and context.compacted:
                yield StreamEvent(
                    StreamEventType.COMPACTION_END,
                    data={
                        "turn": turn_number,
                        "token_count": context.token_count,
                    },
                )
            try:
                completion = self.backend.complete(
                    context_messages, self.tool_schemas
                )
                async for event in completion:
                    self.context_assembler.observe_event(event)
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial_blocks.append(event.content)
                        if event.delta is not None:
                            partial_blocks.append(TextContent(event.delta))
                    if event.message is not None and event.type is StreamEventType.MESSAGE_END:
                        assistant_message = event.message
                    if event.type is StreamEventType.MESSAGE_END:
                        completion_succeeded = True
                    yield event
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
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("backend_error", str(exc)),
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            cleanup_error = await _close_completion(completion)
            if _task_is_cancelling():
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise asyncio.CancelledError()
            if cleanup_error is not None:
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("backend_error", str(cleanup_error)),
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

            completed_tool_indexes: set[int] = set()
            # Cap advisory output at 128 events; final results stay complete.
            stream_updates: asyncio.Queue[StreamEvent] = asyncio.Queue(
                maxsize=128 + len(calls) * 3
            )

            def enqueue_tool_update(event: StreamEvent) -> None:
                retained: list[StreamEvent] = []
                while not stream_updates.empty():
                    retained.append(stream_updates.get_nowait())
                update_count = sum(
                    item.type is StreamEventType.TOOL_EXECUTION_UPDATE
                    for item in retained
                )
                if update_count >= 128:
                    first_update = next(
                        index
                        for index, item in enumerate(retained)
                        if item.type is StreamEventType.TOOL_EXECUTION_UPDATE
                    )
                    del retained[first_update]
                for queued in retained:
                    stream_updates.put_nowait(queued)
                stream_updates.put_nowait(event)

            def enqueue_tool_lifecycle(kind: str, tool_call: ToolCall) -> None:
                event_type = {
                    "approval_start": StreamEventType.TOOL_APPROVAL_START,
                    "approval_end": StreamEventType.TOOL_APPROVAL_END,
                    "execution_start": StreamEventType.TOOL_EXECUTION_START,
                }.get(kind)
                if event_type is not None:
                    stream_updates.put_nowait(
                        StreamEvent(event_type, tool_call=tool_call)
                    )

            active_task: asyncio.Task[StructuredToolResult] | None = None
            parallel_tasks: dict[
                asyncio.Task[StructuredToolResult], tuple[int, ToolCall]
            ] = {}
            parallel_results: list[ToolResult | None] = [None] * len(calls)
            try:
                call_index = 0
                while call_index < len(calls):
                    parallel_calls: list[ToolCall] = []
                    if self.tool_registry is not None:
                        definition = self.tool_registry.definitions_by_name.get(
                            calls[call_index].name
                        )
                        if definition is not None and definition.parallel_safe:
                            parallel_calls.append(calls[call_index])
                            while call_index + len(parallel_calls) < len(calls):
                                next_call = calls[call_index + len(parallel_calls)]
                                next_definition = self.tool_registry.definitions_by_name.get(
                                    next_call.name
                                )
                                if next_definition is None or not next_definition.parallel_safe:
                                    break
                                parallel_calls.append(next_call)
                    if len(parallel_calls) > 1:
                        parallel_tasks = {
                            self._create_task(
                                self.tool_registry.execute(
                                    tool_call,
                                    _stream_sink=enqueue_tool_update,
                                    _lifecycle_sink=(
                                        lambda kind, call=tool_call: enqueue_tool_lifecycle(
                                            kind, call
                                        )
                                    ),
                                )
                            ): (call_index + offset, tool_call)
                            for offset, tool_call in enumerate(parallel_calls)
                        }
                        while parallel_tasks:
                            update_task = self._create_task(stream_updates.get())
                            done, _ = await asyncio.wait(
                                (*parallel_tasks, update_task),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if update_task in done:
                                yield update_task.result()
                            else:
                                update_task.cancel()
                                await asyncio.gather(
                                    update_task,
                                    return_exceptions=True,
                                )
                            for task in done:
                                if task is update_task:
                                    continue
                                index, tool_call = parallel_tasks.pop(task)
                                parallel_results[index] = _validated_tool_result(
                                    task.result(),
                                    tool_call.id,
                                )
                        while not stream_updates.empty():
                            yield stream_updates.get_nowait()
                        batch_end = call_index + len(parallel_calls)
                        results = self._finalize_tool_results(
                            parallel_calls,
                            parallel_results[call_index:batch_end],
                        )
                        parallel_results[call_index:batch_end] = [None] * len(parallel_calls)
                        completed_tool_indexes.update(range(call_index, batch_end))
                        for tool_call, result in zip(parallel_calls, results, strict=True):
                            yield StreamEvent(
                                StreamEventType.TOOL_EXECUTION_END,
                                tool_call=tool_call,
                                tool_result=result,
                            )
                        call_index += len(parallel_calls)
                        continue

                    tool_call = calls[call_index]
                    active_task = self._create_task(
                        self.tool_registry.execute(
                            tool_call,
                            _stream_sink=enqueue_tool_update,
                            _lifecycle_sink=(
                                lambda kind, call=tool_call: enqueue_tool_lifecycle(
                                    kind, call
                                )
                            ),
                        )
                    )
                    try:
                        while True:
                            update_task = self._create_task(stream_updates.get())
                            done, _ = await asyncio.wait(
                                (active_task, update_task),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if update_task in done:
                                yield update_task.result()
                            else:
                                update_task.cancel()
                                await asyncio.gather(
                                    update_task,
                                    return_exceptions=True,
                                )
                            if active_task in done:
                                while not stream_updates.empty():
                                    yield stream_updates.get_nowait()
                                break
                        result = active_task.result()
                        active_task = None
                    except Exception as exc:
                        result = ToolResult(tool_call.id, str(exc), is_error=True)
                    result = _validated_tool_result(result, tool_call.id)
                    result = self._finalize_tool_results([tool_call], [result])[0]
                    completed_tool_indexes.add(call_index)
                    yield StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                    call_index += 1
            except (asyncio.CancelledError, GeneratorExit):
                pending_tasks: list[asyncio.Task[StructuredToolResult]] = []
                await asyncio.sleep(0)
                if active_task is not None and not active_task.done():
                    active_task.cancel()
                    pending_tasks.append(active_task)
                for task, (index, tool_call) in list(parallel_tasks.items()):
                    parallel_tasks.pop(task)
                    if task.done():
                        if task.cancelled():
                            continue
                        try:
                            result = _validated_tool_result(task.result(), tool_call.id)
                        except BaseException:
                            continue
                        parallel_results[index] = result
                    else:
                        task.cancel()
                        pending_tasks.append(task)
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                pending_indexes = [
                    index for index in range(len(calls)) if index not in completed_tool_indexes
                ]
                self._finalize_tool_results(
                    [calls[index] for index in pending_indexes],
                    [parallel_results[index] for index in pending_indexes],
                )
                raise
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
                result = ToolResult(
                    call.id,
                    "tool execution canceled",
                    is_error=True,
                    structured_content={
                        "turns_used": self._agent_child_turns.get(call.id, 0),
                        "child_session_path": str(child_store.session_dir),
                    },
                )
            if result is None:
                result = slot or ToolResult(
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
        return results

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        if assistant_message is not None:
            self.store.append_message(_durable_message(assistant_message))
        else:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent) or block.signature
            ]
            if durable_blocks:
                self.store.append_message(Message(MessageRole.ASSISTANT, durable_blocks))

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
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message)
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


def _assistant_text(message: Message) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def _assistant_text_snippet(message: Message) -> str:
    text = _assistant_text(message).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= 160 else f"{text[:157]}..."
