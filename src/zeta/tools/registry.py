"""Tool registration and MCP-compatible execution results.

Text blocks always include ``truncated`` and ``full_size``. ``full_size`` is
the original UTF-8 byte length before a character cap is applied. Fetch pages
also include ``full_size_chars`` for their readable-text character length.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import math
import os
import pkgutil
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

from ..core.abort import AbortGenerationRegistry
from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRequest,
)
from ..core.approval import canceled_result as _canceled_result
from ..core.store import ConversationStore
from ..types import (
    StreamEvent,
    StreamEventType,
    ToolContentBlock,
    StructuredContentValue,
    StructuredToolResult,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolTextBlock,
    validate_tool_content_block,
)
from ._process import BackgroundTaskRegistry

AbortSignal = ToolAbortSignal
MAX_STRUCTURED_CONTENT_DEPTH = 32
ToolHook = Callable[[str, dict[str, Any]], bool | str | Awaitable[bool | str] | None]
ToolHandlerResult = str | StructuredToolResult | ToolResult
ToolHandler = Callable[..., ToolHandlerResult | Awaitable[ToolHandlerResult]]
ToolHandlerFactory = Callable[["ToolRegistry"], ToolHandler]
ToolStream = Literal["stdout", "stderr"]

def _bind_handler(handler: ToolHandler, registry: ToolRegistry) -> ToolHandler:
    return partial(handler, registry)


def _discover_tool_modules() -> list[str]:
    package = importlib.import_module(__package__)
    modules = (
        f"{package.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(package.__path__)
        if not module_info.name.startswith("_")
    )
    return sorted(modules, key=lambda name: (name.endswith(".agent"), name))


def _register_discovered_tools(registry: ToolRegistry) -> None:
    """Load modules with ``register(registry)``; underscore modules are helpers."""

    for module_name in _discover_tool_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - identify broken modules clearly
            raise RuntimeError(
                f"failed to load tool module {module_name}: {exc}"
            ) from exc
        if not hasattr(module, "register"):
            continue
        register = module.register
        if not callable(register):
            raise TypeError(
                f"tool module {module_name} has a non-callable register contract"
            )
        try:
            result = register(registry)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("register must be synchronous")
        except Exception as exc:  # noqa: BLE001 - name malformed modules clearly
            raise RuntimeError(
                f"failed to register tool module {module_name}: {exc}"
            ) from exc


class ToolStreamPublisher(Protocol):
    """Publish advisory output for one tool call without changing its result."""

    def publish(self, text: str, stream: ToolStream) -> None:
        """Publish one output chunk."""

    def set_metadata(self, metadata: Mapping[str, object]) -> None: ...

ToolStreamSink = Callable[[StreamEvent], None]
ToolLifecycleSink = Callable[..., None]


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Per-call state passed to handlers that need execution ownership."""

    tool_call: ToolCall
    agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None
    lifecycle_sink: ToolLifecycleSink | None


@dataclass(slots=True)
class _ToolCallStreamPublisher:
    """Adapt handler chunks into non-blocking events for the agent loop."""

    tool_call: ToolCall
    abort_signal: ToolAbortSignal
    sink: ToolStreamSink
    closed: bool = False
    metadata: dict[str, object] = dataclass_field(default_factory=dict)

    def set_metadata(self, metadata: Mapping[str, object]) -> None: self.metadata.update(metadata)

    def publish(self, text: str, stream: ToolStream) -> None:
        if self.closed or _signal_is_set(self.abort_signal):
            return
        self.sink(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_UPDATE,
                tool_call=self.tool_call,
                delta=text,
                data={"stream": stream, **self.metadata},
            )
        )

    def close(self) -> None:
        self.closed = True


class _ToolCanceled(Exception):
    pass


def _validate_unique_tool_call_ids(tool_calls: Sequence[ToolCall]) -> None:
    call_ids = [tool_call.id for tool_call in tool_calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("duplicate tool call id in one execution batch")


async def _yield_for_abort(
    abort_signal: ToolAbortSignal,
) -> None:
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()
    await asyncio.sleep(0)
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()


class _BoundedText:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._parts: list[str] = []
        self._length = 0
        self._full_size = 0
        self._has_line = False
        self.truncated = False

    @property
    def retained_chars(self) -> int:
        return self._length

    @property
    def full_size(self) -> int:
        return self._full_size

    def append(self, value: str) -> None:
        self._full_size += len(value.encode("utf-8"))
        remaining = self.limit - self._length
        if remaining > 0:
            retained = value[:remaining]
            self._parts.append(retained)
            self._length += len(retained)
        if len(value) > remaining:
            self.truncated = True

    def append_captured(self, value: str, full_size: int) -> None:
        self.append(value)
        self._full_size += max(0, full_size - len(value.encode("utf-8")))

    def begin_line(self) -> None:
        if self._has_line:
            self.append("\n")
        self._has_line = True

    def append_line(self, value: str) -> None:
        self.begin_line()
        self.append(value)

    def render(self, *, full_size: int | None = None) -> ToolTextBlock:
        return text_block(
            "".join(self._parts),
            full_size=self.full_size if full_size is None else full_size,
        )


def text_block(
    text: str,
    *,
    cap: int | None = None,
    full_size: int | None = None,
) -> ToolTextBlock:
    """Build a text block and expose any output cap to the caller."""

    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("text block cap must be a positive integer")
    if full_size is not None and (type(full_size) is not int or full_size < 0):
        raise ValueError("text block full_size must be a nonnegative integer")
    original_size = len(text.encode("utf-8")) if full_size is None else full_size
    shown = text if cap is None else text[:cap]
    truncated = shown != text or original_size > len(shown.encode("utf-8"))
    return {
        "type": "text",
        "text": shown,
        "truncated": truncated,
        "full_size": original_size,
    }


def _success_result(
    block: ToolTextBlock,
    *,
    structured_content: Mapping[str, StructuredContentValue] | None = None,
) -> StructuredToolResult:
    normalized_content = (
        None
        if structured_content is None
        else dict(structured_content)
    )
    return {
        "content": [block],
        "isError": False,
        "structuredContent": normalized_content,
    }


def _error_result(message: str) -> StructuredToolResult:
    return {
        "content": [text_block(message)],
        "isError": True,
        "structuredContent": None,
    }


def _legacy_result(result: ToolResult) -> StructuredToolResult:
    if type(result.content) is not str:
        return _error_result("invalid tool result: content")
    blocks = (
        result.content_blocks
        if result.content_blocks is not None
        else [text_block(result.content)]
    )
    try:
        return validate_tool_result(
            {
                "content": blocks,
                "isError": result.is_error,
                "structuredContent": None,
            }
        )
    except ValueError as exc:
        return _error_result(f"invalid tool result: {exc}")


def _normalize_result(
    result: StructuredToolResult,
    max_output_chars: int,
) -> StructuredToolResult:
    content: list[ToolContentBlock] = []
    remaining = max_output_chars
    for block in result["content"]:
        if block["type"] != "text":
            content.append(block)
            continue
        full_size = block["full_size"]
        shown = block["text"][:remaining]
        normalized = text_block(shown, full_size=full_size)
        if "annotations" in block:
            normalized["annotations"] = block["annotations"]
        normalized["truncated"] = block["truncated"] or shown != block["text"]
        if "full_size_chars" in block:
            normalized["full_size_chars"] = block["full_size_chars"]
        if "next_offset" in block:
            normalized["next_offset"] = block["next_offset"]
        remaining -= len(shown)
        content.append(normalized)
    return {**result, "content": content}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    handler_factory: ToolHandlerFactory | None = None
    parallel_safe: bool = False
    validate_arguments: bool = True
    requires_approval: bool = True

    def schema(self) -> ToolSchema:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }


def _copy_definition(definition: ToolDefinition, registry: ToolRegistry | None = None) -> ToolDefinition:
    handler = (
        definition.handler_factory(registry)
        if registry is not None and definition.handler_factory is not None
        else definition.handler
    )
    return replace(definition, parameters=copy.deepcopy(definition.parameters), handler=handler)


class ToolRegistry:
    """One provider-neutral registry for built-in and custom tools."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        pre_execute_hook: ToolHook | None = None,
        hook: ToolHook | None = None,
        abort_signal: ToolAbortSignal | None = None,
        approval_policy: ApprovalPolicy | None = None,
        approval_store: ConversationStore | None = None,
        session_store: ConversationStore | None = None,
        max_output_chars: int = 10_000,
        register_builtin: bool = True,
    ) -> None:
        self.cwd = Path(os.path.abspath(os.fspath(Path(cwd).expanduser())))
        cwd_fd = -1
        try:
            cwd_fd = os.open(
                self.cwd,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            cwd_stat = os.fstat(cwd_fd)
        except OSError as exc:
            if cwd_fd >= 0:
                os.close(cwd_fd)
            raise ValueError(
                f"tool cwd is not a directory or is a symlink: {self.cwd}"
            ) from exc
        self._cwd_fd = cwd_fd
        self._cwd_identity = (cwd_stat.st_dev, cwd_stat.st_ino)
        self._cwd_finalizer = weakref.finalize(self, os.close, cwd_fd)
        if type(max_output_chars) is not int or max_output_chars < 1:
            raise ValueError("max_output_chars must be a positive integer")
        if pre_execute_hook is not None and hook is not None:
            raise ValueError("pass only one pre-execution hook")
        self.pre_execute_hook = pre_execute_hook or hook
        if abort_signal is None:
            self._abort_registry = AbortGenerationRegistry()
            self.abort_signal = self._abort_registry.new_generation()
        else:
            self.abort_signal = abort_signal
            self._abort_registry = abort_signal.registry
        self.approval_policy = approval_policy
        self._approval_gate = ApprovalGate(self.approval_policy, self.pre_execute_hook)
        if self.approval_policy is not None and approval_store is not None:
            self.approval_policy.bind_store(approval_store)
        self.max_output_chars = max_output_chars
        self._session_store = session_store
        self._agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None = None
        self.background_tasks = BackgroundTaskRegistry(
            session_dir=session_store.session_dir if session_store is not None else None,
        )
        self.bash_cwd = (
            session_store.bash_cwd if session_store is not None else str(self.cwd)
        )
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtin = register_builtin
        if register_builtin:
            _register_discovered_tools(self)

    @property
    def schemas(self) -> list[ToolSchema]:
        return [definition.schema() for definition in self._tools.values()]

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return self.schemas

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(_copy_definition(definition) for definition in self._tools.values())

    @property
    def definitions_by_name(self) -> Mapping[str, ToolDefinition]:
        return {
            name: _copy_definition(definition)
            for name, definition in self._tools.items()
        }

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
        input_schema: Mapping[str, Any] | None = None,
        schema: Mapping[str, Any] | None = None,
        parallel_safe: bool = False,
        validate_arguments: bool = True,
        requires_approval: bool = True,
        handler_factory: ToolHandlerFactory | None = None,
    ) -> ToolDefinition:
        if type(name) is not str or not name:
            raise ValueError("tool name must be a nonempty string")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        supplied_schemas = [
            candidate
            for candidate in (parameters, input_schema, schema)
            if candidate is not None
        ]
        if len(supplied_schemas) > 1:
            raise ValueError("pass only one tool parameter schema")
        normalized = _normalize_schema(
            supplied_schemas[0] if supplied_schemas else None,
            validate_definition=validate_arguments,
        )
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=normalized,
            handler=handler,
            handler_factory=handler_factory,
            parallel_safe=parallel_safe,
            validate_arguments=validate_arguments,
            requires_approval=requires_approval,
        )
        self._tools[name] = definition
        return _copy_definition(definition)

    register_tool = register

    def register_session_tool(self, name: str, handler: ToolHandler, **kwargs: Any) -> ToolDefinition:
        return self.register(name, _bind_handler(handler, self), handler_factory=partial(_bind_handler, handler), **kwargs)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    @property
    def agent_runner(self) -> Callable[..., Awaitable[ToolHandlerResult]] | None:
        return self._agent_runner

    def set_agent_runner(
        self,
        runner: Callable[..., Awaitable[ToolHandlerResult]] | None,
    ) -> None:
        self._agent_runner = runner

    def clone_for_session(
        self,
        store: ConversationStore,
        *,
        exclude_names: set[str] | frozenset[str] = frozenset(),
    ) -> ToolRegistry:
        clone = copy.copy(self)
        clone._cwd_fd = os.dup(self._cwd_fd)
        clone._cwd_finalizer = weakref.finalize(clone, os.close, clone._cwd_fd)
        clone._tools = {
            name: _copy_definition(definition, clone)
            for name, definition in self._tools.items()
            if name not in exclude_names
        }
        clone._session_store = store
        clone.background_tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
        )
        clone.bash_cwd = store.bash_cwd
        clone.abort_signal = clone._abort_registry.new_generation()
        clone._approval_gate = ApprovalGate(
            clone.approval_policy,
            clone.pre_execute_hook,
        )
        clone._agent_runner = None
        return clone
    def abort(self) -> None:
        self.abort_signal.abort()

    def start_batch(self) -> None:
        """Rotate the active signal before a new tool batch."""
        self.abort_signal = self._abort_registry.new_generation()

    def bind_approval_store(self, store: ConversationStore) -> None:
        if self.approval_policy is not None:
            self.approval_policy.bind_store(store)

    def bind_session_store(self, store: ConversationStore) -> None:
        self._session_store = store
        self.background_tasks.bind_session_dir(store.session_dir)
        self.bash_cwd = store.bash_cwd

    @property
    def session_store(self) -> ConversationStore:
        if self._session_store is None:
            raise ValueError("tool requires a bound session store")
        return self._session_store

    async def close(self) -> None:
        """Stop session-owned background processes."""

        await self.background_tasks.close()

    def set_pre_execute_hook(self, hook: ToolHook | None) -> None:
        self.pre_execute_hook = hook
        self._approval_gate.hook = hook

    def update_bash_cwd(self, cwd: str) -> None:
        if self._session_store is not None:
            self._session_store.set_bash_cwd(cwd)
        self.bash_cwd = cwd

    def set_approval_policy(self, policy: ApprovalPolicy | None) -> None:
        self.approval_policy = policy
        self._approval_gate.policy = policy

    def prepare_approval(self, tool_call: ToolCall) -> ApprovalRequest | None:
        if self.approval_policy is None:
            return None
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return None
        if not definition.requires_approval:
            return None
        if definition.validate_arguments:
            try:
                _validate_arguments(tool_call.arguments, definition.parameters)
            except (AttributeError, KeyError, TypeError, ValueError):
                self._abort_approval(tool_call)
                return None
        return self.approval_policy.prepare(tool_call)

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        abort_signal: ToolAbortSignal | None = None,
        _scope_signal: ToolAbortSignal | None = None,
        _boundary_signal: ToolAbortSignal | None = None,
        _stream_sink: ToolStreamSink | None = None,
        _lifecycle_sink: ToolLifecycleSink | None = None,
        _persist_approval: bool = True,
        _log_path: str | Path | None = None,
        _background: bool = False, _capture_output: bool = False, _skip_approval: bool = False,
    ) -> StructuredToolResult:
        signal_state = abort_signal or self.abort_signal
        if _boundary_signal is not None and _signal_is_set(_boundary_signal):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, _boundary_signal, _scope_signal
            )
            if abort_result is not None:
                return _normalize_result(abort_result, self.max_output_chars)
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return _normalize_result(abort_result, self.max_output_chars)
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return _normalize_result(
                _error_result(f"unknown tool: {tool_call.name}"),
                self.max_output_chars,
            )
        try:
            arguments = (
                _validate_arguments(tool_call.arguments, definition.parameters)
                if definition.validate_arguments
                else _coerce_arguments(tool_call.arguments)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._abort_approval(tool_call)
            return _normalize_result(
                _error_result(f"invalid arguments: {exc}"),
                self.max_output_chars,
            )
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return _normalize_result(abort_result, self.max_output_chars)
        gate_result, execution_signal = await self._approval_gate.run(
            tool_call,
            arguments,
            signal_state,
            lambda current: self._next_abort_generation(current, _scope_signal),
            _lifecycle_sink,
            skip_approval=_skip_approval or not definition.requires_approval,
            persist_request=_persist_approval,
        )
        if gate_result is not None:
            return _normalize_result(
                _legacy_result(gate_result), self.max_output_chars
            )
        if _scope_signal is not None:
            execution_signal = _scope_signal
            if execution_signal.is_set():
                return _normalize_result(
                    _legacy_result(_canceled_result(tool_call.id)),
                    self.max_output_chars,
                )
        if _lifecycle_sink is not None:
            _lifecycle_sink("execution_start")
        stream_publisher = (
            _ToolCallStreamPublisher(tool_call, execution_signal, _stream_sink)
            if _stream_sink is not None
            else None
        )
        execution_context = ToolExecutionContext(tool_call, self._agent_runner, _lifecycle_sink)
        handler = _bind_execution_context(definition.handler, execution_context)
        execution_arguments = dict(arguments, **({"_log_path": str(_log_path)} if _log_path is not None else {}) | ({"_background": True} if _background else {}) | ({"_capture_output": True} if _capture_output else {}))
        result = await self._invoke_handler_with_abort(
            handler,
            execution_arguments,
            execution_signal,
            stream_publisher,
            tool_call.id,
        )
        if isinstance(result, ToolResult):
            if result.tool_call_id != tool_call.id:
                normalized_result = _error_result(
                    f"tool result id mismatch: expected {tool_call.id}, "
                    f"got {result.tool_call_id}"
                )
            else:
                normalized_result = _legacy_result(result)
        elif isinstance(result, Mapping):
            try:
                normalized_result = validate_tool_result(result)
            except ValueError as exc:
                normalized_result = _error_result(f"invalid tool handler result: {exc}")
        elif isinstance(result, str):
            normalized_result = _success_result(text_block(result))
        else:
            normalized_result = _error_result(
                "invalid tool handler result: expected str or structured tool result"
            )
        return _normalize_result(normalized_result, self.max_output_chars)

    async def _invoke_handler_with_abort(
        self,
        handler: ToolHandler,
        arguments: dict[str, Any],
        execution_signal: ToolAbortSignal,
        stream_publisher: ToolStreamPublisher | None,
        tool_call_id: str,
    ) -> ToolHandlerResult:
        current = asyncio.current_task()

        async def cancel_on_abort() -> None:
            await execution_signal.wait()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if current is not None and not current.done():
                current.cancel()

        abort_wait = asyncio.create_task(cancel_on_abort())
        try:
            try:
                return await _invoke_handler(
                    handler,
                    arguments,
                    execution_signal,
                    stream_publisher,
                )
            except _ToolCanceled:
                return _legacy_result(_canceled_result(tool_call_id))
            except asyncio.CancelledError:
                if execution_signal.is_set():
                    return _legacy_result(_canceled_result(tool_call_id))
                raise
            except Exception as exc:  # noqa: BLE001 - tool handlers must fail closed
                return _error_result(str(exc))
        finally:
            if stream_publisher is not None:
                stream_publisher.close()
            if not abort_wait.done():
                abort_wait.cancel()
                await asyncio.gather(abort_wait, return_exceptions=True)

    def _abort_approval(self, tool_call: ToolCall) -> ApprovalDecision | None:
        if self.approval_policy is None:
            return None
        try:
            return self.approval_policy.abort_or_winner(tool_call.id)
        except RuntimeError:
            return None

    def _arbitrate_abort(
        self,
        tool_call: ToolCall,
        signal_state: ToolAbortSignal,
        scope_signal: ToolAbortSignal | None = None,
    ) -> tuple[ToolAbortSignal, StructuredToolResult | None]:
        winner = self._abort_approval(tool_call)
        if winner is ApprovalDecision.ALLOW:
            signal_state = self._next_abort_generation(signal_state, scope_signal)
            if _signal_is_set(signal_state):
                return signal_state, _legacy_result(_canceled_result(tool_call.id))
            return signal_state, None
        if winner is ApprovalDecision.DENY:
            return signal_state, _error_result("tool execution denied")
        return signal_state, _legacy_result(_canceled_result(tool_call.id))

    def _next_abort_generation(
        self,
        signal_state: ToolAbortSignal,
        scope_signal: ToolAbortSignal | None = None,
    ) -> ToolAbortSignal:
        if scope_signal is not None:
            return scope_signal
        if self.abort_signal is signal_state:
            self.abort_signal = self._abort_registry.new_generation()
        return self.abort_signal

    async def execute_many(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        abort_signal: ToolAbortSignal | None = None,
    ) -> list[StructuredToolResult]:
        """Execute calls with safe contiguous groups in parallel, preserving order."""

        _validate_unique_tool_call_ids(tool_calls)
        parent_signal = abort_signal or self.abort_signal
        boundary_signal = parent_signal if _signal_is_set(parent_signal) else None
        scope_signal = parent_signal
        if abort_signal is None:
            scope_signal = self._abort_registry.new_generation()
            self.abort_signal = scope_signal
        results: list[StructuredToolResult | None] = [None] * len(tool_calls)
        index = 0
        while index < len(tool_calls):
            definition = self._tools.get(tool_calls[index].name)
            if definition is None or not definition.parallel_safe:
                results[index] = await self.execute(
                    tool_calls[index],
                    abort_signal=scope_signal,
                    _scope_signal=scope_signal,
                    _boundary_signal=boundary_signal,
                )
                index += 1
                continue
            end = index + 1
            while end < len(tool_calls):
                next_definition = self._tools.get(tool_calls[end].name)
                if next_definition is None or not next_definition.parallel_safe:
                    break
                end += 1
            group = await asyncio.gather(
                *(
                    self.execute(
                        call,
                        abort_signal=scope_signal,
                        _scope_signal=scope_signal,
                        _boundary_signal=boundary_signal,
                    )
                    for call in tool_calls[index:end]
                )
            )
            results[index:end] = group
            index = end
        return [result for result in results if result is not None]

    def _path(self, raw_path: object) -> Path:
        if type(raw_path) is not str or not raw_path:
            raise ValueError("path must be a nonempty string")
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else self.cwd / candidate

    def _open_cwd(self) -> int:
        try:
            cwd_fd = os.open(
                self.cwd,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise ValueError("session cwd was replaced") from exc
        try:
            cwd_stat = os.fstat(cwd_fd)
        except OSError as exc:
            os.close(cwd_fd)
            raise ValueError("session cwd was replaced") from exc
        if (cwd_stat.st_dev, cwd_stat.st_ino) != self._cwd_identity:
            os.close(cwd_fd)
            raise ValueError("session cwd was replaced")
        return cwd_fd


async def _invoke_handler(
    handler: ToolHandler,
    arguments: dict[str, Any],
    abort_signal: ToolAbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
) -> ToolHandlerResult:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        result = handler(arguments, abort_signal)
    else:
        if stream_publisher is not None:
            try:
                signature.bind(arguments, abort_signal, stream_publisher)
            except TypeError:
                try:
                    signature.bind(
                        arguments,
                        abort_signal=abort_signal,
                        stream_publisher=stream_publisher,
                    )
                except TypeError:
                    try:
                        signature.bind(arguments, abort_signal, stream_publisher=stream_publisher)
                    except TypeError:
                        result = _invoke_handler_without_stream(handler, signature, arguments, abort_signal)
                    else:
                        result = handler(
                            arguments,
                            abort_signal,
                            stream_publisher=stream_publisher,
                        )
                else:
                    result = handler(
                        arguments,
                        abort_signal=abort_signal,
                        stream_publisher=stream_publisher,
                    )
            else:
                result = handler(arguments, abort_signal, stream_publisher)
        else:
            result = _invoke_handler_without_stream(handler, signature, arguments, abort_signal)
    if inspect.isawaitable(result):
        result = await result
    return result


def _invoke_handler_without_stream(
    handler: ToolHandler,
    signature: inspect.Signature,
    arguments: dict[str, Any],
    abort_signal: ToolAbortSignal,
) -> ToolHandlerResult | Awaitable[ToolHandlerResult]:
    try:
        signature.bind(arguments, abort_signal)
    except TypeError:
        try:
            signature.bind(arguments, abort_signal=abort_signal)
        except TypeError:
            signature.bind(arguments)
            return handler(arguments)
        return handler(arguments, abort_signal=abort_signal)
    return handler(arguments, abort_signal)


def _bind_execution_context(
    handler: ToolHandler,
    context: ToolExecutionContext,
) -> ToolHandler:
    try:
        inspect.signature(handler).parameters["execution_context"]
    except (KeyError, TypeError, ValueError):
        return handler
    return partial(handler, execution_context=context)


def validate_tool_result(result: object) -> StructuredToolResult:
    """Validate one complete MCP-compatible structured tool result."""

    if type(result) is not dict:
        raise ValueError("expected a structured result object")
    if any(type(key) is not str for key in result):
        raise ValueError("top-level keys must be strings")
    expected_keys = {"content", "isError", "structuredContent"}
    result_keys = set(result)
    if "content_blocks" in result_keys:
        raise ValueError("legacy content_blocks is not allowed")
    missing_keys = expected_keys - result_keys
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"missing top-level keys: {missing}")
    extra_keys = result_keys - expected_keys
    if extra_keys:
        extra = ", ".join(sorted(extra_keys))
        raise ValueError(f"unexpected top-level keys: {extra}")

    content = result["content"]
    if type(content) is not list:
        raise ValueError("content must be an array")
    is_error = result["isError"]
    if type(is_error) is not bool:
        raise ValueError("isError must be a boolean")
    structured_content = result["structuredContent"]
    if structured_content is not None:
        if type(structured_content) is not dict:
            raise ValueError("structuredContent must be an object or null")
        _validate_structured_content(structured_content)

    normalized_content = [
        validate_tool_content_block(index, block)
        for index, block in enumerate(content)
    ]
    return {**result, "content": normalized_content}


def _validate_structured_content(value: object) -> None:
    pending: list[tuple[object, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while pending:
        current, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(current))
            continue
        if depth > MAX_STRUCTURED_CONTENT_DEPTH:
            raise ValueError(
                f"structuredContent depth > {MAX_STRUCTURED_CONTENT_DEPTH}"
            )
        if current is None or type(current) in {str, int, bool}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("structuredContent must contain finite numbers")
            continue
        if type(current) not in {list, dict}:
            raise ValueError("structuredContent must contain JSON values")
        current_id = id(current)
        if current_id in active:
            raise ValueError("cyclic structuredContent")
        active.add(current_id)
        if type(current) is list:
            pending.append((current, depth, True))
            pending.extend((item, depth + 1, False) for item in reversed(current))
            continue
        items = list(current.items())
        pending.append((current, depth, True))
        for key, item in reversed(items):
            if type(key) is not str:
                raise ValueError("structuredContent object keys must be strings")
            pending.append((item, depth + 1, False))


def _normalize_schema(
    schema: Mapping[str, Any] | None,
    *,
    validate_definition: bool = True,
) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise TypeError("tool parameter schema must be an object")
    try:
        normalized = copy.deepcopy(dict(schema))
    except Exception as exc:
        raise ValueError("schema must contain JSON data") from exc
    if validate_definition:
        _validate_schema_definition(normalized, "schema")
    _validate_json_data(normalized, "schema")
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("schema must contain JSON data") from exc
    return normalized


def _validate_json_data(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, float, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_data(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"schema must contain JSON data at {path}")
            _validate_json_data(item, f"{path}.{key}")
        return
    raise ValueError(f"schema must contain JSON data at {path}")


def _validate_arguments(arguments: object, schema: Mapping[str, Any]) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    _validate_finite_numbers(arguments, "arguments")
    _validate_schema(arguments, schema, "arguments")
    return dict(arguments)


def _coerce_arguments(arguments: object) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    _validate_finite_numbers(arguments, "arguments")
    return dict(arguments)


def _validate_finite_numbers(value: Any, path: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_numbers(child, f"{path}[{index}]")


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise ValueError(f"{path} must be {expected_type}")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if "const" in schema and not _schema_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal the declared constant")
    if "enum" in schema and not any(
        _schema_equal(value, option) for option in schema["enum"]
    ):
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
    if type(value) in {int, float} and type(value) is not bool:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path} is not above the exclusive minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above the maximum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} has unexpected properties: {', '.join(extra)}")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")


def _matches_type(value: Any, expected: object) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in {int, float}
    if expected == "null":
        return value is None
    return False


def _schema_equal(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            set(left) == set(right)
            and all(_schema_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _schema_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


_SCHEMA_KEYS = {
    "description",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minLength",
    "maxLength",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "minItems",
    "maxItems",
    "enum",
    "const",
}
_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def _validate_schema_definition(schema: Mapping[str, Any], path: str) -> None:
    unsupported = set(schema) - _SCHEMA_KEYS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported schema keywords at {path}: {names}")
    expected_type = schema.get("type")
    if expected_type is not None and (
        type(expected_type) is not str or expected_type not in _SCHEMA_TYPES
    ):
        raise ValueError(f"unsupported schema type at {path}")
    description = schema.get("description")
    if description is not None and type(description) is not str:
        raise ValueError(f"schema description must be a string at {path}")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError(f"schema properties must be an object at {path}")
        for name, child in properties.items():
            if type(name) is not str or not isinstance(child, Mapping):
                raise ValueError(f"invalid schema property at {path}")
            _validate_schema_definition(child, f"{path}.{name}")
    required = schema.get("required")
    if required is not None and (
        type(required) is not list or any(type(name) is not str for name in required)
    ):
        raise ValueError(f"schema required must be a string array at {path}")
    additional = schema.get("additionalProperties")
    if additional is not None and type(additional) is not bool:
        raise ValueError(f"schema additionalProperties must be boolean at {path}")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"schema items must be an object at {path}")
        _validate_schema_definition(items, f"{path}.items")
    enum = schema.get("enum")
    if enum is not None and type(enum) is not list:
        raise ValueError(f"schema enum must be an array at {path}")
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    ):
        value = schema.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"schema {key} must be a nonnegative integer at {path}")
    for key in ("minimum", "exclusiveMinimum", "maximum"):
        value = schema.get(key)
        if value is not None and (type(value) not in {int, float} or type(value) is bool):
            raise ValueError(f"schema {key} must be numeric at {path}")
    if expected_type == "object" and schema.get("items") is not None:
        raise ValueError(f"schema items is not valid for an object at {path}")
    if expected_type != "object" and schema.get("properties") is not None:
        raise ValueError(f"schema properties is only valid for an object at {path}")
    if expected_type != "array" and schema.get("items") is not None:
        raise ValueError(f"schema items is only valid for an array at {path}")


def _signal_is_set(signal_state: ToolAbortSignal) -> bool:
    return signal_state.is_set()
