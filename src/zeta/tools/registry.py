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
import os
import pkgutil
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from ..protocol.types import (
    StructuredToolResult,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from ..runtime.execution import (
    ToolExecutionContext,
    ToolHandler,
    ToolHandlerResult,
    ToolLifecycleSink,
    ToolStream,  # noqa: F401 - preserve the public registry import
    ToolStreamPublisher,  # noqa: F401 - preserve the public registry import
    ToolStreamSink,
    _signal_is_set,
    _ToolCallStreamPublisher,
    _ToolCanceled,  # noqa: F401 - preserve the shell tool's import
    _yield_for_abort,  # noqa: F401 - preserve the read tool's import
    bind_execution_context,
    build_execution_arguments,
    run_handler_with_abort,
)
from ..skills import SkillCatalog
from ._results import (
    _apply_error_governance,
    _BoundedText,  # noqa: F401 - preserve the registry import
    _error_result,
    _legacy_result,
    _normalize_result,
    _success_result,
    text_block,
)
from ._shared.process import BackgroundTaskRegistry
from ._shared.sandbox import SandboxPolicy
from ._validation import (
    MAX_STRUCTURED_CONTENT_DEPTH,  # noqa: F401 - preserve the registry import
    _coerce_arguments,
    _normalize_schema,
    _validate_arguments,
    validate_tool_result,
)

if TYPE_CHECKING:
    from ..skills.agent_catalog import AgentCatalog

AbortSignal = ToolAbortSignal
ToolHook = Callable[[str, dict[str, Any]], bool | str | Awaitable[bool | str] | None]
ToolHandlerFactory = Callable[["ToolRegistry"], ToolHandler]


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
        except Exception as exc:
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
        except Exception as exc:
            raise RuntimeError(
                f"failed to register tool module {module_name}: {exc}"
            ) from exc


def _validate_unique_tool_call_ids(tool_calls: Sequence[ToolCall]) -> None:
    call_ids = [tool_call.id for tool_call in tool_calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("duplicate tool call id in one execution batch")


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
    # Argument that ``tool(pattern)`` approval rules match against (ZETA-86).
    approval_subject: str | None = None

    def schema(self) -> ToolSchema:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }


def _copy_definition(
    definition: ToolDefinition, registry: ToolRegistry | None = None
) -> ToolDefinition:
    handler = (
        definition.handler_factory(registry)
        if registry is not None and definition.handler_factory is not None
        else definition.handler
    )
    return replace(
        definition, parameters=copy.deepcopy(definition.parameters), handler=handler
    )


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
        enforce_approvals: bool = False,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog | None = None,
    ) -> None:
        if enforce_approvals and approval_policy is None:
            raise ValueError("enforced approvals require a policy")
        self.enforce_approvals = enforce_approvals
        # Deliberately shared by session clones so child denials reach the run record.
        self.denied_tools: list[str] = []
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
        self.policy = SandboxPolicy(self.cwd)
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
        self._todo_store = session_store
        self._agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None = None
        self.background_tasks = BackgroundTaskRegistry(
            session_dir=session_store.session_dir
            if session_store is not None
            else None,
            directory_fd=session_store.directory_fd
            if session_store is not None
            else None,
        )
        self.bash_cwd = (
            session_store.bash_cwd if session_store is not None else str(self.cwd)
        )
        self._tools: dict[str, ToolDefinition] = {}
        self.skill_catalog = skill_catalog
        if agent_catalog is None:
            from ..skills.agent_catalog import discover_packaged_agents

            agent_catalog = discover_packaged_agents()
        self.agent_catalog = agent_catalog
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
        return tuple(
            _copy_definition(definition) for definition in self._tools.values()
        )

    @property
    def definitions_by_name(self) -> Mapping[str, ToolDefinition]:
        return {
            name: _copy_definition(definition)
            for name, definition in self._tools.items()
        }

    @property
    def registered_names(self) -> frozenset[str]:
        """Return the current tool names without copying definitions."""

        return frozenset(self._tools)

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
        approval_subject: str | None = None,
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
        properties = normalized.get("properties")
        if approval_subject is not None and (
            type(approval_subject) is not str
            or not approval_subject
            or (
                isinstance(properties, Mapping)
                and properties
                and approval_subject not in properties
            )
        ):
            raise ValueError(
                f"approval_subject {approval_subject!r} must name a parameter of tool {name!r}"
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
            approval_subject=approval_subject,
        )
        self._tools[name] = definition
        if self.approval_policy is not None:
            self.approval_policy.declare_subjects({name: approval_subject})
        return _copy_definition(definition)

    register_tool = register

    def register_session_tool(
        self, name: str, handler: ToolHandler, **kwargs: Any
    ) -> ToolDefinition:
        return self.register(
            name,
            _bind_handler(handler, self),
            handler_factory=partial(_bind_handler, handler),
            **kwargs,
        )

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
        clone._todo_store = store
        clone.background_tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
        )
        clone.bash_cwd = store.bash_cwd
        clone.abort_signal = clone._abort_registry.new_generation()
        clone._approval_gate = ApprovalGate(
            clone.approval_policy,
            clone.pre_execute_hook,
        )
        clone._agent_runner = None
        clone.agent_catalog = self.agent_catalog
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
        if self._todo_store is None:
            self._todo_store = store
        self.background_tasks.bind_session_dir(store.session_dir, store.directory_fd)
        self.bash_cwd = store.bash_cwd

    @property
    def session_store(self) -> ConversationStore:
        if self._session_store is None:
            raise ValueError("tool requires a bound session store")
        return self._session_store

    @property
    def todo_store(self) -> ConversationStore:
        if self._todo_store is None:
            raise ValueError("todo tool requires a bound session store")
        return self._todo_store

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
        if self.enforce_approvals and policy is None:
            raise ValueError("cannot remove an enforced approval policy")
        self.approval_policy = policy
        self._approval_gate.policy = policy
        if policy is not None:  # tell the policy which argument scopes each tool
            policy.declare_subjects(
                {name: tool.approval_subject for name, tool in self._tools.items()}
            )

    def prepare_approval(self, tool_call: ToolCall) -> ApprovalRequest | None:
        if self.approval_policy is None:
            return None
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return None
        if not definition.requires_approval and not self.enforce_approvals:
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
        _background: bool = False,
        _capture_output: bool = False,
        _skip_approval: bool = False,
    ) -> StructuredToolResult:
        signal_state = abort_signal or self.abort_signal

        def finalize(result: StructuredToolResult) -> StructuredToolResult:
            normalized = _normalize_result(result, self.max_output_chars)
            return _apply_error_governance(normalized, tool_call.name)

        if _boundary_signal is not None and _signal_is_set(_boundary_signal):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, _boundary_signal, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return finalize(
                _error_result(
                    f"unknown tool: {tool_call.name}",
                    kind="unknown_tool",
                )
            )
        try:
            arguments = (
                _validate_arguments(tool_call.arguments, definition.parameters)
                if definition.validate_arguments
                else _coerce_arguments(tool_call.arguments)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._abort_approval(tool_call)
            return finalize(
                _error_result(
                    f"invalid arguments: {exc}",
                    kind="invalid_arguments",
                )
            )
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        gate_result, execution_signal = await self._approval_gate.run(
            tool_call,
            arguments,
            signal_state,
            lambda current: self._next_abort_generation(current, _scope_signal),
            _lifecycle_sink,
            skip_approval=(
                not self.enforce_approvals
                and (_skip_approval or not definition.requires_approval)
            ),
            persist_request=_persist_approval,
        )
        if gate_result is not None:
            if (
                self.enforce_approvals
                and gate_result.content == "tool execution denied"
            ):
                self.denied_tools.append(tool_call.name)
            return finalize(_legacy_result(gate_result))
        if _scope_signal is not None:
            execution_signal = _scope_signal
            if execution_signal.is_set():
                return finalize(_legacy_result(_canceled_result(tool_call.id)))
        if _lifecycle_sink is not None:
            _lifecycle_sink("execution_start")
        stream_publisher = (
            _ToolCallStreamPublisher(tool_call, execution_signal, _stream_sink)
            if _stream_sink is not None
            else None
        )
        execution_context = ToolExecutionContext(
            tool_call, self._agent_runner, _lifecycle_sink
        )
        handler = bind_execution_context(definition.handler, execution_context)
        execution_arguments = build_execution_arguments(
            arguments,
            log_path=_log_path,
            background=_background,
            capture_output=_capture_output,
        )
        result = await run_handler_with_abort(
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
                    f"got {result.tool_call_id}",
                    kind="invalid_result",
                )
            else:
                normalized_result = _legacy_result(result)
        elif isinstance(result, Mapping):
            try:
                normalized_result = validate_tool_result(result)
            except ValueError as exc:
                normalized_result = _error_result(
                    f"invalid tool handler result: {exc}",
                    kind="invalid_result",
                )
        elif isinstance(result, str):
            normalized_result = _success_result(text_block(result))
        else:
            normalized_result = _error_result(
                "invalid tool handler result: expected str or structured tool result",
                kind="invalid_result",
            )
        return finalize(normalized_result)

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
            return signal_state, _error_result("tool execution denied", kind="denied")
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
        return self.policy.resolve(raw_path).absolute

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
