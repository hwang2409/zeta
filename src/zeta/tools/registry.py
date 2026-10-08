"""Tool registration and MCP-compatible execution results.

Text blocks always include ``truncated`` and ``full_size``. ``full_size`` is
the original UTF-8 byte length before a character cap is applied. Fetch pages
also include ``full_size_chars`` for their readable-text character length.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import pkgutil
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.receipt import MIN_AGENT_RECEIPT_BYTES
from ..config.tool_policy import ToolPolicy, parse_tool_selector
from ..core.abort import AbortGenerationRegistry
from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRule,
    ApprovedPathExecution,
    parse_approval_rule,
)
from ..core.approval import canceled_result as _canceled_result
from ..core.store import ConversationStore
from ..media.image_policy import ANTHROPIC_IMAGE_POLICY, ImagePolicy
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
from ._action_metadata import (
    ApprovalBinding,
    InvalidActionArguments,
    ResolvedCapability,
    ToolAction,
    UnknownToolAction,
    normalize_actions,
    resolve_capability,
)
from ._action_metadata import (
    resolve_approval_rule as resolve_registered_approval_rule,
)
from ._results import (
    _apply_error_governance,
    _BoundedText,  # noqa: F401 - preserve the registry import
    _error_result,
    _legacy_result,
    _normalize_result_async,
    _success_result,
    text_block,
)
from ._shared.process import BackgroundTaskRegistry
from ._shared.sandbox import SandboxPolicy
from ._spill import SpillStore
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
    approval_subject_resolver: Callable[[Mapping[str, object]], str | None] | None = None
    actions: Mapping[str, ToolAction] | None = None

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


def _open_directory_fd(path: Path) -> tuple[int, tuple[int, int]]:
    """Open a directory descriptor for a session cwd, rejecting symlinks."""

    cwd_fd = -1
    try:
        cwd_fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        cwd_stat = os.fstat(cwd_fd)
    except OSError as exc:
        if cwd_fd >= 0:
            os.close(cwd_fd)
        raise ValueError(
            f"tool cwd is not a directory or is a symlink: {path}"
        ) from exc
    return cwd_fd, (cwd_stat.st_dev, cwd_stat.st_ino)


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
        project_id: str | None = None,
        project_registry: Any = None,
        inbox_enabled: bool = True,
        tool_allow: Sequence[str] | None = None,
        tool_deny: Sequence[str] = (),
        tool_allow_layers: Sequence[Sequence[str]] = (),
        required_tool_names: Sequence[str] | None = None,
        image_policy: ImagePolicy = ANTHROPIC_IMAGE_POLICY,
    ) -> None:
        """Create a registry with a shared tool-output limit.

        ``max_output_chars`` bounds ordinary tool results. Terminal agent
        receipts require room for their persisted envelope and therefore use
        ``max(max_output_chars, 1000)`` instead.
        """

        if enforce_approvals and approval_policy is None:
            raise ValueError("enforced approvals require a policy")
        self.image_policy = image_policy
        self.tool_policy = ToolPolicy.create(
            tool_allow, tool_deny, allow_layers=tool_allow_layers
        )
        self._required_tool_names = (
            self.tool_policy.required_exact_names
            if required_tool_names is None
            else tuple(dict.fromkeys(required_tool_names))
        )
        self.enforce_approvals = enforce_approvals
        # Deliberately shared by session clones so child denials reach the run record.
        self.denied_tools: list[str] = []
        self.cwd = Path(os.path.abspath(os.fspath(Path(cwd).expanduser())))
        self.project_id = project_id
        # Concrete capability captured at composition time; tools must never
        # rediscover it through ambient ZETA_HOME.
        self.project_registry = project_registry
        self.inbox_enabled = inbox_enabled
        cwd_fd, self._cwd_identity = _open_directory_fd(self.cwd)
        self._cwd_fd = cwd_fd
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
        self._todo_store = session_store
        self._agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None = None
        self.spills = SpillStore(
            session_dir=session_store.session_dir
            if session_store is not None
            else None,
            directory_fd=session_store.directory_fd
            if session_store is not None
            else None,
        )
        self.policy = SandboxPolicy(
            self.cwd,
            allow_outside=not self.tool_policy.restricted,
        )
        # Set by AgentLoop; copied into child session clones.  Kept optional so
        # registries used by standalone tool tests remain valid.
        self._agent_owner: Any = None
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
        # Set by MCPMount.  It is deliberately copied by clone_for_session so
        # discovery remains available in child sessions without sharing tools.
        self._mcp_mount: Any = None
        self._mcp_excluded_names: frozenset[str] = frozenset()
        # MCP definitions are owned per registry so reconnects cannot remove
        # unrelated custom tools or stale definitions in another session.
        self._mcp_owned: dict[str, tuple[object, int]] = {}
        self._mcp_hidden: set[str] = set()
        self._closed = False
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self.skill_catalog = skill_catalog
        if agent_catalog is None:
            from ..skills.agent_catalog import discover_packaged_agents

            agent_catalog = discover_packaged_agents()
        self.agent_catalog = agent_catalog
        self._register_builtin = register_builtin
        if register_builtin:
            _register_discovered_tools(self)

    @property
    def tool_allow(self) -> tuple[str, ...] | None:
        return self.tool_policy.allow

    @property
    def tool_deny(self) -> tuple[str, ...]:
        return self.tool_policy.deny

    def tool_is_allowed(self, name: str) -> bool:
        definition = self._tools.get(name)
        actions = tuple(definition.actions) if definition and definition.actions else None
        return self.tool_policy.allows_tool(name, actions)

    @property
    def missing_required_tools(self) -> tuple[str, ...]:
        registered = self.registered_names
        missing: list[str] = []
        for selector_text in self._required_tool_names:
            selector = parse_tool_selector(selector_text)
            definition = self._tools.get(selector.name)
            if definition is None or selector.name not in registered:
                missing.append(selector_text)
                continue
            if selector.action is not None and (
                definition.actions is None
                or selector.action not in definition.actions
                or not self.tool_policy.allows_call(
                    ResolvedCapability(
                        selector.name,
                        selector.action,
                        False,
                        None,
                        None,
                        ApprovalBinding.NONE,
                        None,
                    )
                )
            ):
                missing.append(selector_text)
        return tuple(missing)

    @property
    def schemas(self) -> list[ToolSchema]:
        schemas: list[ToolSchema] = []
        for name, definition in self._tools.items():
            if name in self._mcp_hidden or not self.tool_is_allowed(name):
                continue
            schema = definition.schema()
            if definition.actions is not None:
                filtered = self.tool_policy.filter_schema(schema)
                if filtered is None:
                    continue
                schema = filtered
            schemas.append(schema)
        return schemas

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return self.schemas

    def allowed_schemas(
        self, schemas: Sequence[ToolSchema]
    ) -> list[ToolSchema]:
        """Filter an external schema snapshot through this registry's policy."""

        allowed: list[ToolSchema] = []
        for schema in schemas:
            name = schema.get("name")
            if not isinstance(name, str) or not self.tool_is_allowed(name):
                continue
            definition = self._tools.get(name)
            if definition is not None and definition.actions is not None:
                filtered = self.tool_policy.filter_schema(schema)
                if filtered is not None:
                    allowed.append(filtered)
            else:
                allowed.append(copy.deepcopy(schema))
        return allowed

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            _copy_definition(definition)
            for name, definition in self._tools.items()
            if self.tool_is_allowed(name)
        )

    @property
    def definitions_by_name(self) -> Mapping[str, ToolDefinition]:
        return {
            name: _copy_definition(definition)
            for name, definition in self._tools.items()
            if self.tool_is_allowed(name)
        }

    @property
    def registered_names(self) -> frozenset[str]:
        """Return the current tool names without copying definitions."""

        return frozenset(
            name for name in self._tools if self.tool_is_allowed(name)
        )

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
        actions: Mapping[str, ToolAction] | None = None,
        handler_factory: ToolHandlerFactory | None = None,
        approval_subject: str | None = None,
        approval_subject_resolver: Callable[[Mapping[str, object]], str | None] | None = None,
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
        normalized_actions = normalize_actions(name, normalized, actions)
        if normalized_actions is not None and approval_subject is not None:
            raise ValueError(
                "action tools declare approval subjects in their action metadata"
            )
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
        binding_actions = (
            ()
            if normalized_actions is None
            else tuple(
                action
                for action, metadata in normalized_actions.items()
                if metadata.binding is not ApprovalBinding.NONE
            )
        )
        if approval_subject == "path" or binding_actions:
            try:
                accepts_execution_context = (
                    "execution_context" in inspect.signature(handler).parameters
                )
            except (TypeError, ValueError):
                accepts_execution_context = False
            if not accepts_execution_context:
                capability = (
                    f"path approval tool {name!r}"
                    if not binding_actions
                    else f"bound approval action {name}({binding_actions[0]})"
                )
                raise ValueError(f"{capability} must accept execution_context")
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
            approval_subject_resolver=approval_subject_resolver,
            actions=normalized_actions,
        )
        self._tools[name] = definition
        if self.approval_policy is not None:
            self.approval_policy.bind_rule_resolver(self.resolve_approval_rule)
            self.approval_policy.bind_capability_resolver(
                self.resolve_tool_call_capability
            )
        return _copy_definition(definition)

    register_tool = register

    def resolve_call(
        self, name: str, arguments: Mapping[str, object]
    ) -> ResolvedCapability:
        """Validate one call and resolve all registry-owned authorization facts."""

        definition = self._tools.get(name)
        if definition is None:
            raise KeyError(name)
        return resolve_capability(definition, arguments)

    def resolve_tool_call_capability(
        self, tool_call: ToolCall
    ) -> ResolvedCapability:
        """Resolve one persisted or live call through the registry seam."""

        return self.resolve_call(tool_call.name, tool_call.arguments)

    def call_allowed_by(self, tool_call: ToolCall, policy: ToolPolicy) -> bool:
        """Resolve and authorize one call through the supplied policy layer."""

        try:
            capability = self.resolve_tool_call_capability(tool_call)
        except (KeyError, TypeError, ValueError):
            return False
        return policy.allows_call(capability)

    def resolve_approval_rule(
        self, rule: str | ApprovalRule
    ) -> ApprovalRule:
        """Parse and validate one approval rule against registered metadata."""

        parsed = rule if isinstance(rule, ApprovalRule) else parse_approval_rule(rule)
        definition = self._tools.get(parsed.tool)
        if definition is None:
            return parsed
        return resolve_registered_approval_rule(definition, parsed)

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
        self._mcp_owned.pop(name, None)

    def register_mcp(
        self,
        name: str,
        handler: ToolHandler,
        *,
        owner: object,
        generation: int,
        **kwargs: Any,
    ) -> bool:
        """Register one actor-owned MCP definition without replacing collisions."""
        if not self.tool_is_allowed(name) or name in self._tools:
            return False
        self.register(name, handler, **kwargs)
        self._mcp_hidden.discard(name)
        self._mcp_owned[name] = (owner, generation)
        return True

    def is_mcp_owned(self, name: str, owner: object, generation: int) -> bool:
        """Return whether an MCP definition has this exact owner and generation."""
        return self._mcp_owned.get(name) == (owner, generation)

    def mcp_owned_names(self, owner: object) -> set[str]:
        """Return every registered name currently owned by ``owner``."""
        return {
            name
            for name, (current_owner, _generation) in self._mcp_owned.items()
            if current_owner is owner
        }

    def unregister_mcp_owner(self, owner: object) -> None:
        """Remove only definitions currently owned by ``owner``."""
        for name, (current_owner, _generation) in tuple(self._mcp_owned.items()):
            if current_owner is owner:
                self._tools.pop(name, None)
                self._mcp_owned.pop(name, None)
                self._mcp_hidden.discard(name)

    def hide_mcp_owner(self, owner: object) -> None:
        for name, (current_owner, _generation) in self._mcp_owned.items():
            if current_owner is owner:
                self._mcp_hidden.add(name)

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
        cwd: str | Path | None = None,
        image_policy: ImagePolicy | None = None,
    ) -> ToolRegistry:
        clone = copy.copy(self)
        if image_policy is not None:
            clone.image_policy = image_policy
        if cwd is not None:
            # Re-anchor file tools and sandbox resolution to the child's cwd.
            resolved = Path(os.path.abspath(os.fspath(Path(cwd).expanduser())))
            new_fd, clone._cwd_identity = _open_directory_fd(resolved)
            clone.cwd = resolved
            clone._cwd_fd = new_fd
            clone._cwd_finalizer = weakref.finalize(clone, os.close, new_fd)
        else:
            clone._cwd_fd = os.dup(self._cwd_fd)
            clone._cwd_finalizer = weakref.finalize(clone, os.close, clone._cwd_fd)
        clone._cleanup_callbacks = []
        clone._mcp_excluded_names = frozenset(exclude_names)
        clone._tools = {
            name: _copy_definition(definition, clone)
            for name, definition in self._tools.items()
            if name not in exclude_names
        }
        clone._mcp_hidden = self._mcp_hidden.intersection(clone._tools)
        clone._mcp_owned = {
            name: ownership
            for name, ownership in self._mcp_owned.items()
            if name in clone._tools
        }
        for owner, _generation in clone._mcp_owned.values():
            register_registry = getattr(owner, "register_registry", None)
            if callable(register_registry):
                register_registry(clone)
        clone._session_store = store
        clone._todo_store = store
        clone.spills = SpillStore(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
        )
        clone.policy = SandboxPolicy(
            clone.cwd,
            allow_outside=not clone.tool_policy.restricted,
        )
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
        clone.project_id = self.project_id
        clone.project_registry = self.project_registry
        return clone

    def abort(self) -> None:
        self.abort_signal.abort()

    def start_batch(self) -> None:
        """Rotate the active signal before a new tool batch."""
        self.abort_signal = self._abort_registry.new_generation()

    def bind_approval_store(self, store: ConversationStore) -> None:
        if self.approval_policy is not None:
            self.approval_policy.bind_store(store)
            self.approval_policy.bind_capability_resolver(
                self.resolve_tool_call_capability
            )
            bind_display = getattr(self.approval_policy, "bind_display_resolver", None)
            if bind_display is not None:
                bind_display(self._approval_display)

    def _approval_display(self, request: ApprovalRequest) -> ApprovalRequest:
        if request.tool_call.name != "project_update":
            return request
        arguments = request.tool_call.arguments
        project = None
        if self.project_registry is not None and self.project_id is not None:
            try:
                project = self.project_registry.show_project(self.project_id)
            except (OSError, ValueError):
                pass
        name = arguments.get("name")
        content = arguments.get("content")
        if not isinstance(name, str) or not isinstance(content, str):
            return request
        return replace(
            request,
            project_id=self.project_id,
            project_name=getattr(project, "name", None),
            filename=name,
            content_bytes=len(content.encode("utf-8")),
            preview=content[:240].replace("\n", "\\n"),
        )

    def approval_display(self, tool_call: ToolCall) -> ApprovalRequest:
        """Resolve the harness-owned display facts for a tool call.

        This is the single trusted display used by every frontend: it derives
        project id, name, filename, byte size, and a bounded preview from the
        bound project registry, never from provider arguments a caller could
        spoof.  Callers must render these fields rather than the raw arguments.
        """
        definition = self._tools.get(tool_call.name)
        action = None
        if definition is not None:
            try:
                action = self.resolve_call(
                    tool_call.name, tool_call.arguments
                ).action
            except (KeyError, UnknownToolAction):
                pass
        return self._approval_display(
            ApprovalRequest(tool_call.id, tool_call, action=action)
        )

    def bind_session_store(self, store: ConversationStore) -> None:
        self._session_store = store
        if self._todo_store is None:
            self._todo_store = store
        self.spills.close()
        self.spills = SpillStore(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
        )
        self.policy = SandboxPolicy(
            self.cwd,
            allow_outside=not self.tool_policy.restricted,
        )
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

    async def close(self) -> tuple[str, ...]:
        """Stop session-owned resources and background processes.

        Returns the ids of background tasks killed by this close so callers can
        surface them (for example in a child agent's completion receipt).
        """

        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        try:
            for callback in callbacks:
                await callback()
        finally:
            try:
                killed = await self.background_tasks.close()
            finally:
                self.spills.close()
                self._closed = True
                # Closing is a lifecycle boundary: detach from every MCP owner so a
                # later reconnect cannot republish stale definitions into this
                # closed registry, and drop the actor-owned definitions it holds.
                owners = {owner for owner, _generation in self._mcp_owned.values()}
                for owner in owners:
                    unregister = getattr(owner, "unregister_registry", None)
                    if callable(unregister):
                        unregister(self)
                for name in self._mcp_owned:
                    self._tools.pop(name, None)
                self._mcp_owned.clear()
                self._mcp_hidden.clear()
        return killed

    def add_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._cleanup_callbacks.append(callback)

    def set_pre_execute_hook(self, hook: ToolHook | None) -> None:
        self.pre_execute_hook = hook
        self._approval_gate.hook = hook

    def update_bash_cwd(self, cwd: str) -> None:
        if self._session_store is not None:
            self._session_store.set_bash_cwd(cwd)
        self.bash_cwd = cwd

    def set_approval_subject_resolver(
        self, tool: str, resolver: Callable[[Mapping[str, object]], str | None]
    ) -> None:
        definition = self._tools.get(tool)
        if definition is None:
            raise KeyError(tool)
        self._tools[tool] = replace(
            definition, approval_subject_resolver=resolver
        )
        if self.approval_policy is not None:
            self.approval_policy.bind_rule_resolver(self.resolve_approval_rule)

    def set_approval_policy(self, policy: ApprovalPolicy | None) -> None:
        if self.enforce_approvals and policy is None:
            raise ValueError("cannot remove an enforced approval policy")
        self.approval_policy = policy
        self._approval_gate.policy = policy
        if policy is not None:
            policy.bind_rule_resolver(self.resolve_approval_rule)
            policy.bind_capability_resolver(self.resolve_tool_call_capability)

    def prepare_approval(self, tool_call: ToolCall) -> ApprovalRequest | None:
        if self.approval_policy is None:
            return None
        if not self.tool_is_allowed(tool_call.name):
            self.abort_approval(tool_call)
            return None
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self.abort_approval(tool_call)
            return None
        try:
            capability = self.resolve_call(tool_call.name, tool_call.arguments)
        except (KeyError, UnknownToolAction, InvalidActionArguments):
            self.abort_approval(tool_call)
            return None
        if not capability.requires_approval and not self.enforce_approvals:
            return None
        if definition.validate_arguments:
            try:
                _validate_arguments(tool_call.arguments, definition.parameters)
            except (AttributeError, KeyError, TypeError, ValueError):
                self.abort_approval(tool_call)
                return None
        request = self.approval_policy.prepare(
            tool_call,
            capability=capability,
        )
        if request is not None:
            request = self._approval_display(request)
        return request

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

        async def finalize(result: StructuredToolResult) -> StructuredToolResult:
            structured = result.get("structuredContent")
            terminal_agent = (
                tool_call.name.casefold() == "agent"
                and not (
                    isinstance(structured, Mapping)
                    and structured.get("status") == "running"
                )
            )
            output_limit = (
                max(self.max_output_chars, MIN_AGENT_RECEIPT_BYTES)
                if terminal_agent
                else self.max_output_chars
            )
            normalized = await _normalize_result_async(
                result,
                output_limit,
                spill=lambda texts: self.spills.awrite_text_group(
                    tool_call.name,
                    tool_call.id,
                    texts,
                ),
            )
            return _apply_error_governance(normalized, tool_call.name)

        if _boundary_signal is not None and _signal_is_set(_boundary_signal):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, _boundary_signal, _scope_signal
            )
            if abort_result is not None:
                return await finalize(abort_result)
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return await finalize(abort_result)
        if not self.tool_is_allowed(tool_call.name):
            self.abort_approval(tool_call)
            return await finalize(
                _error_result(
                    f"tool not allowed by session tool policy: {tool_call.name}",
                    kind="tool_not_allowed",
                )
            )
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self.abort_approval(tool_call)
            return await finalize(
                _error_result(
                    f"unknown tool: {tool_call.name}",
                    kind="unknown_tool",
                )
            )
        try:
            capability = self.resolve_call(tool_call.name, tool_call.arguments)
        except UnknownToolAction as exc:
            self.abort_approval(tool_call)
            return await finalize(
                _error_result(str(exc), kind="invalid_tool_action")
            )
        except InvalidActionArguments as exc:
            self.abort_approval(tool_call)
            return await finalize(
                _error_result(str(exc), kind="invalid_arguments")
            )
        if not self.tool_policy.allows_call(capability):
            self.abort_approval(tool_call)
            capability_name = (
                tool_call.name
                if capability.action is None
                else f"{tool_call.name}({capability.action})"
            )
            return await finalize(
                _error_result(
                    f"tool action not allowed by session tool policy: {capability_name}",
                    kind="tool_action_not_allowed",
                )
            )
        try:
            arguments = (
                _validate_arguments(tool_call.arguments, definition.parameters)
                if definition.validate_arguments
                else _coerce_arguments(tool_call.arguments)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.abort_approval(tool_call)
            return await finalize(
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
                return await finalize(abort_result)
        execution_token = uuid.uuid4().hex
        cleanup_binding = getattr(
            self.approval_policy, "cleanup_execution_binding", None
        )
        try:
            gate_result, execution_signal = await self._approval_gate.run(
                tool_call,
                arguments,
                signal_state,
                lambda current: self._next_abort_generation(current, _scope_signal),
                _lifecycle_sink,
                execution_token=execution_token,
                skip_approval=(
                    not self.enforce_approvals
                    and (
                        _skip_approval
                        or not capability.requires_approval
                    )
                ),
                persist_request=_persist_approval,
                capability=capability,
            )
            approved_execution = None
            if self.approval_policy is not None:
                consume_binding = getattr(
                    self.approval_policy, "consume_execution_binding", None
                )
                if callable(consume_binding):
                    approved_execution = consume_binding(execution_token)
            if gate_result is not None:
                if (
                    self.enforce_approvals
                    and gate_result.content.startswith("tool execution denied")
                ):
                    self.denied_tools.append(tool_call.name)
                return await finalize(_legacy_result(gate_result))
        finally:
            if callable(cleanup_binding):
                cleanup_binding(execution_token)
        if _scope_signal is not None:
            execution_signal = _scope_signal
            if execution_signal.is_set():
                return await finalize(_legacy_result(_canceled_result(tool_call.id)))
        if _lifecycle_sink is not None:
            _lifecycle_sink("execution_start")
        stream_publisher = (
            _ToolCallStreamPublisher(tool_call, execution_signal, _stream_sink)
            if _stream_sink is not None
            else None
        )
        execution_context = ToolExecutionContext(
            tool_call,
            self._agent_runner,
            _lifecycle_sink,
            approved_execution,
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
        if (
            isinstance(approved_execution, ApprovedPathExecution)
            and not execution_context.path_binding_consumed
        ):
            result = ToolResult(
                tool_call.id,
                "approved path binding was not consumed by the shared opener",
                True,
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
        return await finalize(normalized_result)

    def abort_approval(self, tool_call: ToolCall) -> ApprovalDecision | None:
        """Abort an unresolved approval without replacing a concurrent decision."""
        if self.approval_policy is None:
            return None
        try:
            capability = self.resolve_call(tool_call.name, tool_call.arguments)
        except (KeyError, UnknownToolAction, InvalidActionArguments, TypeError, ValueError):
            try:
                self.approval_policy.abort(tool_call.id)
            except RuntimeError:
                pass
            return None
        try:
            return self.approval_policy.abort_or_winner(
                tool_call.id, capability=capability
            )
        except RuntimeError:
            return None

    def _arbitrate_abort(
        self,
        tool_call: ToolCall,
        signal_state: ToolAbortSignal,
        scope_signal: ToolAbortSignal | None = None,
    ) -> tuple[ToolAbortSignal, StructuredToolResult | None]:
        winner = self.abort_approval(tool_call)
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


    def _path(self, raw_path: object) -> Path:
        return self.policy.resolve(raw_path).absolute

    def verify_cwd_identity(self) -> None:
        """Fail closed if the pathname no longer names the captured cwd."""

        cwd_fd = self._open_cwd()
        os.close(cwd_fd)

    @staticmethod
    def open_verified_directory(path: str | Path, identity: tuple[int, int]) -> int:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            stat_result = os.fstat(fd)
        except OSError as exc:
            if "fd" in locals():
                os.close(fd)
            raise ValueError("approved shell cwd was replaced") from exc
        if (stat_result.st_dev, stat_result.st_ino) != identity:
            os.close(fd)
            raise ValueError("approved shell cwd was replaced")
        return fd

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
