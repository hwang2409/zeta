"""The session loop's per-code-call admission and settlement."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from ...agent.tool_results import validated_tool_result
from ...core.abort import AbortSignal
from ...core.tool_dispatch import execute_owned_tool
from ...core.tool_dispatch_coordinator import ToolDispatchCoordinator
from ...protocol.types import (
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from ...tools.registry import ToolOrigin
from ..execution import NestedToolParent

if TYPE_CHECKING:
    from ...tools.registry import ToolDefinition, ToolRegistry
    from .agent import AgentLoop


CODE_MODE_ALLOWED_NAMES: Final = frozenset({
    "read", "write", "edit", "bash", "fetch", "websearch", "project",
    "recall_history", "run_background", "task_output",
})


def authority_snapshot(registry: ToolRegistry, schemas: Sequence[ToolSchema]) -> Mapping[str, ToolDefinition]:
    return MappingProxyType({
        name: definition
        for schema in schemas
        if (definition := registry.definition_reference(name := schema["name"])) is not None
        and definition.origin is ToolOrigin.PACKAGED_BUILTIN
    })


def describe_code(registry: ToolRegistry, schemas: Sequence[ToolSchema]) -> None:
    code = next((schema for schema in schemas if schema["name"] == "code"), None)
    definition = registry.definition_reference("code")
    if code is None or definition is None or definition.origin is not ToolOrigin.PACKAGED_BUILTIN:
        return
    names = ", ".join(sorted(CODE_MODE_ALLOWED_NAMES.intersection(authority_snapshot(registry, schemas)))) or "none"
    code["description"] = (
        "Run JavaScript to combine results from several tools. Pass {script: string}. "
        "Use await tools.call(name, args) or await tools[name](args). "
        f"Available tools: {names}. "
        "Only output(value), called once, sends JSON to the model. "
        "Inner approvals still apply; catch ToolDenied or ToolError. "
        "Use Promise.all for independent calls (up to 8 run at once). "
        "Use at most 50 calls, 64 KiB of script and 60 seconds of active JavaScript. "
        "No filesystem, network, clocks, timers or persistent state except through tools. "
        "Use ordinary calls for one tool or shell commands that already return selective output."
    )


class LoopNestedToolExecutor:
    def __init__(self, loop: AgentLoop, outer: ToolCall, abort: AbortSignal,
                 publish: Callable[[StreamEvent], None], environment: Mapping[str, str] | None,
                 *, coordinator: ToolDispatchCoordinator) -> None:
        self.loop = loop
        self.outer = outer
        self.abort = abort
        self.publish = publish
        self.environment = environment
        registry = loop.tool_registry
        self.authority = authority_snapshot(registry, loop._active_tool_schemas())
        self.exposed_names = CODE_MODE_ALLOWED_NAMES.intersection(self.authority)
        self._seen: set[str] = set()
        self._coordinator = coordinator

    async def __call__(self, call: ToolCall, parent: NestedToolParent) -> ToolResult:
        call_id = f"{self.outer.id}:code:{parent.call_index}"
        if parent.call_id != self.outer.id or call.id != call_id or not 1 <= parent.call_index <= 50:
            return ToolResult(call.id, "invalid nested call identity", True)
        if call.id in self._seen:
            return ToolResult(call.id, "duplicate_tool_call", True)
        self._seen.add(call.id)
        expected = self.authority.get(call.name)
        async with self._coordinator.slot(parallel_safe=bool(expected and expected.parallel_safe)):
            if self.abort.is_set():
                return ToolResult(call.id, "tool execution canceled", True, is_canceled=True)
            definition = self.loop.tool_registry.definition_reference(call.name)
            refusal = f"tool {call.name!r} is not exposed to code mode"
            if (call.name in CODE_MODE_ALLOWED_NAMES and definition is not None
                    and definition.origin is not ToolOrigin.PACKAGED_BUILTIN):
                return ToolResult(call.id, f"{refusal}: active definition is not a packaged built-in", True)
            if (call.name not in CODE_MODE_ALLOWED_NAMES or definition is None
                    or self.authority.get(call.name) is not definition):
                return ToolResult(call.id, refusal, True)
            return await self._run(call, parent)

    async def _run(self, call: ToolCall, parent: NestedToolParent) -> ToolResult:
        loop = self.loop
        registry = loop.tool_registry
        data = {"code_parent_id": parent.call_id, "code_call_index": parent.call_index}
        publication_error = None

        def publish(event: StreamEvent) -> None:
            nonlocal publication_error
            try:
                self.publish(replace(event, data={**(event.data or {}), **data}))
            except Exception as exc:  # noqa: BLE001 - settle first, report the first publication error
                if publication_error is None:
                    publication_error = exc

        def lifecycle(kind: str, payload: dict[str, object] | None = None) -> None:
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is not None:
                publish(StreamEvent(event_type, tool_call=call, data=payload or {}))

        async def settle(result: ToolResult) -> ToolResult:
            if result.is_canceled:
                await registry.abort_approval(call)
            if loop.store.approval_state_now(call.id) is not None:
                return loop._finalize_tool_results([call], [result])[0]
            registry.take_approval_origin(call.id)
            loop._run_tool_post_effects(call, result)
            return result

        canceled = False
        try:
            result = await execute_owned_tool(
                loop, call, validated_tool_result, abort_signal=self.abort, _scope_signal=self.abort,
                _host_environment=self.environment, _expected_definition=self.authority[call.name],
                _stream_sink=publish, _lifecycle_sink=lifecycle, _lifecycle_payload_sink=lifecycle,
                _retain_approval_origin=True, _pending_execution_owner=asyncio.current_task(),
            )
        except asyncio.CancelledError:
            canceled = True
            result = ToolResult(call.id, "tool execution canceled", True, is_canceled=True)
        settlement = asyncio.create_task(settle(result))
        while not settlement.done():
            try:
                await asyncio.shield(settlement)
            except asyncio.CancelledError:
                canceled = True
        result = settlement.result()
        publish(StreamEvent(StreamEventType.TOOL_EXECUTION_END, tool_call=call, tool_result=result))
        if canceled:
            raise asyncio.CancelledError
        if publication_error is not None:
            return ToolResult(call.id, f"tool lifecycle publication failed: {publication_error}", True)
        return result
