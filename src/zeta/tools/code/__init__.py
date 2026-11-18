"""The code tool and its fixed registry authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from ...protocol.types import ToolCall, ToolResult, ToolSchema
from ...runtime.execution import NestedToolParent, ToolExecutionContext
from ._guest import FRAME, GuestError, run_guest

if TYPE_CHECKING:
    from ..registry import ToolDefinition, ToolRegistry

CODE_MODE_ALLOWED_NAMES: Final = frozenset({
    "read", "write", "edit", "bash", "fetch", "websearch", "project",
    "recall_history", "run_background", "task_output",
})


def authority_snapshot(registry: ToolRegistry, schemas: Sequence[ToolSchema]) -> Mapping[str, ToolDefinition]:
    from ..registry import ToolOrigin

    return MappingProxyType({
        name: definition
        for schema in schemas
        if (name := schema["name"]) in CODE_MODE_ALLOWED_NAMES
        and (definition := registry.definition_reference(name)) is not None
        and definition.origin is ToolOrigin.PACKAGED_BUILTIN
    })


def describe_code(registry: ToolRegistry, schemas: Sequence[ToolSchema]) -> None:
    from ..registry import ToolOrigin

    code = next((schema for schema in schemas if schema["name"] == "code"), None)
    definition = registry.definition_reference("code")
    if code is None or definition is None or definition.origin is not ToolOrigin.PACKAGED_BUILTIN:
        return
    names = ", ".join(sorted(authority_snapshot(registry, schemas))) or "none"
    code["description"] = (
        "Run JavaScript to combine results from several tools. Pass {script: string}. "
        "Use await tools.call(name, args) or await tools[name](args). "
        f"Available tools: {names}. "
        "Only output(value), called once, sends JSON to the model. "
        "Inner approvals still apply; catch ToolDenied or ToolError. "
        "Use at most 50 sequential calls, 64 KiB of script and 60 seconds of active JavaScript. "
        "No filesystem, network, clocks, timers or persistent state except through tools. "
        "Use ordinary calls for one tool or shell commands that already return selective output."
    )


def _bounded_output(data: bytes) -> str:
    if len(data) <= FRAME:
        return data.decode()
    # The marker stays valid JSON even if the head and tail cut the original JSON.
    size = FRAME // 2 - 128
    while True:
        head = data[:size].decode(errors="ignore")
        tail = data[-size:].decode(errors="ignore")
        omitted = len(data) - len(head.encode()) - len(tail.encode())
        text = json.dumps({"head": head, "omitted_bytes": omitted, "tail": tail}, ensure_ascii=False)
        if len(text.encode()) <= FRAME:
            return text
        size //= 2


async def _code(arguments, abort_signal, *, execution_context: ToolExecutionContext) -> ToolResult:
    call = execution_context.tool_call
    execute = execution_context.nested_tool_executor
    if execute is None:
        return ToolResult(call.id, "code mode needs an owning AgentLoop", is_error=True)
    summaries = []

    async def dispatch(name, arguments, index):
        inner_id = f"{call.id}:code:{index}"
        result = await execute(ToolCall(inner_id, name, arguments), NestedToolParent(call.id, index))
        structured = result.structured_content or {}
        error = structured.get("error") or {}
        unexposed = result.content.startswith(f"tool {name!r} is not exposed to code mode")
        if not unexposed:
            summaries.append({
                "id": inner_id, "name": name, "index": index,
                "state": "canceled" if result.is_canceled else "error" if result.is_error else "completed",
                "summary": result.content[:240],
            })
        if result.is_error or result.is_canceled:
            return {"ok": False, "error": {
                "name": "AbortError" if result.is_canceled else "ToolDenied"
                if isinstance(error, dict) and error.get("kind") == "denied" else "ToolError",
                "tool": name, "callId": index, "content": result.content,
                "structuredContent": result.structured_content, "canceled": result.is_canceled,
            }}
        return {"ok": True, "value": {
            "content": result.content, "structuredContent": result.structured_content,
            "isCanceled": result.is_canceled,
        }}

    source = arguments["script"]
    error = False
    emitted = b""
    try:
        emitted = await run_guest(source, sorted(execute.exposed_names), dispatch)
        content = _bounded_output(emitted)
    except (GuestError, TimeoutError, UnicodeError) as exc:
        error = True
        content = str(exc) or "code mode wall deadline exceeded"
    projection = {
        "version": 1, "state": "error" if error else "completed",
        "source_sha256": hashlib.sha256(source.encode(errors="surrogatepass")).hexdigest(),
        "calls": summaries, "output_bytes": len(emitted), "omitted_calls": 0,
    }
    while summaries and len(json.dumps(projection).encode()) > FRAME:
        summaries.pop()
        projection["omitted_calls"] += 1
    return ToolResult(call.id, content, is_error=error, structured_content={"code_mode": projection})


def register(registry: ToolRegistry) -> None:
    if registry.code_mode_enabled:
        registry.register(
            "code", _code, binding="shared", requires_approval=False,
            parameters={"type": "object", "properties": {"script": {"type": "string"}},
                        "required": ["script"], "additionalProperties": False},
        )
