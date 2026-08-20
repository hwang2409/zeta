"""Tool registration, validation, execution, and session-cwd tools."""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ToolCall, ToolResult, ToolSchema


class ToolAbortSignal:
    """Cooperative cancellation state shared by a turn's tool calls."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def abort(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def aborted(self) -> bool:
        return self.is_set()


AbortSignal = ToolAbortSignal
ToolHook = Callable[[str, dict[str, Any]], bool | Awaitable[bool] | None]
ToolHandler = Callable[..., str | ToolResult | Awaitable[str | ToolResult]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    parallel_safe: bool = False

    def schema(self) -> ToolSchema:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


class ToolRegistry:
    """One provider-neutral registry for built-in and custom tools."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        pre_execute_hook: ToolHook | None = None,
        hook: ToolHook | None = None,
        abort_signal: ToolAbortSignal | asyncio.Event | None = None,
        max_output_chars: int = 10_000,
        register_builtin: bool = True,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"tool cwd is not a directory: {self.cwd}")
        if type(max_output_chars) is not int or max_output_chars < 1:
            raise ValueError("max_output_chars must be a positive integer")
        if pre_execute_hook is not None and hook is not None:
            raise ValueError("pass only one pre-execution hook")
        self.pre_execute_hook = pre_execute_hook or hook
        self.abort_signal = abort_signal or ToolAbortSignal()
        self.max_output_chars = max_output_chars
        self._tools: dict[str, ToolDefinition] = {}
        if register_builtin:
            self._register_builtin_tools()

    @property
    def schemas(self) -> list[ToolSchema]:
        return [definition.schema() for definition in self._tools.values()]

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return self.schemas

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    @property
    def definitions_by_name(self) -> Mapping[str, ToolDefinition]:
        return self._tools

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
        normalized = _normalize_schema(supplied_schemas[0] if supplied_schemas else None)
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=normalized,
            handler=handler,
            parallel_safe=parallel_safe,
        )
        self._tools[name] = definition
        return definition

    register_tool = register

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def abort(self) -> None:
        abort = getattr(self.abort_signal, "abort", None)
        if callable(abort):
            abort()
        else:
            set_signal = getattr(self.abort_signal, "set", None)
            if not callable(set_signal):
                raise TypeError("abort signal does not support abort or set")
            set_signal()

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        abort_signal: ToolAbortSignal | asyncio.Event | None = None,
    ) -> ToolResult:
        signal_state = abort_signal or self.abort_signal
        if _signal_is_set(signal_state):
            return _canceled_result(tool_call.id)
        definition = self._tools.get(tool_call.name)
        if definition is None:
            return ToolResult(tool_call.id, f"unknown tool: {tool_call.name}", True)
        try:
            arguments = _validate_arguments(tool_call.arguments, definition.parameters)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return ToolResult(tool_call.id, f"invalid arguments: {exc}", True)
        if _signal_is_set(signal_state):
            return _canceled_result(tool_call.id)
        if self.pre_execute_hook is not None:
            try:
                allowed = self.pre_execute_hook(tool_call.name, arguments)
                if inspect.isawaitable(allowed):
                    allowed = await allowed
            except Exception as exc:
                return ToolResult(tool_call.id, f"pre-execution hook failed: {exc}", True)
            if allowed is False:
                return ToolResult(tool_call.id, "tool execution denied", True)
        if _signal_is_set(signal_state):
            return _canceled_result(tool_call.id)
        try:
            result = await _invoke_handler(definition.handler, arguments, signal_state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult(tool_call.id, str(exc), True)
        if isinstance(result, ToolResult):
            if result.tool_call_id == tool_call.id:
                return result
            return ToolResult(
                tool_call.id,
                f"tool result id mismatch: expected {tool_call.id}, got {result.tool_call_id}",
                True,
            )
        if isinstance(result, str):
            return ToolResult(tool_call.id, result)
        return ToolResult(
            tool_call.id,
            "invalid tool handler result: expected str or ToolResult",
            True,
        )

    async def execute_many(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        abort_signal: ToolAbortSignal | asyncio.Event | None = None,
    ) -> list[ToolResult]:
        """Execute calls with safe contiguous groups in parallel, preserving order."""

        signal_state = abort_signal or self.abort_signal
        results: list[ToolResult | None] = [None] * len(tool_calls)
        index = 0
        while index < len(tool_calls):
            if _signal_is_set(signal_state):
                for remaining in range(index, len(tool_calls)):
                    results[remaining] = _canceled_result(tool_calls[remaining].id)
                break
            definition = self._tools.get(tool_calls[index].name)
            if definition is None or not definition.parallel_safe:
                results[index] = await self.execute(
                    tool_calls[index], abort_signal=signal_state
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
                    self.execute(call, abort_signal=signal_state)
                    for call in tool_calls[index:end]
                )
            )
            results[index:end] = group
            index = end
        return [result for result in results if result is not None]

    def _register_builtin_tools(self) -> None:
        self.register(
            "read",
            self._read,
            description="Read a UTF-8 file inside the session cwd.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )
        self.register(
            "list",
            self._list,
            description="List files and directories inside the session cwd.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "depth": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        )
        self.register(
            "exec",
            self._exec,
            description=(
                "Run a shell command from the session cwd. "
                "This tool is not a sandbox."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "timeout": {"type": "number", "exclusiveMinimum": 0},
                    "max_output": {"type": "integer", "minimum": 1},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    async def _read(
        self,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> str:
        path = self._jail_path(arguments["path"])
        if not path.is_file():
            raise ValueError(f"not a file: {arguments['path']}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read file: {exc}") from exc
        lines = text.splitlines()
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit")
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        return "\n".join(selected)

    async def _list(
        self,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> str:
        relative_path = arguments.get("path", ".")
        path = self._jail_path(relative_path)
        if not path.is_dir():
            raise ValueError(f"not a directory: {relative_path}")
        depth = arguments.get("depth", 1)
        found: list[str] = []
        await self._list_children(path, depth, found, abort_signal)
        return "\n".join(found)

    async def _list_children(
        self,
        path: Path,
        depth: int,
        found: list[str],
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> None:
        if _signal_is_set(abort_signal):
            return
        try:
            entries = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"could not list directory: {exc}") from exc
        for entry in entries:
            if _signal_is_set(abort_signal):
                return
            try:
                resolved = entry.resolve()
                resolved.relative_to(self.cwd)
            except (OSError, ValueError):
                continue
            relative = os.fspath(entry.relative_to(self.cwd))
            if entry.is_dir() and not entry.is_symlink():
                relative += "/"
            found.append(relative)
            if depth > 1 and entry.is_dir() and not entry.is_symlink():
                await self._list_children(entry, depth - 1, found, abort_signal)

    async def _exec(
        self,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> str:
        timeout = arguments.get("timeout", 30.0)
        output_limit = arguments.get("max_output", self.max_output_chars)
        try:
            process = await asyncio.create_subprocess_shell(
                arguments["command"],
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.send_signal(signal.SIGKILL)
                stdout, stderr = await process.communicate()
                result = _format_exec_result(
                    -signal.SIGKILL,
                    stdout,
                    stderr,
                    output_limit,
                    suffix="command timed out",
                )
                raise ValueError(result)
        except OSError as exc:
            raise ValueError(f"could not execute command: {exc}") from exc
        result = _format_exec_result(process.returncode, stdout, stderr, output_limit)
        if process.returncode:
            raise ValueError(result)
        return result

    def _jail_path(self, raw_path: object) -> Path:
        if type(raw_path) is not str or not raw_path:
            raise ValueError("path must be a nonempty string")
        try:
            candidate = Path(raw_path)
            resolved = (candidate if candidate.is_absolute() else self.cwd / candidate).resolve()
            resolved.relative_to(self.cwd)
        except (OSError, ValueError) as exc:
            raise ValueError(f"path escapes session cwd: {raw_path}") from exc
        return resolved


async def _invoke_handler(
    handler: ToolHandler,
    arguments: dict[str, Any],
    abort_signal: ToolAbortSignal | asyncio.Event,
) -> str | ToolResult:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        result = handler(arguments, abort_signal)
    else:
        try:
            signature.bind(arguments, abort_signal)
        except TypeError:
            try:
                signature.bind(arguments, abort_signal=abort_signal)
            except TypeError:
                signature.bind(arguments)
                result = handler(arguments)
            else:
                result = handler(arguments, abort_signal=abort_signal)
        else:
            result = handler(arguments, abort_signal)
    if inspect.isawaitable(result):
        result = await result
    return result


def _normalize_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise TypeError("tool parameter schema must be an object")
    return dict(schema)


def _validate_arguments(arguments: object, schema: Mapping[str, Any]) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    _validate_schema(arguments, schema, "arguments")
    return dict(arguments)


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> None:
    for alternative_key in ("anyOf", "oneOf"):
        alternatives = schema.get(alternative_key)
        if alternatives is not None:
            matches = 0
            for alternative in alternatives:
                try:
                    _validate_schema(value, alternative, path)
                except ValueError:
                    pass
                else:
                    matches += 1
            if (alternative_key == "anyOf" and matches == 0) or (
                alternative_key == "oneOf" and matches != 1
            ):
                raise ValueError(f"{path} does not match {alternative_key}")
            return
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise ValueError(f"{path} must be {expected_type}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal the declared constant")
    if "enum" in schema and value not in schema["enum"]:
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
    return True


def _signal_is_set(signal_state: object) -> bool:
    is_set = getattr(signal_state, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(getattr(signal_state, "aborted", False))


def _canceled_result(call_id: str) -> ToolResult:
    return ToolResult(call_id, "tool execution canceled", True)


def _format_exec_result(
    returncode: int | None,
    stdout: bytes,
    stderr: bytes,
    output_limit: int,
    *,
    suffix: str | None = None,
) -> str:
    output = f"stdout:\n{stdout.decode(errors='replace')}\nstderr:\n{stderr.decode(errors='replace')}"
    if suffix:
        output = f"{suffix}\n{output}"
    output = _truncate(output, output_limit)
    return f"exit_code: {returncode}\n{output}"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[output truncated]"
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker
