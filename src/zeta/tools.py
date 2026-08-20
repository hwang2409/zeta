"""Tool registration, validation, execution, and session-cwd defaults."""

from __future__ import annotations

import asyncio
import copy
import heapq
import inspect
import json
import math
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

    async def wait(self) -> None:
        await self._event.wait()

    @property
    def aborted(self) -> bool:
        return self.is_set()


AbortSignal = ToolAbortSignal
ToolHook = Callable[[str, dict[str, Any]], bool | Awaitable[bool] | None]
ToolHandler = Callable[..., str | ToolResult | Awaitable[str | ToolResult]]


class _ToolCanceled(Exception):
    pass


async def _yield_for_abort(
    abort_signal: ToolAbortSignal | asyncio.Event,
) -> None:
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()
    await asyncio.sleep(0)
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()


class _BoundedOutput:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._data = bytearray()

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    @property
    def retained_bytes(self) -> int:
        return len(self._data)

    def append(self, chunk: bytes) -> None:
        remaining = self.limit - len(self._data)
        if remaining > 0:
            self._data.extend(chunk[:remaining])


class _BoundedText:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._parts: list[str] = []
        self._length = 0
        self._has_line = False
        self.truncated = False

    @property
    def retained_chars(self) -> int:
        return self._length

    def append(self, value: str) -> None:
        remaining = self.limit - self._length
        if remaining > 0:
            retained = value[:remaining]
            self._parts.append(retained)
            self._length += len(retained)
        if len(value) > remaining:
            self.truncated = True

    def begin_line(self) -> None:
        if self._has_line:
            self.append("\n")
        self._has_line = True

    def append_line(self, value: str) -> None:
        self.begin_line()
        self.append(value)

    def render(self) -> str:
        text = "".join(self._parts)
        if self.truncated:
            return _truncate(text + _TRUNCATION_MARKER, self.limit)
        return text


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
            "parameters": copy.deepcopy(self.parameters),
        }


def _copy_definition(definition: ToolDefinition) -> ToolDefinition:
    return ToolDefinition(
        name=definition.name,
        description=definition.description,
        parameters=copy.deepcopy(definition.parameters),
        handler=definition.handler,
        parallel_safe=definition.parallel_safe,
    )


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
        return _copy_definition(definition)

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

    def start_batch(self) -> None:
        """Rotate the active signal before a new tool batch."""
        self.abort_signal = ToolAbortSignal()

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
        except _ToolCanceled:
            return _canceled_result(tool_call.id)
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
            description="Read a UTF-8 file. Relative paths use the session cwd.",
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
            description="List a directory. Relative paths use the session cwd.",
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
        path = self._path(arguments["path"])
        if not path.is_file():
            raise ValueError(f"not a file: {arguments['path']}")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit")
        output = _BoundedText(self.max_output_chars)
        line_index = 0
        selected_count = 0
        line_has_data = False
        line_started = False
        try:
            with path.open("r", encoding="utf-8", newline=None) as handle:
                while True:
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    for character in chunk:
                        if character == "\n":
                            selected = line_index >= offset and (
                                limit is None or selected_count < limit
                            )
                            if selected:
                                if not line_started:
                                    output.begin_line()
                                selected_count += 1
                                if output.truncated:
                                    return output.render()
                            line_index += 1
                            line_has_data = False
                            line_started = False
                            if limit is not None and selected_count >= limit:
                                return output.render()
                            continue
                        line_has_data = True
                        if line_index < offset or (
                            limit is not None and selected_count >= limit
                        ):
                            continue
                        if not line_started:
                            output.begin_line()
                            line_started = True
                        output.append(character)
                        if output.truncated:
                            return output.render()
                    await _yield_for_abort(abort_signal)
                if line_has_data:
                    selected = line_index >= offset and (
                        limit is None or selected_count < limit
                    )
                    if selected:
                        if not line_started:
                            output.begin_line()
                        selected_count += 1
        except OSError as exc:
            raise ValueError(f"could not read file: {exc}") from exc
        return output.render()

    async def _list(
        self,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> str:
        relative_path = arguments.get("path", ".")
        path = self._path(relative_path)
        if not path.is_dir():
            raise ValueError(f"not a directory: {relative_path}")
        depth = arguments.get("depth", 1)
        output = _BoundedText(self.max_output_chars)
        await self._list_children(path, depth, output, abort_signal)
        if _signal_is_set(abort_signal):
            raise _ToolCanceled()
        return output.render()

    async def _list_children(
        self,
        path: Path,
        depth: int,
        output: _BoundedText,
        abort_signal: ToolAbortSignal | asyncio.Event,
    ) -> bool:
        if _signal_is_set(abort_signal):
            raise _ToolCanceled()
        try:
            entries = heapq.nsmallest(
                max(1, self.max_output_chars - output.retained_chars),
                path.iterdir(),
                key=lambda item: item.name,
            )
        except OSError as exc:
            raise ValueError(f"could not list directory: {exc}") from exc
        for index, entry in enumerate(entries):
            if _signal_is_set(abort_signal):
                raise _ToolCanceled()
            if index % 64 == 0:
                await _yield_for_abort(abort_signal)
            try:
                relative = os.fspath(entry.relative_to(self.cwd))
            except ValueError:
                relative = os.fspath(entry)
            if entry.is_dir() and not entry.is_symlink():
                relative += "/"
            output.append_line(relative)
            if output.truncated:
                return True
            if depth > 1 and entry.is_dir() and not entry.is_symlink():
                if await self._list_children(entry, depth - 1, output, abort_signal):
                    return True
        return False

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
                start_new_session=True,
            )
        except OSError as exc:
            raise ValueError(f"could not execute command: {exc}") from exc

        stdout_capture = _BoundedOutput(output_limit)
        stderr_capture = _BoundedOutput(output_limit)
        process_wait = asyncio.create_task(process.wait())
        stdout_drain = asyncio.create_task(
            _drain_stream(process.stdout, stdout_capture)
        )
        stderr_drain = asyncio.create_task(
            _drain_stream(process.stderr, stderr_capture)
        )
        process_tasks = (process_wait, stdout_drain, stderr_drain)
        abort_wait = asyncio.create_task(_wait_for_abort(abort_signal))
        timeout_wait = asyncio.create_task(asyncio.sleep(timeout))
        try:
            pending: set[asyncio.Task[Any]] = {
                *process_tasks,
                abort_wait,
                timeout_wait,
            }
            while True:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if _signal_is_set(abort_signal):
                    await _kill_and_reap(process, process_tasks)
                    raise _ToolCanceled()
                if all(task.done() for task in process_tasks):
                    process_wait.result()
                    result = _format_exec_result(
                        process.returncode,
                        stdout_capture.data,
                        stderr_capture.data,
                        output_limit,
                    )
                    if process.returncode:
                        raise ValueError(result)
                    return result
                if timeout_wait in done:
                    break

            await _kill_and_reap(process, process_tasks)
            raise ValueError(
                _format_exec_result(
                    process.returncode,
                    stdout_capture.data,
                    stderr_capture.data,
                    output_limit,
                    suffix="command timed out",
                )
            )
        except asyncio.CancelledError:
            await _kill_and_reap(process, process_tasks)
            raise
        except BaseException:
            await _kill_and_reap(process, process_tasks)
            raise
        finally:
            for waiter in (abort_wait, timeout_wait):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(abort_wait, timeout_wait, return_exceptions=True)

    def _path(self, raw_path: object) -> Path:
        if type(raw_path) is not str or not raw_path:
            raise ValueError("path must be a nonempty string")
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else self.cwd / candidate


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
    try:
        normalized = copy.deepcopy(dict(schema))
    except Exception as exc:
        raise ValueError("schema must contain JSON data") from exc
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


_TRUNCATION_MARKER = "\n...[output truncated]"


_SCHEMA_KEYS = {
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
    if expected_type is not None:
        if type(expected_type) is not str or expected_type not in _SCHEMA_TYPES:
            raise ValueError(f"unsupported schema type at {path}")
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


def _signal_is_set(signal_state: object) -> bool:
    is_set = getattr(signal_state, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(getattr(signal_state, "aborted", False))


async def _wait_for_abort(signal_state: object) -> None:
    if _signal_is_set(signal_state):
        return
    wait = getattr(signal_state, "wait", None)
    if callable(wait):
        result = wait()
        if inspect.isawaitable(result):
            await result
        return
    while not _signal_is_set(signal_state):
        await asyncio.sleep(0.01)


async def _drain_stream(stream: object, capture: _BoundedOutput) -> None:
    if stream is None:
        return
    read = getattr(stream, "read")
    while True:
        chunk = await read(65_536)
        if not chunk:
            return
        capture.append(chunk)


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    process_tasks: Sequence[asyncio.Task[Any]],
) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
    for task in process_tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if not task.cancelled():
            task.exception()


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
    return _truncate(f"exit_code: {returncode}\n{output}", output_limit)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = _TRUNCATION_MARKER
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker
