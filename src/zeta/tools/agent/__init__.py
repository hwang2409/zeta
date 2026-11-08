"""The built-in bounded sub-agent tool."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ...agent.explicit_finish import (
    format_idle_resource_usage,
    idle_resource_usage,
)
from ...agent.presets import (
    GENERAL_PRESET,
    AgentType,
)
from ...agent.receipt import (
    MAX_AGENT_RESULT_BYTES,
    agent_stats,
    build_agent_progress,
    build_agent_receipt,
    encode_json,
    format_agent_stats,
    terminal_state,
)
from ...core.checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ...core.child_runs import (
    TERMINAL_AGENT_STATES as _TERMINAL_AGENT_STATES,
)
from ...core.child_runs import (
    active_agent_receipts as _active_agent_receipts,
)
from ...core.child_runs import (
    project_agent_status as _project_agent_status,
)
from ...core.child_runs import (
    read_agent_status,
)
from ...core.child_runs import (
    read_child_lifecycle as _read_child_lifecycle,
)
from ...core.child_runs import (
    unreadable_agent_status as _unreadable_agent_status,
)
from ...core.session_files import (
    child_directory,
    open_session_file,
    read_session_file,
    session_root,
)
from ...core.store import ConversationStore
from ...models.catalog import known_model_names
from ...protocol.types import (
    Message,
    StructuredContentValue,
    TextContent,
    ToolUseContent,
)
from ..agent_send import (  # noqa: F401
    AGENT_SEND_COMMIT_TIMEOUT_SECONDS,
    register_send,
    send_to_run,
)
from ..registry import (
    AbortSignal,
    ToolExecutionContext,
    ToolRegistry,
    ToolStreamPublisher,
    text_block,
)
from .approval import ChildApprovalPolicy  # noqa: F401

_TRUNCATION_NOTE = "\n[truncated]"
_STATUS_ROW_STRING_FIELDS = (
    "final_result",
    "description",
    "current_step",
    "agent_type",
    "reason",
    "started_at",
    "finished_at",
    "cwd",
)
_AGENT_OUTPUT_CACHE_MAX_ENTRIES = 4
_AGENT_OUTPUT_CACHE_MAX_CHARS = 32 * 1024 * 1024


def _agent_error(message: str, max_bytes: int) -> dict[str, object]:
    result = {
        "content": [text_block(f"agent error: {message}")],
        "isError": True,
        "structuredContent": None,
    }
    if len(encode_json(result)) <= max_bytes:
        return result
    return {
        "content": [text_block("agent error: response exceeds response limit")],
        "isError": True,
        "structuredContent": None,
    }


def _bounded_agent_result(
    result: dict[str, object], max_bytes: int
) -> dict[str, object]:
    if len(encode_json(result)) <= max_bytes:
        return result
    return _agent_error("response exceeds response limit", max_bytes)














def _agent_output_text(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolUseContent):
            arguments = json.dumps(
                block.tool_call.arguments, ensure_ascii=False, sort_keys=True
            )
            parts.append(f"[tool call: {block.tool_call.name} {arguments}]")
    if message.tool_result is not None and not parts:
        parts.append(message.tool_result.content)
    return "\n".join(parts)


def _serialize_agent_transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _agent_output_text(message)
        if text:
            lines.append(f"{message.role.value}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _agent_output_page(
    output: str,
    *,
    handle: str,
    offset: int,
    limit: int | None,
    max_bytes: int,
) -> dict[str, object] | None:
    total = len(output)
    requested_end = total if limit is None else min(total, offset + limit)

    def build(candidate: str, candidate_end: int) -> dict[str, object]:
        block = text_block(candidate, full_size=len(output.encode("utf-8")))
        block["full_size_chars"] = total
        block["truncated"] = candidate_end < total
        if candidate_end < total:
            block["next_offset"] = candidate_end
        return {
            "content": [block],
            "isError": False,
            "structuredContent": {
                "handle": handle,
                "offset": offset,
                "truncated": candidate_end < total,
                "total": total,
                **({"next_offset": candidate_end} if candidate_end < total else {}),
            },
        }

    low = offset
    high = requested_end
    while low < high:
        middle = (low + high + 1) // 2
        result = build(output[offset:middle], middle)
        if len(encode_json(result)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    result = build(output[offset:low], low)
    return result if len(encode_json(result)) <= max_bytes else None


def _read_child_file(store: ConversationStore, child_path: Path, name: str) -> bytes:
    with ExitStack() as cleanup:
        directory_fd = store.directory_fd
        for component in child_path.relative_to(store.session_dir.absolute()).parts:
            directory_fd = cleanup.enter_context(
                child_directory(directory_fd, component)
            )
        return read_session_file(directory_fd, name)




def _read_agent_transcript(
    child_path: Path,
    cached: tuple[tuple[int, int, int], str] | None,
) -> tuple[tuple[int, int, int], str, bool]:
    try:
        with session_root(child_path) as directory_fd:
            fd = open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY)
            with os.fdopen(fd, "rb") as handle:
                before = os.fstat(handle.fileno())
                key = (before.st_ino, before.st_size, before.st_mtime_ns)
                if cached is not None and cached[0] == key:
                    return (*cached, True)
                raw = handle.read()
                after = os.fstat(handle.fileno())
            after_key = (after.st_ino, after.st_size, after.st_mtime_ns)
        rows = raw.splitlines(keepends=True)
        entries: list[ConversationEntry] = []
        for index, line in enumerate(rows):
            try:
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                if (
                    index == len(rows) - 1
                    and not line.endswith(b"\n")
                    and isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError))
                ):
                    break
                raise ConversationIntegrityError(
                    f"invalid conversation row {index + 1}: {child_path}"
                ) from exc
            if index == 0:
                if not isinstance(row, dict) or row.get("type") != "header":
                    raise ConversationIntegrityError(
                        f"unsupported conversation schema: {child_path}"
                    )
                continue
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {index + 1} is not an object: {child_path}"
                )
            entries.append(ConversationEntry.from_dict(row))
        by_id = {entry.id: entry for entry in entries}
        current = entries[-1] if entries else None
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(
                    f"conversation parent cycle at {current.id}"
                )
            seen.add(current.id)
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        messages = [
            Message.from_dict(entry.data["message"])
            for entry in reversed(branch)
            if entry.type == "message"
        ]
        output = _serialize_agent_transcript(messages)
        return after_key, output, after_key == key
    except (OSError, ValueError, ConversationIntegrityError) as exc:
        raise ValueError("could not read child transcript") from exc


def _read_agent_output(
    child_path: Path,
    *,
    requested_handle: str,
    offset: int,
    limit: int | None,
    max_bytes: int,
    cached: tuple[tuple[int, int, int], str] | None = None,
) -> tuple[tuple[int, int, int], str, bool, dict[str, object]]:
    identity, output, cacheable = _read_agent_transcript(child_path, cached)
    result = _agent_output_page(
        output,
        handle=requested_handle,
        offset=offset,
        limit=limit,
        max_bytes=max_bytes,
    )
    if result is None:
        raise ValueError("transcript page exceeds response limit")
    return identity, output, cacheable, result








def _is_finished_agent_status(child: dict[str, StructuredContentValue]) -> bool:
    reason = child.get("reason")
    return (
        type(child.get("state")) is str
        and child.get("state") in _TERMINAL_AGENT_STATES
        and child.get("finished_at") is not None
        and not (
            isinstance(reason, str)
            and reason.startswith(("lost:", "unreadable:"))
        )
    )


def _truncate_status_text(value: str, max_bytes: int) -> str:
    if len(encode_json(value)) <= max_bytes:
        return value
    note_size = len(encode_json(_TRUNCATION_NOTE))
    if max_bytes <= note_size:
        return ""
    low = 0
    high = len(value)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = value[:middle] + _TRUNCATION_NOTE
        if len(encode_json(candidate)) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _status_response(
    children: list[dict[str, StructuredContentValue]],
    *,
    offset: int,
    total: int,
    truncated: bool,
    next_offset: int | None,
    finished_omitted: int,
    idle_usage: dict[str, object] | None = None,
) -> dict[str, object]:
    details = "\n".join(
        (
            "child {handle}: state: {state}; started_at: {started_at}; "
            "finished_at: {finished_at}; elapsed: {elapsed:.2f}s; "
            "turns_used: {turns_used}; step: {current_step}; "
            "{cwd_field}result: {final_result}{report_status_field}{reason_field}"
        ).format(
            **{
                **child,
                "final_result": child.get("final_result", ""),
                "cwd_field": (
                    f"cwd: {child['cwd']}; " if child.get("cwd") else ""
                ),
                "report_status_field": (
                    f"; report_status: {child['report_status']}"
                    if child.get("report_status")
                    else ""
                ),
                "reason_field": (
                    f"; reason: {child['reason']}" if child.get("reason") else ""
                ),
            }
        )
        + format_agent_stats(child)
        for child in children
    )
    omission_notice = (
        f"\n{finished_omitted} finished children omitted (query by handle for results)"
        if finished_omitted
        else ""
    )
    notice = (
        f"\nmore children available: call agent_status with offset={next_offset}"
        if truncated and next_offset is not None
        else ""
    )
    details_text = f"\n{details}" if details else ""
    idle_usage = idle_usage or idle_resource_usage(children)
    resource_text = (
        "\n" + format_idle_resource_usage(idle_usage)
        if idle_usage["idle_child_count"]
        else ""
    )
    content_text = (
        f"agent status: {len(children)} children{resource_text}"
        f"{details_text}{notice}{omission_notice}"
    )
    structured: dict[str, object] = {
        "children": children,
        "offset": offset,
        "truncated": truncated,
        "total": total,
    }
    if idle_usage["idle_child_count"]:
        structured.update(idle_usage)
    if finished_omitted:
        structured["finished_omitted"] = finished_omitted
    if truncated and next_offset is not None:
        structured["next_offset"] = next_offset
    return {
        "content": [text_block(content_text)],
        "isError": False,
        "structuredContent": structured,
    }


def _status_row_fits(
    row: dict[str, StructuredContentValue],
    *,
    offset: int,
    total: int,
    truncated: bool,
    next_offset: int | None,
    finished_omitted: int,
    max_bytes: int,
) -> bool:
    try:
        result = _status_response(
            [row],
            offset=offset,
            total=total,
            truncated=truncated,
            next_offset=next_offset,
            finished_omitted=finished_omitted,
        )
        return len(encode_json(result)) <= max_bytes
    except (OverflowError, RecursionError, TypeError, ValueError, UnicodeError):
        return False


def _fit_status_row(
    row: dict[str, StructuredContentValue],
    *,
    offset: int,
    total: int,
    truncated: bool,
    next_offset: int | None,
    finished_omitted: int,
    max_bytes: int,
) -> dict[str, StructuredContentValue] | None:
    candidate = dict(row)
    if _status_row_fits(
        candidate,
        offset=offset,
        total=total,
        truncated=truncated,
        next_offset=next_offset,
        finished_omitted=finished_omitted,
        max_bytes=max_bytes,
    ):
        return candidate

    for _ in range(16):
        changed = False
        for field in _STATUS_ROW_STRING_FIELDS:
            value = candidate.get(field)
            if type(value) is not str:
                continue
            encoded_size = len(encode_json(value))
            shortened = _truncate_status_text(value, encoded_size // 2)
            if shortened != value:
                candidate[field] = shortened
                changed = True
            if _status_row_fits(
                candidate,
                offset=offset,
                total=total,
                truncated=truncated,
                next_offset=next_offset,
                finished_omitted=finished_omitted,
                max_bytes=max_bytes,
            ):
                return candidate
        if not changed:
            break

    minimal = _unreadable_agent_status(
        str(row.get("handle", "")),
        reason="row too large",
    )
    if _status_row_fits(
        minimal,
        offset=offset,
        total=total,
        truncated=truncated,
        next_offset=next_offset,
        finished_omitted=finished_omitted,
        max_bytes=max_bytes,
    ):
        return _fit_status_row(
            minimal,
            offset=offset,
            total=total,
            truncated=truncated,
            next_offset=next_offset,
            finished_omitted=finished_omitted,
            max_bytes=max_bytes,
        )
    for _ in range(16):
        handle = minimal["handle"]
        if type(handle) is not str:
            break
        shortened = _truncate_status_text(handle, len(encode_json(handle)) // 2)
        if shortened == handle:
            break
        minimal["handle"] = shortened
        if _status_row_fits(
            minimal,
            offset=offset,
            total=total,
            truncated=truncated,
            next_offset=next_offset,
            finished_omitted=finished_omitted,
            max_bytes=max_bytes,
        ):
            return minimal
    return None




def _read_agent_status(
    store: object,
    *,
    requested_handle: str | None = None,
    running_handles: frozenset[str] | None = None,
) -> list[dict[str, StructuredContentValue]]:
    return read_agent_status(
        store,
        requested_handle=requested_handle,
        running_handles=running_handles,
        read_lifecycle=_read_child_lifecycle,
        project_status=_project_agent_status,
    )


async def _agent_status(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del abort_signal, stream_publisher, execution_context
    max_bytes = registry.max_output_chars
    requested = arguments.get("handle")
    if requested is not None and (type(requested) is not str or not requested):
        return _agent_error("handle must be a nonempty string", max_bytes)
    owner = registry._agent_owner
    running_handles = owner.active_instance_ids if owner is not None else None
    try:
        children = await asyncio.to_thread(
            _read_agent_status,
            registry.session_store,
            requested_handle=requested,
            running_handles=running_handles,
        )
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)
    if requested is None:
        finished_omitted = sum(_is_finished_agent_status(child) for child in children)
        children = [child for child in children if not _is_finished_agent_status(child)]
    else:
        finished_omitted = 0
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    if type(offset) is not int or offset < 0:
        return _agent_error("offset must be a nonnegative integer", max_bytes)
    if limit is not None and (type(limit) is not int or limit < 1):
        return _agent_error("limit must be a positive integer", max_bytes)
    total = len(children)
    root_idle_usage = idle_resource_usage(children)
    raw_page = children[offset : offset + limit if limit is not None else None]
    page: list[dict[str, StructuredContentValue]] = []
    for index, child in enumerate(raw_page):
        child_offset = offset + index
        child_truncated = child_offset + 1 < total
        fitted = _fit_status_row(
            child,
            offset=child_offset,
            total=total,
            truncated=child_truncated,
            next_offset=child_offset + 1 if child_truncated else None,
            finished_omitted=finished_omitted,
            max_bytes=max_bytes,
        )
        if fitted is None:
            break
        page.append(fitted)

    while page:
        next_offset = offset + len(page)
        truncated = next_offset < total
        result = _status_response(
            page,
            offset=offset,
            total=total,
            truncated=truncated,
            next_offset=next_offset if truncated else None,
            finished_omitted=finished_omitted,
            idle_usage=root_idle_usage,
        )
        try:
            fits = len(encode_json(result)) <= max_bytes
        except (OverflowError, RecursionError, TypeError, ValueError, UnicodeError):
            fits = False
        if fits:
            return result
        page.pop()
    if offset < len(children):
        return _agent_error("status item exceeds response limit", max_bytes)
    count = len(children)
    label = "child" if count == 1 else "children"
    result = _status_response(
        [],
        offset=offset,
        total=count,
        truncated=False,
        next_offset=None,
        finished_omitted=finished_omitted,
        idle_usage=root_idle_usage,
    )
    if count == 1:
        result["content"] = [
            text_block(
                f"agent status: {count} {label}"
                + (
                    f"\n{finished_omitted} finished children omitted "
                    "(query by handle for results)"
                    if finished_omitted
                    else ""
                )
            )
        ]
    try:
        return (
            result
            if len(encode_json(result)) <= max_bytes
            else _agent_error("response exceeds response limit", max_bytes)
        )
    except (OverflowError, RecursionError, TypeError, ValueError, UnicodeError):
        return _agent_error("response exceeds response limit", max_bytes)


async def _agent_output(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del abort_signal, stream_publisher, execution_context
    max_bytes = registry.max_output_chars
    handle = arguments.get("handle")
    if type(handle) is not str or not handle:
        return _agent_error("handle must be a nonempty string", max_bytes)
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    if type(offset) is not int or offset < 0:
        return _agent_error("offset must be a nonnegative integer", max_bytes)
    if limit is not None and (type(limit) is not int or limit < 1):
        return _agent_error("limit must be a positive integer", max_bytes)
    try:
        child_path = _active_agent_receipts(registry.session_store).get(handle)
        if child_path is None:
            raise ValueError("unknown child handle")
        cached = registry._agent_output_cache.get(child_path)
        identity, output, cacheable, result = await asyncio.to_thread(
            _read_agent_output,
            child_path,
            requested_handle=handle,
            offset=offset,
            limit=limit,
            max_bytes=max_bytes,
            cached=cached,
        )
        previous = registry._agent_output_cache.pop(child_path, None)
        if previous is not None:
            registry._agent_output_cache_chars -= len(previous[1])
        if cacheable and len(output) <= _AGENT_OUTPUT_CACHE_MAX_CHARS:
            registry._agent_output_cache[child_path] = (identity, output)
            registry._agent_output_cache_chars += len(output)
            registry._agent_output_cache.move_to_end(child_path)
            while (
                len(registry._agent_output_cache) > _AGENT_OUTPUT_CACHE_MAX_ENTRIES
                or registry._agent_output_cache_chars > _AGENT_OUTPUT_CACHE_MAX_CHARS
            ):
                _, evicted = registry._agent_output_cache.popitem(last=False)
                registry._agent_output_cache_chars -= len(evicted[1])
        return _bounded_agent_result(result, max_bytes)
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)


def agent_result(
    text: str,
    *,
    tool_call_id: str = "",
    error: bool,
    turns_used: int,
    child_session_path: str,
    agent_type: AgentType | None = None,
    status: str | None = None,
    child_instance_id: str | None = None,
    description: str | None = None,
    depth: int | None = None,
    stats: dict[str, object],
    include_stats: bool = True,
    canceled: bool = False,
    notice: str | None = None,
    notice_items: Sequence[str] | None = None,
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
) -> dict[str, object]:
    structured_content: dict[str, object] = {
        "turns_used": turns_used,
        "child_session_path": child_session_path,
    }
    if agent_type is not None and agent_type != GENERAL_PRESET.name:
        structured_content["agent_type"] = agent_type
    if status is not None:
        structured_content["status"] = status
    if child_instance_id is not None:
        structured_content["child_instance_id"] = child_instance_id
    if description is not None:
        structured_content["description"] = description
    # Keep depth-one result payloads byte-compatible with the pre-nesting shape.
    if depth is not None and depth != 1:
        structured_content["depth"] = depth
    if status == "running":
        return build_agent_progress(
            text,
            stats,
            structured_content=structured_content,
            tool_call_id=tool_call_id,
            max_bytes=max_bytes,
            include_stats=include_stats,
        )
    state = terminal_state(error=error, canceled=canceled, status=status)
    return build_agent_receipt(
        state,
        text,
        stats,
        structured_content=structured_content,
        tool_call_id=tool_call_id,
        max_bytes=max_bytes,
        notice=notice,
        notice_items=notice_items,
    )


async def _agent(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del registry
    runner = execution_context.agent_runner if execution_context is not None else None
    call = execution_context.tool_call if execution_context is not None else None
    if runner is None or call is None:
        return agent_result(
            "agent error: agent tool is unavailable outside an agent loop",
            error=True,
            turns_used=0,
            child_session_path="",
            stats=agent_stats({}, status="failed", turns_used=0),
        )
    return await runner(
        call,
        arguments,
        abort_signal,
        stream_publisher,
        execution_context,
    )


def register(registry: ToolRegistry) -> None:
    agent_names = registry.agent_catalog.names()
    agent_description = registry.agent_catalog.description()
    registry.register_session_tool(
        "agent",
        _agent,
        binding="shared",
        description=(
            "Delegate multi-step exploration or research that would pollute the "
            "main context. The child has its own bounded context and may spawn "
            "one level of grandchildren, but grandchildren cannot spawn agents. "
            "Pass model to run the child on another provider's model and "
            "orchestrate it from here. Pass cwd to start the child in another "
            "directory or git worktree so its bash session and relative file "
            "paths resolve there instead of your own working directory. The "
            "returned child_instance_id is the "
            "stable handle for agent_status. Background completion is announced "
            "automatically: while you run, after your current tool call and "
            "before your next model call; while you are idle, at your next "
            "turn. Do not poll. "
            "Built-in types: "
            f"{agent_description}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string"},
                "agent_type": {
                    "type": "string",
                    "enum": agent_names,
                    "description": agent_description,
                },
                "preset": {
                    "type": "string",
                    "enum": agent_names,
                    "description": (
                        "Alias for agent_type. Select a preset from the session catalog."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": known_model_names(),
                    "description": (
                        "Run the child on this model instead of inheriting the "
                        "parent's. The provider follows from the model, so this "
                        "is how one provider delegates to another. Implies "
                        "background unless background is passed explicitly; "
                        "--print runs children synchronously."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Directory the child works in: its bash session cwd and "
                        "relative read/edit/write paths resolve here. Absolute, "
                        "or relative to your own working directory; must be an "
                        "existing directory (a symlinked directory is rejected). "
                        "Pass this when delegating work in another directory or "
                        "git worktree so the child does not run in your checkout. "
                        "Omit to inherit your working directory. Grandchildren "
                        "default to their parent's cwd."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": "Keep the child running across parent turns and return a handle.",
                },
            },
            "required": ["prompt", "description"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
    registry.register_session_tool(
        "agent_status",
        _agent_status,
        binding="shared",
        description=(
            "Inspect child agents from this session. Pass a child handle for "
            "one child, or omit it to list unfinished children plus terminal "
            "lost or unreadable records. Other finished children are omitted from "
            "list responses; query by handle for their results. "
            "Completion is announced automatically. Use this for on-demand "
            "inspection, not polling."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Stable child handle returned by the agent tool.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Child index for a status page continuation.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of children in this status page.",
                },
            },
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
    registry.register_session_tool(
        "agent_output",
        _agent_output,
        binding="shared",
        description=(
            "Read a child agent transcript by handle. Output is bounded and "
            "paginated by character offset. This is read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Stable child handle returned by the agent tool.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Readable transcript character offset.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum readable transcript characters.",
                },
            },
            "required": ["handle"],
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
    register_send(registry)
