"""Child lifetime turn admission and epoch-free result projection."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.store import ConversationStore
    from ..runtime.loop import AgentLoop

from ..core.agent_progress import publish_child_started
from ..core.checkpoints import _now
from ..core.store._async_writes import _EnsureTurnLimitHandoffCommand
from ..protocol.types import Message, StreamEvent, StreamEventType, harness_notice
from .presets import valid_max_turns
from .receipt import _without_agent_receipt_suffix, has_agent_receipt_suffix

HANDOFF_NAME = "turn-limit-handoff.md"
_MARKER_PREFIX = "[zeta:turn-limit:v1] "


@dataclass(frozen=True, slots=True)
class TurnLimitProjection:
    status: str
    handoff_path: str
    readable_text: str


class ChildTurnControl:
    """Reserve completions against one child's lifetime limit."""

    def __init__(self, limit: int, child_session_path: str) -> None:
        if not valid_max_turns(limit):
            raise ValueError("child max_turns must be a positive integer")
        self.limit = limit
        self.handoff_path = str(Path(child_session_path) / HANDOFF_NAME)
        self.used = 0
        self._reserved = False
        self._warned = False

    def begin_completion(self) -> Message | None:
        if self._reserved:
            return None
        if self.used >= self.limit:
            return None
        self._reserved = True
        number = self.used + 1
        if not self._warned and number == math.ceil(0.9 * self.limit):
            self._warned = True
            remaining = self.limit - self.used
            return harness_notice(
                "child turn limit",
                f"{remaining} model completion(s) remain, including this one. "
                "finish current work now; do not start new scope. Commit local work "
                f"if applicable and write the handoff to {self.handoff_path}.",
                event="child_turn_limit",
            )
        return None

    def finish_completion(self) -> bool:
        if not self._reserved:
            raise RuntimeError("no child completion is reserved")
        self._reserved = False
        self.used += 1
        return self.exhausted

    def abandon_completion(self) -> None:
        self._reserved = False

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def reserved(self) -> bool:
        return self._reserved


def turn_control_for(
    limit: int | None, child_session_path: str
) -> ChildTurnControl | None:
    return None if limit is None else ChildTurnControl(limit, child_session_path)


async def forward_nested_agent_start(child_loop: AgentLoop, event: StreamEvent) -> None:
    if event.type is StreamEventType.AGENT_START and child_loop.agent_depth > 1:
        await child_loop._wait_for_background_event(
            StreamEvent(
                StreamEventType.AGENT_START,
                data={"agent_instance_id": child_loop.agent_instance_id},
            )
        )


async def append_child_start(
    store: ConversationStore,
    run_id: str,
    parent_run_id: str | None,
    agent_type: str,
    description: str,
    *,
    accepts_follow_ups: bool,
) -> None:
    """Publish a child start only after its immediate parent stores it."""
    entry = await store.append_child_run_started_async(
        run_id, parent_run_id, agent_type, description, _now()
    )
    publish_child_started(
        run_id,
        entry.seq,
        entry.data,
        accepts_follow_ups=accepts_follow_ups,
    )


def handoff_path(child_session_path: str | Path) -> str:
    return str(Path(child_session_path) / HANDOFF_NAME)


def marker_for(child_session_path: str | Path) -> str:
    payload = json.dumps(
        {"handoff_path": handoff_path(child_session_path)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _MARKER_PREFIX + payload


def build_turn_limit_result(receipt: str, child_session_path: str | Path) -> str:
    """Append the trusted marker after one canonical completed receipt."""
    if not has_agent_receipt_suffix(receipt):
        raise ValueError("turn-limit result must end with a canonical receipt")
    return f"{receipt}\n{marker_for(child_session_path)}"


def parse_turn_limit_result(
    text: object, child_session_path: str | Path
) -> TurnLimitProjection | None:
    if type(text) is not str:
        return None
    separator = text.rfind("\n")
    if separator < 0:
        return None
    readable = text[:separator]
    marker = text[separator + 1 :]
    if not has_agent_receipt_suffix(readable) or marker != marker_for(child_session_path):
        return None
    return TurnLimitProjection("turn_limit", handoff_path(child_session_path), readable)


def project_notification_data(data: dict[str, Any]) -> dict[str, Any]:
    """Project a validated stored completion without changing its stored shape."""
    if data.get("status") != "completed":
        return data
    projection = parse_turn_limit_result(
        data.get("text"), data.get("child_session_path", "")
    )
    if projection is None:
        return data
    return {
        **data,
        "status": projection.status,
        "handoff_path": projection.handoff_path,
        "text": projection.readable_text,
    }


def project_tool_result_data(result: dict[str, Any]) -> dict[str, Any]:
    """Project a persisted tool result through the same epoch-free marker."""
    structured = result.get("structured_content")
    if not isinstance(structured, dict) or structured.get("status") != "completed":
        return result
    child_path = structured.get("child_session_path")
    if type(child_path) is not str:
        return result
    projection = parse_turn_limit_result(result.get("content"), child_path)
    if projection is None:
        return result
    return {
        **result,
        "content": projection.readable_text,
        "structured_content": {
            **structured,
            "status": projection.status,
            "handoff_path": projection.handoff_path,
        },
    }


def project_payload(payload: dict[str, object], child_session_path: str) -> dict[str, object]:
    """Return the public non-error payload for a live turn-limit completion."""
    projected = dict(payload)
    structured = dict(projected.get("structuredContent", {}))
    structured.update(
        {"status": "turn_limit", "handoff_path": handoff_path(child_session_path)}
    )
    projected["structuredContent"] = structured
    return projected


def background_turn_limit_state(
    *,
    result: object,
    child_turns: Callable[[], int],
    child_loop: Any,
    child_instance_id: str,
    child_session_path: str,
    final_text: str,
    background_owner: Any,
) -> tuple[bool, int, bytes | None]:
    structured = (
        result.get("structuredContent", {}) if isinstance(result, dict) else {}
    )
    control = getattr(child_loop, "child_turn_control", None)
    turn_limited = control is not None and control.exhausted
    turns_used = max(child_turns(), int(structured.get("turns_used", 0)))
    if not turn_limited:
        return False, turns_used, None
    return (
        True,
        turns_used,
        background_fallback_handoff(
            control=control,
            child_instance_id=child_instance_id,
            child_session_path=child_session_path,
            final_text=final_text,
            background_owner=background_owner,
            child_loop=child_loop,
        ),
    )


def background_fallback_handoff(
    *,
    control: ChildTurnControl,
    child_instance_id: str,
    child_session_path: str,
    final_text: str,
    background_owner: Any,
    child_loop: Any,
) -> bytes:
    children = background_owner.owned_running(child_instance_id)
    tasks = tuple(
        (task.task_id, task.headline)
        for task in child_loop.tool_registry.background_tasks.snapshot()
        if task.running
    )
    return fallback_handoff(
        control=control,
        child_instance_id=child_instance_id,
        child_session_path=child_session_path,
        final_text=final_text,
        children=children,
        tasks=tasks,
    )


async def write_background_turn_limit(
    store: ConversationStore,
    handoff_bytes: bytes,
    receipt: str,
    child_session_path: str,
) -> str:
    await store._commit_storage_command(
        _EnsureTurnLimitHandoffCommand(content=handoff_bytes)
    )
    return build_turn_limit_result(receipt, child_session_path)


async def finalize_exhausted_child(
    *,
    control: ChildTurnControl | None,
    text: str,
    child_session_path: str,
    child_instance_id: str,
    child_store: ConversationStore | None,
    background: bool,
    owned_children: Callable[[], Sequence[tuple[str, str]]],
    owned_tasks: Callable[[], Sequence[tuple[str, str]]],
    terminal_result: Callable[..., dict[str, object]],
    finish_turn_limit: Callable[[str], dict[str, object]] | None,
    record_report: Callable[[str], None] | None,
) -> dict[str, object] | None:
    if control is None or not control.exhausted:
        return None
    readable_text = text.strip() or "no final assistant text"
    if record_report is not None and text.strip():
        record_report(text)
    if not background:
        if child_store is None:
            raise RuntimeError("limited child has no session store")
        content = fallback_handoff(
            control=control,
            child_instance_id=child_instance_id,
            child_session_path=child_session_path,
            final_text=text,
            children=owned_children(),
            tasks=owned_tasks(),
        )
        await child_store._commit_storage_command(
            _EnsureTurnLimitHandoffCommand(content=content)
        )
    if background:
        result = terminal_result(state="completed", text=readable_text)
    else:
        if finish_turn_limit is None:
            raise RuntimeError("limited child has no terminal commit")
        result = finish_turn_limit(readable_text)
    return project_payload(result, child_session_path)


def finish_foreground_turn_limit(
    *,
    store: ConversationStore,
    child_session_path: str,
    readable_text: str,
    turns_used: int,
    stats: dict[str, object],
    build_result: Callable[..., dict[str, object]],
) -> dict[str, object]:
    payload = build_result(
        readable_text,
        state="completed",
        stats=stats,
        persist_receipt=False,
    )
    receipt = payload["content"][0]["text"]
    store.finish_agent_lifecycle(
        "completed",
        final_result=build_turn_limit_result(receipt, child_session_path),
        turns_used=turns_used,
        canonical_receipt=True,
    )
    return payload


def make_publish_child_lifecycle(
    *,
    account_context: Any,
    selected_accounts: dict[str, object],
    child_store: Any,
    child_instance_id: str,
    child_depth: int,
    preset: Any,
    description: str,
    child_cwd: str,
) -> Callable[[], None]:
    def publish_child_lifecycle() -> None:
        account_store = getattr(account_context, "store", None)
        protected_accounts = {
            provider: account
            for provider, account in selected_accounts.items()
            if account_store is not None and account_store.contains(account)
        }
        guard = (
            account_store.pin_guard(protected_accounts)
            if protected_accounts
            else nullcontext()
        )
        with guard:
            child_store.start_agent_lifecycle(
                handle=child_instance_id,
                started_at=_now(),
                depth=child_depth,
                agent_type=preset.name,
                description=description,
                cwd=child_cwd,
                selected_accounts={
                    provider: str(account)
                    for provider, account in selected_accounts.items()
                }
                or None,
            )

    return publish_child_lifecycle


def make_child_progress_callbacks(
    *,
    loop: Any,
    child_store: Any,
    child_marker_key: str,
    child_tool_call_id: str,
) -> tuple[Callable[[int], None], Callable[[], None]]:
    def update_turns(turns: int) -> None:
        loop._agent_child_turns[child_tool_call_id] = turns
        child_store.update_agent_lifecycle(turns_used=turns)
        loop.store.update_agent_child_turns(child_marker_key, turns)

    def record_tool_call() -> None:
        child_store.update_agent_lifecycle(tool_calls=1)

    return update_turns, record_tool_call


def make_finish_limited_child_turn(
    *,
    store: ConversationStore,
    child_session_path: str,
    turns_used: Callable[[], int],
    stats: Callable[[int], dict[str, object]],
    build_result: Callable[..., dict[str, object]],
) -> Callable[[str], dict[str, object]]:
    def finish(readable_text: str) -> dict[str, object]:
        used = turns_used()
        return finish_foreground_turn_limit(
            store=store,
            child_session_path=child_session_path,
            readable_text=readable_text,
            turns_used=used,
            stats=stats(used),
            build_result=build_result,
        )

    return finish


def fallback_handoff(
    *,
    control: ChildTurnControl,
    child_instance_id: str,
    child_session_path: str,
    final_text: str,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
) -> bytes:
    lines = [
        "# child turn limit handoff",
        "",
        f"the configured turn limit was reached ({control.used}/{control.limit} completions).",
        f"child: {child_instance_id}",
        f"session: {child_session_path}",
        "",
        "## final assistant text",
        _without_agent_receipt_suffix(final_text).strip()
        or "no final assistant text",
    ]
    if children:
        lines.extend(("", "## unfinished direct children"))
        lines.extend(f"- {handle}: {description}" for handle, description in children)
    if tasks:
        lines.extend(("", "## unfinished tasks"))
        lines.extend(f"- {task_id}: {headline}" for task_id, headline in tasks)
    return ("\n".join(lines) + "\n").encode()
