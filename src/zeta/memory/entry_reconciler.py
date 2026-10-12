"""Dormant format-2 project-memory reconciliation.

One deep interface owns bounded provider requests, strict operation parsing,
evidence policy, dependency rejection, expiry, and CAS regeneration. Production
format-1 reconciliation remains in :mod:`zeta.memory.reconciler`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from zeta.project_errors import ProjectRegistryError
from zeta.providers.retry_policy import ProviderRetryBudget, use_retry_budget

from .entry_request import (
    _MAX_OPERATIONS,
    _MAX_PROPOSED_TEXT_BYTES,
    _prepare_request,
    _stored_rank,
)
from .entry_store import (
    MAX_ENTRY_TEXT_BYTES,
    UNSET,
    AddOperation,
    ExpireOperation,
    MemoryEntry,
    MemoryOperation,
    MemorySource,
    MemoryState,
    ResolveOperation,
    SupersedeOperation,
    UpdateOperation,
    apply_operations,
)
from .provider import use_response_byte_limit
from .reconciler import (
    PreparedRequest,
    ReconciliationError,
    ReconciliationResponse,
    Transcript,
    _is_user_authored_row,
    _sanitized,
    _transcript_authorship,
    _unsafe_reason,
)
from .safety import contains_secret

if TYPE_CHECKING:
    from zeta.project_registry import ProjectRegistry


EntryInvokeResult = str | ReconciliationResponse
EntryInvoke = Callable[[str], EntryInvokeResult | Awaitable[EntryInvokeResult]]
ReconciliationKey = str | Callable[[int, int], str]

_MAX_RESPONSE_BYTES = 32 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_REASON_BYTES = 1024
_COMPLETION_KINDS = frozenset({"state", "backlog", "threads", "commitments"})
_CODE_LITERAL = re.compile(r"`([^`\\n]{1,256})`")
_DURABLE_LITERAL_CUES = (
    "validated",
    "established",
    "recorded",
    "current",
    "in progress",
    "now complete",
    "completed",
    "supersedes",
    "selected",
    "verification",
    "decision",
    "remember",
)
SUPPORTED_MEMORY_FORMATS = frozenset({1, 2})

_HIGHEST_PRIORITY_WORDS = (
    "correction",
    "actually",
    "instead",
    "no longer",
    "changed to",
    "i decide",
    "we decide",
    "decision:",
)


@dataclass(frozen=True, slots=True)
class EntryReconciliationResult:
    """Result of one complete scheduled format-2 reconciliation attempt."""

    changed_entry_ids: tuple[str, ...]
    rejected_groups: tuple[str, ...]
    usage: Mapping[str, int]
    seq_start: int
    seq_end: int
    reconciliation_key: str


@dataclass(frozen=True, slots=True)
class _ParsedOperation:
    operation: MemoryOperation
    targets: frozenset[str]
    source_rank: int
    direct_user: bool
    observed_at: str


class EntryReconciliationFailure(ReconciliationError):
    """A failed scheduled attempt with provider usage retained for the ledger."""

    def __init__(self, message: str, usage: Mapping[str, int]) -> None:
        self.usage = dict(usage)
        super().__init__(message)


class _ProposalError(ReconciliationError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _exact(
    value: object, allowed: set[str], required: set[str], index: int
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _ProposalError((f"operations[{index}] must be an object",))
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    errors: list[str] = []
    if unknown:
        errors.append(f"operations[{index}] unknown fields: {', '.join(unknown)}")
    if missing:
        errors.append(f"operations[{index}] missing fields: {', '.join(missing)}")
    if errors:
        raise _ProposalError(errors)
    return value


def _message_text(row: Mapping[str, Any]) -> str:
    data = row.get("data")
    message = data.get("message") if isinstance(data, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if isinstance(item, Mapping)
        )
    return ""


def _source_rank(rows: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    authorships = [_transcript_authorship(row) for row in rows]
    direct_user_rows = [
        row
        for row, authorship in zip(rows, authorships, strict=True)
        if authorship == "user"
    ]
    if direct_user_rows:
        text = " ".join(_message_text(row).lower() for row in direct_user_rows)
        return (1 if any(word in text for word in _HIGHEST_PRIORITY_WORDS) else 2), True
    if any(value in {"skill_expansion", "slash_expansion"} for value in authorships):
        return 3, False
    if "agent" in authorships and "tool_output" in authorships:
        return 4, False
    if "agent" in authorships:
        return 5, False
    return 6, False


def _observed_at(rows: Sequence[Mapping[str, Any]], fallback: str) -> str:
    values: list[str] = []
    for row in rows:
        data = row.get("data")
        value = data.get("created_at") if isinstance(data, Mapping) else None
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            values.append(
                parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
            )
    return max(values, default=fallback)


def _sources(
    raw: object,
    transcript: Transcript,
    *,
    now: str,
    index: int,
) -> tuple[tuple[MemorySource, ...], int, bool, str]:
    if not isinstance(raw, list) or not raw:
        raise _ProposalError((f"operations[{index}].sources must be a non-empty list",))
    by_seq = {
        row.get("seq"): row for row in transcript.rows if type(row.get("seq")) is int
    }
    parsed_ranges: list[tuple[int, int, tuple[str, ...], str, int, bool]] = []
    all_rows: list[Mapping[str, Any]] = []
    for source_index, item in enumerate(raw):
        source = _exact(item, {"seq_start", "seq_end"}, {"seq_start", "seq_end"}, index)
        start, end = source["seq_start"], source["seq_end"]
        if type(start) is not int or type(end) is not int or start > end:
            raise _ProposalError(
                (f"operations[{index}].sources[{source_index}] has an invalid range",)
            )
        rows = [by_seq[seq] for seq in range(start, end + 1) if seq in by_seq]
        if not rows:
            raise _ProposalError(
                (
                    f"operations[{index}].sources[{source_index}] is outside the transcript",
                )
            )
        origins = tuple(dict.fromkeys(_transcript_authorship(row) for row in rows))
        observed_at = _observed_at(rows, now)
        rank, direct = _source_rank(rows)
        parsed_ranges.append((start, end, origins, observed_at, rank, direct))
        all_rows.extend(rows)
    operation_rank = min(item[4] for item in parsed_ranges)
    operation_direct = any(
        direct and rank == operation_rank for _, _, _, _, rank, direct in parsed_ranges
    )
    sources = tuple(
        MemorySource(transcript.session_id, start, end, origins, observed_at, rank)
        for start, end, origins, observed_at, rank, _ in parsed_ranges
    )
    return sources, operation_rank, operation_direct, _observed_at(all_rows, now)


def _text(value: object, label: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _ProposalError((f"operations[{index}].{label} must be text",))
    if len(value.encode()) > MAX_ENTRY_TEXT_BYTES:
        raise _ProposalError((f"operations[{index}].{label} exceeds the byte limit",))
    return value


def _kind(value: object, state: MemoryState, index: int) -> str:
    kind = _text(value, "kind", index)
    if kind not in {item.key for item in state.schema.kinds}:
        raise _ProposalError((f"operations[{index}].kind is not in the schema",))
    return kind


def _expiry(
    value: object, state: MemoryState, kind: str, now: str, index: int
) -> str | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 3650:
        raise _ProposalError((f"operations[{index}].expires_in_days is invalid",))
    return (
        (datetime.fromisoformat(now) + timedelta(days=value))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _temporal(value: object, label: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _ProposalError((f"operations[{index}].{label} is invalid",))
    candidate = f"{value}T00:00:00.000000Z" if len(value) == 10 else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise _ProposalError((f"operations[{index}].{label} is invalid",)) from exc
    if parsed.tzinfo is None:
        raise _ProposalError((f"operations[{index}].{label} requires a timezone",))
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _operation_expiry(
    raw: Mapping[str, Any],
    state: MemoryState,
    kind: str,
    now: str,
    index: int,
    valid_until: str | None,
) -> Any:
    if "expires_in_days" in raw:
        return _expiry(raw["expires_in_days"], state, kind, now, index)
    if valid_until is not None:
        grace = 14 if kind == "commitments" else 0
        return (
            (datetime.fromisoformat(valid_until) + timedelta(days=grace))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    return UNSET


def _parse_operation(
    value: object, index: int, state: MemoryState, transcript: Transcript, now: str
) -> _ParsedOperation:
    if not isinstance(value, dict) or not isinstance(value.get("op"), str):
        raise _ProposalError((f"operations[{index}].op is required",))
    op = value["op"]
    common = {"op", "sources", "reason"}
    optionals = {"valid_from", "valid_until", "expires_in_days"}
    if op == "add":
        raw = _exact(
            value,
            common | optionals | {"kind", "text"},
            common | {"kind", "text"},
            index,
        )
    elif op == "update":
        raw = _exact(
            value,
            common | optionals | {"target", "kind", "text"},
            common | {"target"},
            index,
        )
        if not any(
            name in raw
            for name in (
                "kind",
                "text",
                "valid_from",
                "valid_until",
                "expires_in_days",
            )
        ):
            raise _ProposalError((f"operations[{index}] update has no changes",))
    elif op == "supersede":
        raw = _exact(
            value,
            common | optionals | {"targets", "kind", "text"},
            common | {"targets", "kind", "text"},
            index,
        )
    elif op == "resolve":
        raw = _exact(value, common | {"target"}, common | {"target"}, index)
    else:
        raise _ProposalError((f"operations[{index}].op is unknown: {op}",))
    reason = _text(raw["reason"], "reason", index)
    if len(reason.encode()) > _MAX_REASON_BYTES:
        raise _ProposalError((f"operations[{index}].reason exceeds the byte limit",))
    sources, rank, direct, observed = _sources(
        raw["sources"], transcript, now=now, index=index
    )
    valid_from = _temporal(raw.get("valid_from"), "valid_from", index)
    valid_until = _temporal(raw.get("valid_until"), "valid_until", index)
    if valid_from is not None and valid_until is not None and valid_from > valid_until:
        raise _ProposalError((f"operations[{index}] has an invalid validity interval",))
    if op == "add":
        kind = _kind(raw["kind"], state, index)
        expires_at = _operation_expiry(raw, state, kind, now, index, valid_until)
        operation: MemoryOperation = AddOperation(
            kind,
            _text(raw["text"], "text", index),
            sources,
            expires_at=expires_at,
            valid_from=valid_from,
            valid_until=valid_until if valid_until is not None else UNSET,
        )
        targets = frozenset()
    elif op == "update":
        target = _text(raw["target"], "target", index)
        kind = raw.get("kind")
        target_kind = str(kind or _target_kind(state, target))
        expires_at = _operation_expiry(raw, state, target_kind, now, index, valid_until)
        operation = UpdateOperation(
            target,
            sources,
            text=_text(raw["text"], "text", index) if "text" in raw else None,
            kind=_kind(kind, state, index) if kind is not None else None,
            expires_at=expires_at,
            valid_from=valid_from,
            valid_until=valid_until if valid_until is not None else UNSET,
        )
        targets = frozenset((target,))
    elif op == "supersede":
        raw_targets = raw["targets"]
        if (
            not isinstance(raw_targets, list)
            or not raw_targets
            or any(not isinstance(item, str) for item in raw_targets)
        ):
            raise _ProposalError((f"operations[{index}].targets is invalid",))
        kind = _kind(raw["kind"], state, index)
        expires_at = _operation_expiry(raw, state, kind, now, index, valid_until)
        operation = SupersedeOperation(
            tuple(raw_targets),
            kind,
            _text(raw["text"], "text", index),
            sources,
            expires_at=expires_at,
            valid_from=valid_from,
            valid_until=valid_until if valid_until is not None else UNSET,
        )
        targets = frozenset(raw_targets)
    else:
        target = _text(raw["target"], "target", index)
        operation = ResolveOperation(target, sources)
        targets = frozenset((target,))
    return _ParsedOperation(operation, targets, rank, direct, observed)


def _target_kind(state: MemoryState, target: str) -> str:
    entry = state.entries.get(target)
    return entry.kind if isinstance(entry, MemoryEntry) else ""


def _cited_code_literals(
    operation: MemoryOperation, transcript: Transcript
) -> tuple[str, ...]:
    sources = getattr(operation, "sources", ())
    ranges = tuple((source.seq_start, source.seq_end) for source in sources)
    if not ranges:
        return ()
    literals: list[str] = []
    for row in transcript.rows:
        seq = row.get("seq")
        if (
            type(seq) is not int
            or not _is_user_authored_row(row)
            or not any(start <= seq <= end for start, end in ranges)
        ):
            continue
        encoded = json.dumps(row, ensure_ascii=False)
        literals.extend(
            literal
            for literal in _CODE_LITERAL.findall(encoded)
            if not contains_secret(literal)
        )
    return tuple(dict.fromkeys(literals))


def _parse(
    raw_text: str, state: MemoryState, transcript: Transcript, now: str
) -> tuple[_ParsedOperation, ...]:
    if len(raw_text.encode()) > _MAX_RESPONSE_BYTES:
        raise _ProposalError(("response exceeds 32768 bytes",))
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise _ProposalError(
            (f"response is not valid JSON at character {exc.pos}",)
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"operations"}
        or not isinstance(value.get("operations"), list)
    ):
        raise _ProposalError(("response must contain exactly an operations list",))
    values = value["operations"]
    if len(values) > _MAX_OPERATIONS:
        raise _ProposalError((f"operations exceeds {_MAX_OPERATIONS} items",))
    parsed: list[_ParsedOperation] = []
    errors: list[str] = []
    text_bytes = 0
    for index, item in enumerate(values):
        try:
            operation = _parse_operation(item, index, state, transcript, now)
            text = getattr(operation.operation, "text", None)
            if isinstance(text, str):
                text_bytes += len(text.encode())
                missing_literals = tuple(
                    literal
                    for literal in _cited_code_literals(operation.operation, transcript)
                    if literal not in text
                )
                if missing_literals:
                    raise _ProposalError(
                        (
                            f"operations[{index}].text omits cited exact code literal(s): "
                            + ", ".join(missing_literals),
                        )
                    )
            parsed.append(operation)
        except _ProposalError as exc:
            errors.extend(exc.errors)
    if not parsed:
        for row in transcript.rows:
            encoded = json.dumps(row, ensure_ascii=False)
            lowered = encoded.lower()
            if not _is_user_authored_row(row) or not any(
                cue in lowered for cue in _DURABLE_LITERAL_CUES
            ):
                continue
            literals = tuple(
                literal
                for literal in _CODE_LITERAL.findall(encoded)
                if not contains_secret(literal)
            )
            if literals:
                errors.append(
                    "operations omits durable exact code literal(s): "
                    + ", ".join(literals)
                )
    if text_bytes > _MAX_PROPOSED_TEXT_BYTES:
        errors.append(
            f"cumulative operation text exceeds {_MAX_PROPOSED_TEXT_BYTES} bytes"
        )
    if errors:
        raise _ProposalError(errors)
    return tuple(parsed)


def _target_transition_error(
    operation: MemoryOperation,
    entry: MemoryEntry,
    *,
    proposal_rank: int | None,
    direct_user: bool,
    observed_at: str,
    now: str,
) -> str | None:
    """Authorize every automatic content or status transition in one place."""
    if isinstance(operation, ExpireOperation):
        if entry.expires_at is None or datetime.fromisoformat(
            entry.expires_at
        ) > datetime.fromisoformat(now):
            return "entry has not reached its configured expiry"
        return None
    if isinstance(operation, ResolveOperation) and entry.kind not in _COMPLETION_KINDS:
        return "entry kind does not allow completion-based resolution"
    if proposal_rank is None:
        return "transition has no durable evidence rank"
    if entry.accepted_at is not None:
        if not direct_user:
            return "accepted entry requires direct user evidence"
        if observed_at <= entry.seen_at:
            return "accepted entry requires newer direct user evidence"
    target_rank = _stored_rank(entry)
    if proposal_rank > target_rank:
        return "weaker evidence cannot change stronger evidence"
    if proposal_rank == target_rank and observed_at <= entry.seen_at:
        return "non-newer evidence cannot change existing evidence"
    return None


def _direct_user_procedure_fact(item: _ParsedOperation, text: str) -> bool:
    lowered = text.lower()
    return (
        item.direct_user
        and any(word in lowered for word in ("validated", "procedure"))
        and re.search(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", text) is not None
        and not any(
            phrase in lowered
            for phrase in (
                "ignore previous",
                "ignore prior",
                "ignore system",
                "system prompt",
                "developer prompt",
                "you are chatgpt",
                "you are an assistant",
                "you are an agent",
            )
        )
    )


def _semantic_error(
    item: _ParsedOperation,
    state: MemoryState,
    now: str,
    visible_entry_ids: frozenset[str],
) -> str | None:
    operation = item.operation
    text = getattr(operation, "text", None)
    if isinstance(text, str):
        unsafe = _unsafe_reason(text)
        if unsafe and not _direct_user_procedure_fact(item, text):
            return f"unsafe {unsafe} text"
        if isinstance(operation, AddOperation):
            proposed_literals = set(
                re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", text)
            )
            for entry in state.entries.values():
                if not isinstance(entry, MemoryEntry) or entry.status != "active":
                    continue
                existing_literals = set(
                    re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", entry.text)
                )
                if text == entry.text or (
                    proposed_literals and proposed_literals <= existing_literals
                ):
                    return "add duplicates an existing active entry"
    if item.source_rank >= 6:
        return "uncorroborated harness or tool evidence"
    for target in item.targets:
        if target not in visible_entry_ids:
            return "target was not included in the request"
        entry = state.entries.get(target)
        if not isinstance(entry, MemoryEntry) or entry.status != "active":
            return "target is missing or inactive"
        error = _target_transition_error(
            operation,
            entry,
            proposal_rank=item.source_rank,
            direct_user=item.direct_user,
            observed_at=item.observed_at,
            now=now,
        )
        if error is not None:
            return error
    return None


def _dependency_groups(
    items: tuple[_ParsedOperation, ...],
) -> tuple[tuple[_ParsedOperation, ...], ...]:
    remaining = list(items)
    groups: list[tuple[_ParsedOperation, ...]] = []
    while remaining:
        group = [remaining.pop(0)]
        targets = set(group[0].targets)
        changed = True
        while changed:
            changed = False
            for item in tuple(remaining):
                if targets and targets.intersection(item.targets):
                    remaining.remove(item)
                    group.append(item)
                    targets.update(item.targets)
                    changed = True
        groups.append(tuple(group))
    return tuple(groups)


def _select_groups(
    items: tuple[_ParsedOperation, ...],
    state: MemoryState,
    key: str,
    now: str,
    visible_entry_ids: frozenset[str],
) -> tuple[tuple[MemoryOperation, ...], tuple[str, ...]]:
    accepted: list[MemoryOperation] = []
    rejected: list[str] = []
    working_state = state
    for group_index, group in enumerate(_dependency_groups(items)):
        errors = tuple(
            error
            for item in group
            if (error := _semantic_error(item, working_state, now, visible_entry_ids))
        )
        error = errors[0] if errors else None
        if error is None:
            try:
                working_state, _ = apply_operations(
                    working_state,
                    tuple(item.operation for item in group),
                    reconciliation_key=key,
                    automatic=True,
                    now=now,
                )
            except ProjectRegistryError as exc:
                error = str(exc)
        if error is not None:
            rejected.append(f"group[{group_index}]: {error}"[:512])
        else:
            accepted.extend(item.operation for item in group)
    return tuple(accepted), tuple(rejected)


def _expired_operations(state: MemoryState, now: str) -> tuple[ExpireOperation, ...]:
    operations: list[ExpireOperation] = []
    for entry in state.entries.values():
        if not isinstance(entry, MemoryEntry) or entry.status != "active":
            continue
        operation = ExpireOperation(entry.id, "configured expiry reached")
        if (
            _target_transition_error(
                operation,
                entry,
                proposal_rank=None,
                direct_user=False,
                observed_at=now,
                now=now,
            )
            is None
        ):
            operations.append(operation)
    return tuple(operations)


def _repair_prompt(
    request: PreparedRequest, error: _ProposalError, invalid: str
) -> str:
    errors = json.dumps(list(error.errors), ensure_ascii=False, separators=(",", ":"))
    safe_invalid = str(_sanitized(invalid))
    bounded = safe_invalid.encode()[:_MAX_RESPONSE_BYTES].decode(
        "utf-8", errors="ignore"
    )
    prefix = f"Your prior JSON failed structural validation. Indexed errors: {errors}\nInvalid response: {bounded}\nReturn corrected JSON.\n"
    available = _MAX_REQUEST_BYTES - len(prefix.encode())
    if available < 0:
        prefix = prefix[:2048]
        available = _MAX_REQUEST_BYTES - len(prefix.encode())
    prompt = prefix + request.prompt.encode()[:available].decode(
        "utf-8", errors="ignore"
    )
    return prompt.encode()[:_MAX_REQUEST_BYTES].decode("utf-8", errors="ignore")


async def _invoke(
    invoke: EntryInvoke, prompt: str, budget: ProviderRetryBudget
) -> tuple[str, dict[str, int], bool]:
    with use_retry_budget(budget), use_response_byte_limit(_MAX_RESPONSE_BYTES):
        response = invoke(prompt)
        if inspect.isawaitable(response):
            response = await response
    if isinstance(response, ReconciliationResponse):
        return (
            response.text,
            {
                key: value
                for key, value in response.usage.items()
                if type(value) is int and value >= 0
            },
            response.truncated,
        )
    return response, {}, False


def _sum_usage(total: dict[str, int], addition: Mapping[str, int]) -> dict[str, int]:
    result = dict(total)
    for key, value in addition.items():
        result[key] = result.get(key, 0) + value
    return result


async def reconcile_entry_range(
    *,
    registry: ProjectRegistry,
    project_id: str,
    transcript: Transcript,
    reconciliation_key: ReconciliationKey,
    invoke: EntryInvoke,
    cas_retries: int,
    as_of: date,
    now: str,
) -> EntryReconciliationResult:
    """Reconcile one durable transcript fragment into format-2 state."""
    usage: dict[str, int] = {}
    budget = ProviderRetryBudget()
    for attempt in range(cas_retries):
        snapshot = await asyncio.to_thread(registry._entry_memory_state, project_id)
        request = _prepare_request(transcript, snapshot.state, as_of=as_of)
        selected_start = min(int(row["seq"]) for row in request.transcript.rows)
        selected_end = max(int(row["seq"]) for row in request.transcript.rows)
        key = (
            reconciliation_key(selected_start, selected_end)
            if callable(reconciliation_key)
            else reconciliation_key
        )
        raw, first_usage, first_truncated = await _invoke(
            invoke, request.prompt, budget
        )
        usage = _sum_usage(usage, first_usage)
        try:
            if first_truncated:
                raise _ProposalError(("response exceeds 32768 bytes",))
            parsed = _parse(raw, snapshot.state, request.transcript, now)
        except _ProposalError as first_error:
            try:
                repaired, repair_usage, repair_truncated = await _invoke(
                    invoke, _repair_prompt(request, first_error, raw), budget
                )
                usage = _sum_usage(usage, repair_usage)
                if repair_truncated:
                    raise _ProposalError(("response exceeds 32768 bytes",))
                parsed = _parse(repaired, snapshot.state, request.transcript, now)
            except Exception as exc:
                raise EntryReconciliationFailure(str(exc), usage) from exc
        operations, rejected = _select_groups(
            parsed, snapshot.state, key, now, request.visible_entry_ids
        )
        operations = (*_expired_operations(snapshot.state, now), *operations)
        if not operations:
            return EntryReconciliationResult(
                (), rejected, usage, selected_start, selected_end, key
            )
        evidence = (
            transcript.session_id,
            selected_start,
            selected_end,
        )
        try:
            result = await asyncio.to_thread(
                registry._compare_and_swap_entries,
                project_id,
                expected_digest=snapshot.digest,
                operations=tuple(operations),
                reconciliation_key=key,
                automatic=True,
                evidence=evidence,
                rejected_groups=rejected,
                now=now,
            )
        except ProjectRegistryError as exc:
            if "digest mismatch" in str(exc) and attempt + 1 < cas_retries:
                continue
            raise EntryReconciliationFailure(str(exc), usage) from exc
        changed = tuple(
            dict.fromkeys(
                entry_id
                for receipt in result.receipts
                for entry_id in (*receipt.target_ids, *receipt.result_ids)
            )
        )
        return EntryReconciliationResult(
            changed, rejected, usage, selected_start, selected_end, key
        )
    raise ReconciliationError("format-2 memory changed during reconciliation")
