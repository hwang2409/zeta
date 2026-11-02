"""Bounded typed facts retained across deterministic context eviction."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..context_accounting import message_token_count
from ..protocol.types import Message, MessageRole, TextContent

FACT_FLOOR_CHAR_CAP = 4_096
_OLDEST_LIMIT = 8
_NEWEST_LIMIT = 16
_TYPE_ORDER = {
    "pr": 0,
    "pr_verdict": 1,
    "pr_head": 2,
    "child_status": 3,
    "child_head": 4,
}
_VERDICTS = frozenset(
    {
        "pass",
        "fail",
        "inconclusive",
        "MERGE-READY",
        "NOT READY",
        "READY TO IMPLEMENT",
        "NEEDS REVISION",
    }
)
_CHILD_STATUSES = frozenset({"completed", "failed", "canceled"})
_HANDLE = r"[A-Za-z0-9._:-]{1,80}"
_SHA = r"[0-9a-f]{7,40}"
_PR = re.compile(r"\bPR\s*#([1-9][0-9]{0,9})\b", re.IGNORECASE)
_CHILD = re.compile(rf"\bchild\s+({_HANDLE})(?=\s|$)", re.IGNORECASE)
_VERDICT = re.compile(
    r"\b(?:verdict\s*[=:]\s*)?(MERGE-READY|NOT READY|READY TO IMPLEMENT|NEEDS REVISION|pass|fail|inconclusive)\b"
)
_STATUS = re.compile(r"\bstatus\s*[=:]\s*(completed|failed|canceled)\b")
_HEAD = re.compile(rf"\b(?:sha|head)\s*[=:]\s*({_SHA})\b")

FactFloor = dict[str, object]
Aggregate = dict[str, object]
Occurrence = tuple[dict[str, object], int, int]


def _fact_key(fact: Mapping[str, object]) -> tuple[object, ...]:
    kind = fact["type"]
    if kind in {"pr", "pr_verdict", "pr_head"}:
        return (kind, fact["pr"])
    return (kind, fact["handle"])


def _canonical_key(aggregate: Mapping[str, object]) -> bytes:
    fact = aggregate["fact"]
    assert isinstance(fact, Mapping)
    return json.dumps(_fact_key(fact), separators=(",", ":")).encode()


def _selection_key(
    aggregate: Mapping[str, object], sequence_field: str
) -> tuple[object, ...]:
    fact = aggregate["fact"]
    assert isinstance(fact, Mapping)
    return (
        aggregate[sequence_field],
        _TYPE_ORDER[str(fact["type"])],
        _canonical_key(aggregate),
    )


def _valid_fact(value: object) -> bool:
    if type(value) is not dict or type(value.get("type")) is not str:
        return False
    kind = value["type"]
    if kind == "pr":
        return set(value) == {"type", "pr"} and _valid_pr(value.get("pr"))
    if kind == "pr_verdict":
        return (
            set(value) == {"type", "pr", "verdict"}
            and _valid_pr(value.get("pr"))
            and value.get("verdict") in _VERDICTS
        )
    if kind == "pr_head":
        return (
            set(value) == {"type", "pr", "sha"}
            and _valid_pr(value.get("pr"))
            and _valid_sha(value.get("sha"))
        )
    if kind == "child_status":
        return (
            set(value) == {"type", "handle", "status"}
            and _valid_handle(value.get("handle"))
            and value.get("status") in _CHILD_STATUSES
        )
    if kind == "child_head":
        return (
            set(value) == {"type", "handle", "sha"}
            and _valid_handle(value.get("handle"))
            and _valid_sha(value.get("sha"))
        )
    return False


def _valid_pr(value: object) -> bool:
    return type(value) is int and 0 < value <= 9_999_999_999


def _valid_handle(value: object) -> bool:
    return type(value) is str and re.fullmatch(_HANDLE, value) is not None


def _valid_sha(value: object) -> bool:
    return type(value) is str and re.fullmatch(_SHA, value) is not None


def _valid_aggregate(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "fact",
        "first_seq",
        "last_seq",
        "value_rank",
    }:
        return False
    first = value.get("first_seq")
    last = value.get("last_seq")
    rank = value.get("value_rank")
    return (
        _valid_fact(value.get("fact"))
        and type(first) is int
        and first > 0
        and type(last) is int
        and last >= first
        and type(rank) is list
        and len(rank) == 2
        and type(rank[0]) is int
        and rank[0] == last
        and type(rank[1]) is int
        and rank[1] >= 0
    )


def validate_fact_floor(value: object) -> bool:
    if type(value) is not dict or set(value) != {"version", "oldest", "newest"}:
        return False
    oldest = value.get("oldest")
    newest = value.get("newest")
    if (
        value.get("version") != 1
        or type(oldest) is not list
        or type(newest) is not list
        or len(oldest) > _OLDEST_LIMIT
        or len(newest) > _NEWEST_LIMIT
        or not all(_valid_aggregate(item) for item in [*oldest, *newest])
    ):
        return False
    by_key: dict[tuple[object, ...], object] = {}
    for item in [*oldest, *newest]:
        key = _fact_key(item["fact"])
        if key in by_key and by_key[key] != item:
            return False
        by_key[key] = item
    expected_oldest = sorted(oldest, key=lambda item: _selection_key(item, "first_seq"))
    expected_newest = sorted(
        newest, key=lambda item: _selection_key(item, "last_seq"), reverse=True
    )
    return oldest == expected_oldest and newest == expected_newest


def _structured_facts(value: Mapping[str, object]) -> list[dict[str, object]]:
    pr_value = value.get("pr", value.get("pr_number"))
    pr = pr_value if _valid_pr(pr_value) else None
    handle_value = value.get("child_instance_id")
    handle = handle_value if _valid_handle(handle_value) else None
    verdict = value.get("verdict") if value.get("verdict") in _VERDICTS else None
    status = value.get("status") if value.get("status") in _CHILD_STATUSES else None
    sha = value.get("head_sha") if _valid_sha(value.get("head_sha")) else None
    facts: list[dict[str, object]] = []
    if pr is not None:
        facts.append({"type": "pr", "pr": pr})
    if pr is not None and verdict is not None:
        facts.append({"type": "pr_verdict", "pr": pr, "verdict": verdict})
    if pr is not None and sha is not None:
        facts.append({"type": "pr_head", "pr": pr, "sha": sha})
    if handle is not None and status is not None:
        facts.append({"type": "child_status", "handle": handle, "status": status})
    if handle is not None and sha is not None:
        facts.append({"type": "child_head", "handle": handle, "sha": sha})
    return facts


def _identified_records(text: str) -> list[tuple[int, str, object, str]]:
    records: list[tuple[int, str, object, str]] = []
    line_offset = 0
    for line in text.splitlines(keepends=True) or [text]:
        identifiers: list[tuple[int, str, object]] = [
            (match.start(), "pr", int(match.group(1))) for match in _PR.finditer(line)
        ]
        identifiers.extend(
            (match.start(), "child", match.group(1)) for match in _CHILD.finditer(line)
        )
        identifiers.sort()
        for index, (position, kind, identifier) in enumerate(identifiers):
            end = (
                identifiers[index + 1][0]
                if index + 1 < len(identifiers)
                else len(line)
            )
            records.append(
                (line_offset + position, kind, identifier, line[position:end])
            )
        line_offset += len(line)
    return records


def _record_text_facts(
    offset: int, kind: str, identifier: object, record: str
) -> list[tuple[int, dict[str, object]]]:
    if kind == "pr":
        facts = [(offset, {"type": "pr", "pr": identifier})]
        facts.extend(
            (
                offset + match.start(),
                {"type": "pr_verdict", "pr": identifier, "verdict": match.group(1)},
            )
            for match in _VERDICT.finditer(record)
        )
        facts.extend(
            (
                offset + match.start(),
                {"type": "pr_head", "pr": identifier, "sha": match.group(1)},
            )
            for match in _HEAD.finditer(record)
        )
        return facts
    facts = [
        (
            offset + match.start(),
            {
                "type": "child_status",
                "handle": identifier,
                "status": match.group(1),
            },
        )
        for match in _STATUS.finditer(record)
    ]
    facts.extend(
        (
            offset + match.start(),
            {"type": "child_head", "handle": identifier, "sha": match.group(1)},
        )
        for match in _HEAD.finditer(record)
    )
    return facts


def _text_facts(text: str) -> list[tuple[int, dict[str, object]]]:
    facts = [
        fact
        for offset, kind, identifier, record in _identified_records(text)
        for fact in _record_text_facts(offset, kind, identifier, record)
    ]
    return sorted(facts, key=lambda item: (item[0], _TYPE_ORDER[str(item[1]["type"])]))


def _message_facts(message: Message) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    notifications = message.metadata.get("notifications")
    if (
        message.role is MessageRole.SYSTEM
        and message.metadata.get("zeta_event") == "agent_notifications"
        and isinstance(notifications, list)
    ):
        for item in notifications:
            if isinstance(item, Mapping):
                facts.extend(_structured_facts(item))
    if message.role is MessageRole.ASSISTANT:
        for block in message.content:
            if isinstance(block, TextContent):
                facts.extend(fact for _, fact in _text_facts(block.text))
    return facts


def extract_fact_occurrences(
    records: Sequence[tuple[int, Message]],
) -> tuple[Occurrence, ...]:
    occurrences: list[Occurrence] = []
    for seq, message in records:
        if seq <= 0 or message.metadata.get("fact_floor"):
            continue
        facts = _message_facts(message)
        occurrences.extend((fact, seq, ordinal) for ordinal, fact in enumerate(facts))
    return tuple(occurrences)


def _prior_aggregates(previous: object) -> list[Aggregate]:
    if previous is None:
        return []
    if not validate_fact_floor(previous):
        raise ValueError("invalid eviction fact floor")
    assert isinstance(previous, dict)
    aggregates: dict[tuple[object, ...], Aggregate] = {}
    for item in [*previous["oldest"], *previous["newest"]]:
        aggregates[_fact_key(item["fact"])] = dict(item)
    return list(aggregates.values())


def select_fact_floor(previous: object, occurrences: Sequence[Occurrence]) -> FactFloor:
    aggregates = {_fact_key(item["fact"]): item for item in _prior_aggregates(previous)}
    for fact, seq, ordinal in occurrences:
        if (
            not _valid_fact(fact)
            or type(seq) is not int
            or seq <= 0
            or type(ordinal) is not int
            or ordinal < 0
        ):
            continue
        key = _fact_key(fact)
        current = aggregates.get(key)
        if current is None:
            aggregates[key] = {
                "fact": dict(fact),
                "first_seq": seq,
                "last_seq": seq,
                "value_rank": [seq, ordinal],
            }
            continue
        current["first_seq"] = min(int(current["first_seq"]), seq)
        current["last_seq"] = max(int(current["last_seq"]), seq)
        if (seq, ordinal) > tuple(current["value_rank"]):
            current["fact"] = dict(fact)
            current["value_rank"] = [seq, ordinal]
    values = list(aggregates.values())
    oldest = sorted(values, key=lambda item: _selection_key(item, "first_seq"))[
        :_OLDEST_LIMIT
    ]
    oldest_keys = {_fact_key(item["fact"]) for item in oldest}
    newest = sorted(
        values, key=lambda item: _selection_key(item, "last_seq"), reverse=True
    )[:_NEWEST_LIMIT]
    # A newest-only key can never later enter the oldest eight. Normalizing its
    # discarded first rank keeps repeated replacement byte-identical to one shot.
    newest = [
        item
        if _fact_key(item["fact"]) in oldest_keys
        else {**item, "first_seq": item["last_seq"]}
        for item in newest
    ]
    floor: FactFloor = {"version": 1, "oldest": oldest, "newest": newest}
    assert validate_fact_floor(floor)
    return floor


def render_fact_floor(value: object) -> Message | None:
    if value is None:
        return None
    if not validate_fact_floor(value):
        raise ValueError("invalid eviction fact floor")
    assert isinstance(value, dict)
    seen: set[tuple[object, ...]] = set()
    lines = ["[retained typed facts]"]
    for item in [*value["oldest"], *value["newest"]]:
        fact = item["fact"]
        key = _fact_key(fact)
        if key in seen:
            continue
        seen.add(key)
        kind = fact["type"]
        if kind == "pr":
            lines.append(f"pr #{fact['pr']}")
        elif kind == "pr_verdict":
            lines.append(f"pr #{fact['pr']} verdict={fact['verdict']}")
        elif kind == "pr_head":
            lines.append(f"pr #{fact['pr']} head={fact['sha']}")
        elif kind == "child_status":
            lines.append(f"child {fact['handle']} status={fact['status']}")
        else:
            lines.append(f"child {fact['handle']} head={fact['sha']}")
    if len(lines) == 1:
        return None
    text = "\n".join(lines)
    assert len(text) <= FACT_FLOOR_CHAR_CAP
    return Message(
        MessageRole.SYSTEM, [TextContent(text)], metadata={"fact_floor": True}
    )


def _maximum_floor_message() -> Message:
    lines = ["[retained typed facts]"]
    lines.extend(
        f"child {'h' * 80} head={'a' * 40}"
        for _ in range(_OLDEST_LIMIT + _NEWEST_LIMIT)
    )
    text = "\n".join(lines)
    assert len(text) <= FACT_FLOOR_CHAR_CAP
    return Message(
        MessageRole.SYSTEM, [TextContent(text)], metadata={"fact_floor": True}
    )


FACT_FLOOR_TOKEN_CAP = message_token_count(_maximum_floor_message())


def reserve_target(target: int, reserve: int) -> int:
    return max(0, target - reserve)


class _Marker(Protocol):
    data: dict[str, Any]


class _EvictionAccounting(Protocol):
    items_evicted: int
    tokens_before: int
    reached_target: bool


@dataclass(frozen=True, slots=True)
class FactFloorPlan:
    floor: FactFloor | None
    message: Message | None
    reserve: int

    def target(self, target: int) -> int:
        return reserve_target(target, self.reserve)

    def telemetry(
        self,
        result: _EvictionAccounting,
        tokens_after: int,
        token_counter: Callable[[Message], int],
    ) -> dict[str, object]:
        telemetry: dict[str, object] = {
            "kind": "evict",
            "eviction_count": 1,
            "items_evicted": result.items_evicted,
            "tokens_before": result.tokens_before,
            "tokens_after": tokens_after,
        }
        if self.message is not None:
            telemetry.update(
                {
                    "reached_target": result.reached_target,
                    "fact_floor_chars": len(self.message.content[0].text),
                    "fact_floor_tokens": token_counter(self.message),
                }
            )
        return telemetry

    def proposed_messages(
        self,
        system_messages: Sequence[Message],
        view_messages: Sequence[Message],
        latest_user: Message | None,
    ) -> list[Message]:
        return [
            *system_messages,
            *view_messages,
            *([] if self.message is None else [self.message]),
            *([] if latest_user is None else [latest_user]),
        ]


def previous_fact_floor(
    markers: Sequence[_Marker], invalid_markers: Sequence[_Marker]
) -> object:
    return next(
        (
            marker.data.get("fact_floor")
            for marker in reversed(markers)
            if marker not in invalid_markers
        ),
        None,
    )


def floor_rebuild_records(
    markers: Sequence[_Marker], source_messages: Mapping[int, Message]
) -> list[tuple[int, Message]]:
    return [
        (seq, message)
        for seq, message in sorted(source_messages.items())
        if any(
            marker.data["source_seq_start"] <= seq <= marker.data["source_seq_end"]
            for marker in markers
        )
    ]


def prepare_records(
    records: Iterable[tuple[int, Message]],
    previous: object,
    *,
    floor_rebuild_records: Iterable[tuple[int, Message]] = (),
) -> tuple[list[tuple[int, Message]], FactFloorPlan]:
    """Materialize records and extract their facts in the same worker-side walk."""
    materialized: list[tuple[int, Message]] = []
    occurrences: list[Occurrence] = []
    for record in records:
        materialized.append(record)
        occurrences.extend(extract_fact_occurrences((record,)))
    for record in floor_rebuild_records:
        occurrences.extend(extract_fact_occurrences((record,)))
    floor = select_fact_floor(previous, occurrences)
    message = render_fact_floor(floor)
    return materialized, FactFloorPlan(
        floor if message is not None else None,
        message,
        FACT_FLOOR_TOKEN_CAP if occurrences or render_fact_floor(previous) else 0,
    )


def projected_fact_floor(value: object) -> list[Message]:
    if value is not None and not validate_fact_floor(value):
        return []
    message = render_fact_floor(value)
    return [] if message is None else [message]
