import copy
import json
from pathlib import Path

import pytest

from zeta.context_accounting import message_token_count
from zeta.context_eviction._fact_floor import (
    FACT_FLOOR_TOKEN_CAP,
    extract_fact_occurrences,
    render_fact_floor,
    select_fact_floor,
    validate_fact_floor,
)
from zeta.core.context import ContextAssembler
from zeta.core.slash import compaction_history
from zeta.core.store import ConversationEntry, ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)


def _assistant(value: str) -> Message:
    return Message(MessageRole.ASSISTANT, [TextContent(value)])


def _notification(**fields: object) -> Message:
    return Message(
        MessageRole.SYSTEM,
        [TextContent("durable notification")],
        metadata={"zeta_event": "agent_notifications", "notifications": [fields]},
    )


def _floor(
    records: list[tuple[int, Message]], previous: object = None
) -> dict[str, object]:
    return select_fact_floor(previous, extract_fact_occurrences(records))


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_typed_extraction_and_strict_sha_binding() -> None:
    records = [
        (1, _assistant("PR #12 verdict=MERGE-READY sha: abcdef1")),
        (2, _assistant("PR #13 abcdef2")),
        (3, _assistant("standalone deadbee and arbitrary verdict=unknown")),
        (4, _assistant("child root:7 status=failed head: 123abcd")),
        (
            5,
            _notification(
                child_instance_id="root:8",
                status="completed",
                verdict="pass",
                head_sha="456defa",
                pr=14,
            ),
        ),
    ]

    rendered = render_fact_floor(_floor(records))

    assert rendered is not None
    text = rendered.content[0].text
    assert "pr #12 verdict=MERGE-READY" in text
    assert "pr #12 head=abcdef1" in text
    assert "pr #13 head=" not in text
    assert "deadbee" not in text
    assert "unknown" not in text
    assert "child root:7 status=failed" in text
    assert "child root:7 head=123abcd" in text
    assert "pr #14 verdict=pass" in text
    assert "pr #14 head=456defa" in text
    assert "child root:8 status=completed" in text
    assert "child root:8 head=456defa" in text


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"version": 2, "oldest": [], "newest": []},
        {"version": 1, "oldest": "bad", "newest": []},
        {
            "version": 1,
            "oldest": [
                {
                    "fact": {
                        "type": "child_status",
                        "handle": "h",
                        "status": "running",
                    },
                    "first_seq": 1,
                    "last_seq": 1,
                    "value_rank": [1, 0],
                }
            ],
            "newest": [],
        },
        {
            "version": 1,
            "oldest": [
                {
                    "fact": {"type": "pr_head", "pr": 1, "sha": "DEADBEE"},
                    "first_seq": 1,
                    "last_seq": 1,
                    "value_rank": [1, 0],
                }
            ],
            "newest": [],
        },
    ],
)
def test_malformed_persisted_floor_is_rejected(value: object) -> None:
    assert not validate_fact_floor(value)


def test_floor_is_bounded_and_max_receipt_fits_production_token_cap() -> None:
    records = [
        (index, _assistant(f"child {'h' * 78}{index:02d} head: {'a' * 40}"))
        for index in range(1, 41)
    ]
    floor = _floor(records)
    rendered = render_fact_floor(floor)

    assert len(floor["oldest"]) == 8
    assert len(floor["newest"]) == 16
    assert rendered is not None
    assert len(rendered.content[0].text) <= 4096
    assert message_token_count(rendered) <= FACT_FLOOR_TOKEN_CAP


def _partitioned(
    records: list[tuple[int, Message]], cuts: list[int]
) -> dict[str, object]:
    floor: object = None
    start = 0
    for end in [*cuts, len(records)]:
        floor = _floor(records[start:end], floor)
        start = end
    assert isinstance(floor, dict)
    return floor


@pytest.mark.parametrize(
    ("records", "cuts"),
    [
        (
            [(i, _assistant(f"PR #{i}")) for i in range(1, 10)]
            + [(20, _assistant("PR #1 verdict=MERGE-READY"))],
            [9],
        ),
        (
            [(i, _assistant(f"PR #{i}")) for i in range(1, 30)]
            + [(40, _assistant("PR #9 verdict=pass"))],
            [29],
        ),
        (
            [
                (1, _assistant("PR #1 verdict=fail")),
                (2, _assistant("PR #1 verdict=pass")),
            ],
            [1],
        ),
        (
            [(1, _assistant("PR #1")), (1, _assistant("child root:1 status=failed"))],
            [1],
        ),
    ],
)
def test_repeated_selection_matches_one_shot_for_associativity_fixtures(
    records: list[tuple[int, Message]], cuts: list[int]
) -> None:
    one_shot = _floor(records)
    repeated = _partitioned(records, cuts)

    assert _encoded(repeated["oldest"]) == _encoded(one_shot["oldest"])
    assert _encoded(repeated["newest"]) == _encoded(one_shot["newest"])
    assert (
        render_fact_floor(repeated).to_dict() == render_fact_floor(one_shot).to_dict()
    )


def test_ten_replacements_equal_one_shot_and_stay_bounded() -> None:
    records = [
        (i, _assistant(f"PR #{i} verdict={'pass' if i % 2 else 'fail'}"))
        for i in range(1, 81)
    ]
    repeated = _partitioned(records, list(range(8, 80, 8)))
    one_shot = _floor(records)

    assert _encoded(repeated) == _encoded(one_shot)
    assert len(repeated["oldest"]) <= 8
    assert len(repeated["newest"]) <= 16


def test_old_marker_without_floor_projects_without_rewrite() -> None:
    message = _assistant("receipt")
    marker = ConversationEntry(
        seq=2,
        id="marker",
        parent_id=None,
        lane="main",
        type="compaction",
        data={
            "kind": "evict",
            "source_seq_start": 1,
            "source_seq_end": 1,
            "view": [{"seq": 1, "message": message.to_dict()}],
        },
    )
    before = copy.deepcopy(marker.data)

    items = ContextAssembler._marker_items(marker, {1, 2})

    assert [item.message.content[0].text for item in items] == ["receipt"]
    assert marker.data == before


def test_floor_projects_after_view_and_before_retained_rows() -> None:
    floor = _floor([(1, _assistant("PR #7 verdict=pass"))])
    marker = ConversationEntry(
        seq=2,
        id="marker",
        parent_id=None,
        lane="main",
        type="compaction",
        data={
            "kind": "evict",
            "source_seq_start": 1,
            "source_seq_end": 1,
            "view": [{"seq": 1, "message": _assistant("view receipt").to_dict()}],
            "fact_floor": floor,
        },
    )

    items = ContextAssembler._marker_items(marker, {1, 2})

    assert [item.message.content[0].text for item in items] == [
        "view receipt",
        "[retained typed facts]\npr #7\npr #7 verdict=pass",
    ]


@pytest.mark.asyncio
async def test_public_eviction_persists_floor_accounting_and_no_fact_compatibility(
    tmp_path: Path,
) -> None:
    async def compact(
        name: str, old_text: str
    ) -> tuple[dict[str, object], list[Message]]:
        store = ConversationStore(tmp_path / name)
        store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER
            )
        )
        store.append_message(_assistant(old_text))
        store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("recent")]), MessageOrigin.USER
            )
        )
        context = await ContextAssembler(
            store, token_budget=2_000, retained_tail=1
        ).assemble_context(force=True)
        marker = next(
            entry for entry in reversed(store.replay()) if entry.type == "compaction"
        )
        return marker.data, context.messages

    floor_marker, messages = await compact(
        "facts", "PR #7 verdict=pass sha: abcdef1\n" + "large old response " * 2_000
    )
    telemetry = floor_marker["telemetry"]
    floor_message = next(
        message for message in messages if message.metadata.get("fact_floor")
    )
    assert telemetry["reached_target"] is True
    assert telemetry["fact_floor_chars"] == len(floor_message.content[0].text)
    assert telemetry["fact_floor_tokens"] == message_token_count(floor_message)
    assert telemetry["fact_floor_tokens"] <= FACT_FLOOR_TOKEN_CAP
    assert messages.index(floor_message) < len(messages) - 1

    plain_marker, _ = await compact("plain", "large old response " * 2_000)
    assert "fact_floor" not in plain_marker
    assert set(plain_marker["telemetry"]) == {
        "kind",
        "eviction_count",
        "items_evicted",
        "tokens_before",
        "tokens_after",
    }


def test_compaction_surfaces_are_byte_identical_with_floor(tmp_path: Path) -> None:
    def snapshot(with_floor: bool) -> bytes:
        store = ConversationStore(tmp_path / ("floor" if with_floor else "plain"))
        source = store.append_message(_assistant("source"))
        data = _floor([(source.seq, _assistant("PR #7"))]) if with_floor else None
        store.append_compaction_marker(
            "summary",
            source.seq,
            source.seq,
            kind="evict",
            view=[{"seq": source.seq, "message": _assistant("receipt").to_dict()}],
            fact_floor=data,
            telemetry={"tokens_before": 10, "tokens_after": 5},
        )
        history = compaction_history(store.replay(), message_token_count)
        return repr(history).encode()

    # Both TUI and serve use this generic compaction-history render input.
    assert snapshot(False) == snapshot(True)
