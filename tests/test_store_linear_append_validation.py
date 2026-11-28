"""A linear append and a linear reload validate only the new rows.

The incremental path must give the same rows, byte offsets, file bytes, and
errors as a forced complete validation; non-linear history keeps the full path.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from zeta.core import checkpoints
from zeta.core.checkpoints import ConversationIntegrityError
from zeta.core.checkpoints.mutation import AppendConversationRows, ConversationRow
from zeta.core.store import ConversationStore
from zeta.core.store import _log as store_log
from zeta.core.store._rows import ResidentRows
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)


def _user(text: str) -> Message:
    return with_message_origin(
        Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
    )


def _assistant(text: str) -> Message:
    return Message(MessageRole.ASSISTANT, [TextContent(text)])


def _force_full_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every append and reload take the complete validation path."""

    def full_tail(store: ConversationStore, entries: list[object]) -> None:
        length = len(store._resident_rows.rows)
        store._resident_rows.extend(entries)
        try:
            store._validate_entries()
        finally:
            store._resident_rows.truncate_to(length)

    # raising=False: the unchanged base has no incremental tail check to replace
    # test-contract: seam the reference arm replaces the incremental tail check with the full validator
    monkeypatch.setattr(
        ConversationStore, "_validate_linear_tail", full_tail, raising=False
    )
    # test-contract: seam the reference arm forces every reload onto the full load
    monkeypatch.setattr(
        ConversationStore, "_accept_incremental_entries", lambda *_args: False
    )


def _deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = itertools.count()
    monkeypatch.setattr(os, "urandom", lambda size: next(counter).to_bytes(size, "big"))
    for module in (store_log, checkpoints):
        # test-contract: seam fixed clocks so both arms write byte-identical files
        monkeypatch.setattr(module, "_now", lambda: "2026-01-01T00:00:00Z")


def _spans(store: ConversationStore) -> list[tuple[object, ...]]:
    """Every resident row with its absolute byte span and the bytes it reads."""
    rows = store._resident_rows
    data = store.path.read_bytes()
    header = data.find(b"\n") + 1 if rows.relative_offsets else 0
    out = []
    for ref in rows.rows:
        start = ref.offset + header
        raw = data[start : start + ref.length]
        assert raw.endswith(b"\n"), (ref.seq, raw[:60])
        out.append(
            (
                ref.seq,
                ref.id,
                ref.parent_id,
                ref.type,
                start,
                ref.length,
                raw,
                store.entry(ref).to_dict(),
            )
        )
    return out


def _state(store: ConversationStore) -> tuple[object, ...]:
    return (
        _spans(store),
        [ref.id for ref in store.branch_refs()],
        store.path.read_bytes().split(b"\n", 1)[1],
    )


def _error(action: Callable[[], object]) -> str | None:
    try:
        action()
    except ConversationIntegrityError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _script(root: Path) -> list[object]:
    """Linear appends, an external linear tail, a fork, truncation, corruption."""
    trace: list[object] = []
    owner = ConversationStore(root, session_id="differential")
    external = ConversationStore(root, session_id="differential")
    try:
        owner.append_message(_user("one"))
        owner.append_many(
            [
                ("message", {"message": _assistant("two").to_dict()}),
                ("message", {"message": _user("three").to_dict()}),
            ]
        )
        trace.append(("linear", _state(owner)))

        external.append_message(_assistant("external"))
        external.append_message(_user("external two"))
        reloads = owner._full_reload_count
        owner.refresh()
        trace.append(
            ("external tail", _state(owner), owner._full_reload_count - reloads)
        )
        owner.append_message(_assistant("after reload"))
        trace.append(("append after reload", _state(owner)))

        checkpoint = owner.append_checkpoint("cp")
        owner.append_message(_assistant("trailing"))
        owner.append_fork(str(checkpoint.seq))
        owner.append_message(_user("on fork"))
        external.refresh()
        trace.append(("fork", _state(owner), _state(external)))

        data = owner.path.read_bytes()
        owner.path.write_bytes(data[: data.rstrip(b"\n").rfind(b"\n") + 1])
        trace.append(("truncated", _error(owner.refresh), _state(owner)))

        with owner.path.open("ab") as handle:
            handle.write(b'{"seq": 99, "id": "corrupt"}\n')
        trace.append(("corrupt", _error(owner.refresh)))
        trace.append(
            (
                "corrupt reopen",
                _error(lambda: ConversationStore(root, session_id="differential")),
            )
        )
    finally:
        owner.close()
        external.close()
    return trace


def test_incremental_paths_match_forced_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        _deterministic(patched)
        incremental = _script(tmp_path / "incremental")
    with monkeypatch.context() as patched:
        _deterministic(patched)
        _force_full_validation(patched)
        full = _script(tmp_path / "full")

    assert incremental[1][2] == 0, "the external tail must load incrementally"
    assert full[1][2] == 1, "the forced arm must take the full reload"
    assert [
        step[:2] if step[0] == "external tail" else step for step in incremental
    ] == [step[:2] if step[0] == "external tail" else step for step in full]
    assert incremental[-2][1] is not None and incremental[-1][1] is not None


def _approval(store: ConversationStore) -> str:
    call = ToolCall("call-1", "bash", {"cmd": "pwd"})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    return call.id


def _rule_cases() -> Iterator[
    tuple[str, Callable[[ConversationStore], list[tuple[str, dict]]], str]
]:
    def notification(
        store: ConversationStore, ack: bool = False, shown: bool = False
    ) -> str:
        entry = store.append_agent_notification(
            "child-a",
            child_session_path="/c/a",
            description="a",
            status="completed",
            text="done",
        )
        store.record_agent_notification_delivery(
            entry.id, presented=shown, acknowledged=ack
        )
        return entry.id

    yield (
        "ack unlinked",
        lambda s: [("notification_ack", {"notification_id": "nope"})],
        "notification acknowledgement is not linked: nope",
    )
    yield (
        "ack duplicate",
        lambda s: [
            ("notification_ack", {"notification_id": notification(s, ack=True)})
        ],
        "duplicate notification acknowledgement: {id}",
    )
    yield (
        "presented unlinked",
        lambda s: [("notification_tui_presented", {"notification_id": "nope"})],
        "notification TUI presentation is not linked: nope",
    )
    yield (
        "presented duplicate",
        lambda s: [
            (
                "notification_tui_presented",
                {"notification_id": notification(s, shown=True)},
            )
        ],
        "duplicate notification TUI presentation: {id}",
    )
    yield (
        "prompt ack unlinked",
        lambda s: [("pending_prompt_ack", {"prompt_id": "nope"})],
        "pending prompt acknowledgement is not linked: nope",
    )

    def acked_prompt(store: ConversationStore) -> str:
        prompt = store.append_pending_prompt("follow up")
        store.acknowledge_pending_prompt(prompt.id)
        return prompt.id

    yield (
        "prompt ack duplicate",
        lambda s: [("pending_prompt_ack", {"prompt_id": acked_prompt(s)})],
        "duplicate pending prompt acknowledgement: {id}",
    )
    yield (
        "resolution unlinked",
        lambda s: [
            ("approval_resolution", {"request_id": "nope", "decision": "allow"})
        ],
        "approval resolution is not linked to request: nope",
    )

    def resolved(store: ConversationStore) -> str:
        request_id = _approval(store)
        store.append_approval_resolution(request_id, "allow")
        return request_id

    yield (
        "resolution duplicate",
        lambda s: [
            ("approval_resolution", {"request_id": resolved(s), "decision": "allow"})
        ],
        "duplicate approval resolution: {id}",
    )

    def duplicate_request(store: ConversationStore) -> list[tuple[str, dict]]:
        _approval(store)
        prior = store.entry(store.branch_refs()[-1])
        return [("message", dict(prior.data))]

    yield "request duplicate", duplicate_request, "duplicate approval request: call-1"
    yield (
        "batch ack unlinked after notification",
        lambda s: [
            ("message", {"message": _assistant("ok").to_dict()}),
            ("notification_ack", {"notification_id": "nope"}),
        ],
        "notification acknowledgement is not linked: nope",
    )


@pytest.mark.parametrize(
    ("rows_for", "expected"),
    [case[1:] for case in _rule_cases()],
    ids=[case[0] for case in _rule_cases()],
)
def test_linear_append_raises_the_full_validator_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows_for: Callable[[ConversationStore], list[tuple[str, dict]]],
    expected: str,
) -> None:
    def run(root: Path) -> tuple[str | None, bytes, bytes]:
        store = ConversationStore(root, session_id="rules")
        try:
            store.append_message(_user("start"))
            rows = rows_for(store)
            before = store.path.read_bytes()
            error = _error(lambda: store.append_many(rows))
            return error, before, store.path.read_bytes()
        finally:
            store.close()

    with monkeypatch.context() as patched:
        _deterministic(patched)
        incremental = run(tmp_path / "incremental")
    with monkeypatch.context() as patched:
        _deterministic(patched)
        _force_full_validation(patched)
        full = run(tmp_path / "full")

    assert incremental == full
    error, before, after = incremental
    assert error is not None
    prefix = f"ConversationIntegrityError: {expected.removesuffix('{id}')}"
    assert error.startswith(prefix) if "{id}" in expected else error == prefix
    assert before == after, "a rejected append must not write bytes"


def _count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    count = [0]
    original = ResidentRows._decode

    def counted(self: ResidentRows, fd: int, ref: object) -> object:
        count[0] += 1
        return original(self, fd, ref)

    # test-contract: complexity row decodes are the O(rows) cost this pins
    monkeypatch.setattr(ResidentRows, "_decode", counted)
    return count


@pytest.mark.parametrize("rows", [50, 500])
def test_linear_append_and_reload_decode_no_prior_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    owner = ConversationStore(tmp_path, session_id="decodes")
    external = ConversationStore(tmp_path, session_id="decodes")
    try:
        owner.append_many(
            [
                ("message", {"message": _assistant(f"row {i}").to_dict()})
                for i in range(rows)
            ]
        )
        external.refresh()
        decodes = _count_decodes(monkeypatch)

        owner.append_message(_assistant("linear"))
        external.append_message(_assistant("external"))
        reloads = owner._full_reload_count
        owner.refresh()

        assert owner._full_reload_count == reloads
        assert decodes[0] == 0
        # every row reads back byte-exact from its recorded offset
        fresh = ConversationStore(tmp_path, session_id="decodes")
        try:
            assert _spans(owner) == _spans(fresh)
        finally:
            fresh.close()
    finally:
        owner.close()
        external.close()


def test_non_linear_append_keeps_the_full_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="branch")
    try:
        first = store.append_message(_user("one"))
        store.append_message(_assistant("two"))
        calls = [0]
        original = ConversationStore._validate_entries

        def spy(self: ConversationStore, *args: object, **kwargs: object) -> None:
            calls[0] += 1
            original(self, *args, **kwargs)

        # test-contract: complexity the full validator must not run on a linear append
        monkeypatch.setattr(ConversationStore, "_validate_entries", spy)
        store.append_message(_assistant("linear"))
        assert calls[0] == 0

        row = ConversationRow(
            "message", {"message": _assistant("branch").to_dict()}, first.id
        )
        store.commit(AppendConversationRows((row,)), store.conversation_generation)
        assert calls[0] >= 1
        assert store.branch_refs()[0].id == first.id
        assert len(store.branch_refs()) == 2
    finally:
        store.close()
