from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.support.branch_reads import resident_entries
from zeta.core.session_files import session_directory
from zeta.core.store import ConversationStore, TranscriptLoadError
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.storage_format import WriterLease, open_writer


def _recovery():
    import zeta.core.transcript_recovery as recovery

    return recovery


def _message(text: str) -> Message:
    return with_message_origin(
        Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
    )


def _diagnosed_store(
    tmp_path: Path, session_id: str = "damaged"
) -> tuple[Path, bytes, TranscriptLoadError]:
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id=session_id)
    store.append_message(_message("retained"))
    store.close()
    transcript = root / session_id / "conversation.jsonl"
    damaged = transcript.read_bytes() + b'{"seq":"invalid"}\n'
    transcript.write_bytes(damaged)
    with pytest.raises(TranscriptLoadError) as raised:
        refused = ConversationStore(root, session_id=session_id, _read_only=True)
        refused.close()
    assert raised.value.kind == "invalid_row"
    assert getattr(raised.value, "_source_facts", None) is not None
    return root, damaged, raised.value


@pytest.fixture
def home_lease(tmp_path: Path):
    """Retain the home lease before a test holds a session scope.

    A store opened inside that scope reuses it instead of taking the format rank.
    """
    lease = open_writer(tmp_path)
    try:
        yield lease
    finally:
        lease.close()


def _strict_reopen(root: Path, session_id: str, lease: WriterLease) -> None:
    reopened = ConversationStore(
        root, session_id=session_id, _read_only=True, _storage_lease=lease
    )
    reopened.close()


def test_semantic_diagnosis_retains_only_rows_before_first_bad_link(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="links")
    store.append_message(_message("retained"))
    store.append_message(_message("discarded"))
    store.close()
    transcript = root / "links" / "conversation.jsonl"
    rows = transcript.read_text(encoding="utf-8").splitlines()
    damaged = json.loads(rows[2])
    damaged["parent_id"] = "missing"
    rows[2] = json.dumps(damaged, separators=(",", ":"), sort_keys=True)
    transcript.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(TranscriptLoadError) as raised:
        refused = ConversationStore(root, session_id="links", _read_only=True)
        refused.close()
    facts = getattr(raised.value, "_source_facts", None)
    assert facts is not None
    assert raised.value.kind == "broken_links"
    assert facts.first_bad_row == 3
    assert facts.prefix.last_sequence == 1
    assert facts.prefix.last_entry_id == json.loads(rows[1])["id"]


def test_transaction_preserves_complete_evidence_and_strictly_reopens(
    tmp_path: Path,
    home_lease: WriterLease,
) -> None:
    root, damaged, diagnosis = _diagnosed_store(tmp_path)
    tail = damaged[diagnosis._source_facts.prefix.end_offset :]
    recovery = _recovery()
    target = recovery.RepairTarget("damaged")
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            target,
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
            memory_review_required=True,
            tail_evidence=True,
        )
        payload_before = (
            root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.jsonl"
        ).read_bytes()
        assert payload_before == tail
        manifest = recovery.read_evidence_manifest(directory_fd, result.quarantine_id)
        record = recovery.read_recovery_record(directory_fd, result.quarantine_id)
        assert record is not None and record.state == "committed"
        assert record.memory_step == "review_pending"
        status_before = recovery.recovery_status(directory_fd, target)
        evidence_before = (
            root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.json"
        ).read_bytes()
        assert recovery.reconcile_pending(
            directory_fd,
            target,
            strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
        ) == (record,)
        status_after = recovery.recovery_status(directory_fd, target)

    repaired = ConversationStore(root, session_id="damaged", _read_only=True)
    try:
        warning = resident_entries(repaired)[-1]
        assert warning.type == "warning"
        assert warning.data["kind"] == "transcript_recovery"
        assert warning.data["quarantine_id"] == result.quarantine_id
    finally:
        repaired.close()
    assert manifest.payload_digest == hashlib.sha256(tail).hexdigest()
    assert manifest.source_digest == hashlib.sha256(damaged).hexdigest()
    assert status_before == status_after
    assert status_after.quarantine_count == 1
    assert status_after.quarantine_bytes == len(tail)
    assert status_after.abandoned_evidence_count == 0
    assert (
        root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.json"
    ).read_bytes() == evidence_before
    assert (
        root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.jsonl"
    ).read_bytes() == payload_before


def test_repair_retries_short_warning_write_and_strictly_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_lease: WriterLease
) -> None:
    root, _, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    real_write = recovery.os.write
    shortened = False

    def short_warning_write(fd: int, data: bytes | memoryview) -> int:
        nonlocal shortened
        raw = bytes(data)
        if not shortened and b'"kind":"transcript_recovery"' in raw:
            shortened = True
            return real_write(fd, raw[:2])
        return real_write(fd, data)

    monkeypatch.setattr(recovery.os, "write", short_warning_write)
    with session_directory(root, "damaged") as (_, directory_fd):
        recovery.repair_transcript(
            directory_fd,
            recovery.RepairTarget("damaged"),
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
            memory_review_required=False,
            tail_evidence=True,
        )
    assert shortened
    _strict_reopen(root, "damaged", home_lease)


@pytest.mark.parametrize(
    ("step", "active_replaced", "evidence_published"),
    [
        ("quarantine_synced", False, False),
        ("staged_active_synced", False, False),
        ("evidence_published", False, True),
        ("state_pending", False, True),
        ("active_replaced", True, True),
        ("strict_reopened", True, True),
        ("state_committed", True, True),
    ],
)
def test_fault_boundaries_keep_original_or_leave_reconcilable_state(
    tmp_path: Path,
    home_lease: WriterLease,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    active_replaced: bool,
    evidence_published: bool,
) -> None:
    root, damaged, diagnosis = _diagnosed_store(tmp_path, session_id=step)
    recovery = _recovery()
    target = recovery.RepairTarget(step)

    def stop_at(current: str) -> None:
        if current == step:
            raise OSError(f"fault at {step}")

    # test-contract: seam inject each durable transaction boundary
    monkeypatch.setattr(recovery, "_after_recovery_step", stop_at)
    with (
        session_directory(root, step) as (_, directory_fd),
        pytest.raises(OSError, match=f"fault at {step}"),
    ):
        recovery.repair_transcript(
            directory_fd,
            target,
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, step, home_lease),
            memory_review_required=False,
            tail_evidence=True,
        )
    # A writable reopen takes session execution, so it runs outside the shape scope.
    active = (root / step / "conversation.jsonl").read_bytes()
    if active_replaced:
        assert active != damaged
        reopened = ConversationStore(root, session_id=step)
        reopened.close()
    else:
        assert active == damaged
        with pytest.raises(TranscriptLoadError):
            reopened = ConversationStore(root, session_id=step)
            reopened.close()
    with session_directory(root, step) as (_, directory_fd):
        evidence_files = list((root / step).glob("conversation.quarantine.*.json"))
        payload_files = list((root / step).glob("conversation.quarantine.*.jsonl"))
        assert bool(evidence_files) is evidence_published
        assert bool(payload_files) is evidence_published
        if evidence_published:
            assert len(evidence_files) == len(payload_files) == 1
            quarantine_id = (
                evidence_files[0]
                .name.removeprefix("conversation.quarantine.")
                .removesuffix(".json")
            )
            tail = damaged[diagnosis._source_facts.prefix.end_offset :]
            assert payload_files[0].read_bytes() == tail
            record = recovery.read_recovery_record(directory_fd, quarantine_id)
            if record is None:
                record = recovery.reconcile_pending(
                    directory_fd,
                    target,
                    strict_reopen=lambda: _strict_reopen(root, step, home_lease),
                )[0]
            expected = "committed" if active_replaced else "abandoned"
            assert record.state == expected


@pytest.mark.parametrize("change", ["digest", "identity"])
def test_source_identity_change_refuses_before_creating_evidence(
    tmp_path: Path, change: str, home_lease: WriterLease
) -> None:
    root, damaged, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    transcript = root / "damaged" / "conversation.jsonl"
    changed = damaged[:-4] + b"null" if change == "digest" else damaged
    assert len(changed) == len(damaged)
    if change == "digest":
        transcript.write_bytes(changed)
    else:
        replacement = transcript.with_suffix(".replacement")
        replacement.write_bytes(changed)
        replacement.replace(transcript)

    with session_directory(root, "damaged") as (_, directory_fd):
        with pytest.raises(recovery.TranscriptSourceChanged):
            recovery.repair_transcript(
                directory_fd,
                recovery.RepairTarget("damaged"),
                diagnosis._source_facts,
                strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
                memory_review_required=False,
                tail_evidence=True,
            )
        assert (
            recovery.recovery_status(
                directory_fd, recovery.RepairTarget("damaged")
            ).quarantine_count
            == 0
        )
    assert transcript.read_bytes() == changed


@pytest.mark.parametrize(
    ("state", "memory_step"),
    [
        ("pending", "resumed"),
        ("abandoned", "review_pending"),
        ("abandoned", "resumed"),
    ],
)
def test_record_parser_rejects_illegal_state_memory_pairs(
    tmp_path: Path, state: str, memory_step: str, home_lease: WriterLease
) -> None:
    root, _, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            recovery.RepairTarget("damaged"),
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
            memory_review_required=False,
            tail_evidence=True,
        )
        value = recovery.read_recovery_record(
            directory_fd, result.quarantine_id
        ).to_dict()
    value.update(state=state, memory_step=memory_step)
    with pytest.raises(
        recovery.TranscriptRecoveryIntegrityError,
        match="state and memory step",
    ):
        recovery.parse_recovery_record(value)


def test_record_parser_rejects_unknown_fields(
    tmp_path: Path, home_lease: WriterLease
) -> None:
    root, _, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    target = recovery.RepairTarget("damaged")
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            target,
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged", home_lease),
            memory_review_required=False,
            tail_evidence=True,
        )
        value = recovery.read_evidence_manifest(
            directory_fd, result.quarantine_id
        ).to_dict()
    value["unexpected"] = True
    with pytest.raises(recovery.TranscriptRecoveryIntegrityError, match="shape"):
        recovery.parse_evidence_manifest(value)


def _torn_tail_repair(root: Path, session_id: str) -> bytes:
    """Append a torn final row, repair it through a writable open, return the original."""
    transcript = root / session_id / "conversation.jsonl"
    with transcript.open("ab") as handle:
        handle.write(b'{"seq": 999, "id": "torn"')
    damaged = transcript.read_bytes()
    with pytest.warns(RuntimeWarning, match="torn"):
        reopened = ConversationStore(root, session_id=session_id)
    reopened.close()
    return damaged


def _quarantine_ids(session_dir: Path) -> set[str]:
    return {
        path.name.removeprefix("conversation.quarantine.").removesuffix(".json")
        for path in session_dir.glob("conversation.quarantine.*.json")
    }


def test_torn_tail_quarantine_holds_only_the_dropped_tail(
    tmp_path: Path, home_lease: WriterLease
) -> None:
    # home_lease creates the home at WRITER_EPOCH; the automatic torn-tail repair runs.
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="large")
    for index in range(16):
        store.append_message(_message(f"{index}:" + "x" * 64 * 1024))
    store.close()
    transcript = root / "large" / "conversation.jsonl"
    retained = transcript.read_bytes()
    assert len(retained) > 1024 * 1024

    damaged = _torn_tail_repair(root, "large")
    tail = damaged[len(retained) :]

    recovery = _recovery()
    (quarantine_id,) = _quarantine_ids(root / "large")
    payload = (root / "large" / f"conversation.quarantine.{quarantine_id}.jsonl").read_bytes()
    assert payload == tail, (
        f"quarantine kept {len(payload)} bytes; the dropped tail is {len(tail)} bytes"
    )
    with session_directory(root, "large") as (_, directory_fd):
        status = recovery.recovery_status(directory_fd, recovery.RepairTarget("large"))
        manifest = recovery.read_evidence_manifest(directory_fd, quarantine_id)
    assert status.quarantine_bytes == len(tail)
    # The original stays provable from the kept active prefix plus the tail.
    active = transcript.read_bytes()
    assert (
        hashlib.sha256(active[: manifest.valid_prefix_end] + payload).hexdigest()
        == manifest.source_digest
        == hashlib.sha256(damaged).hexdigest()
    )


def test_home_below_tail_evidence_epoch_keeps_full_copy(tmp_path: Path) -> None:
    from zeta.storage_format import FEATURE_EPOCHS, StorageFeature, advance

    advance(
        tmp_path,
        expected_epoch=0,
        target_epoch=FEATURE_EPOCHS[StorageFeature.TRANSCRIPT_TAIL_EVIDENCE] - 1,
    )
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="old-home")
    store.append_message(_message("kept"))
    store.close()

    damaged = _torn_tail_repair(root, "old-home")

    (quarantine_id,) = _quarantine_ids(root / "old-home")
    session_dir = root / "old-home"
    payload = (session_dir / f"conversation.quarantine.{quarantine_id}.jsonl").read_bytes()
    manifest = json.loads(
        (session_dir / f"conversation.quarantine.{quarantine_id}.json").read_text()
    )
    assert payload == damaged
    assert manifest["schema"] == "zeta.transcript-quarantine.v1"


class _PruneCrash(BaseException):
    """Not swallowed by prune, so the crash point is the on-disk state."""


def _sequenced_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import zeta.core.store._log as log_module

    ticks = iter(range(1, 10_000))
    monkeypatch.setattr(
        log_module, "_now", lambda: f"2026-01-01T00:00:00.{next(ticks):06d}+00:00"
    )


def _bad_row_repair(
    root: Path,
    session_id: str,
    lease: WriterLease,
    *,
    memory_review_required: bool = False,
) -> str:
    """Append a newline-terminated bad row, diagnose it, and repair it directly."""
    transcript = root / session_id / "conversation.jsonl"
    transcript.write_bytes(transcript.read_bytes() + b'{"seq":"invalid"}\n')
    with pytest.raises(TranscriptLoadError) as raised:
        refused = ConversationStore(root, session_id=session_id, _read_only=True)
        refused.close()
    recovery = _recovery()
    with session_directory(root, session_id) as (_, directory_fd):
        return recovery.repair_transcript(
            directory_fd,
            recovery.RepairTarget(session_id),
            raised.value._source_facts,
            strict_reopen=lambda: _strict_reopen(root, session_id, lease),
            memory_review_required=memory_review_required,
            tail_evidence=True,
        ).quarantine_id


def test_prune_keeps_unresolved_and_newest_resolved_sets(
    tmp_path: Path, home_lease: WriterLease, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sequenced_clock(monkeypatch)
    recovery = _recovery()
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="pruned")
    store.append_message(_message("kept"))
    store.close()
    session_dir = root / "pruned"

    review = _bad_row_repair(root, "pruned", home_lease, memory_review_required=True)

    def abandon_at_pending(step: str) -> None:
        if step == "state_pending":
            raise _PruneCrash

    # test-contract: seam abandon one repair before its active replace
    monkeypatch.setattr(recovery, "_after_recovery_step", abandon_at_pending)
    with pytest.raises(_PruneCrash):
        _bad_row_repair(root, "pruned", home_lease)
    monkeypatch.setattr(recovery, "_after_recovery_step", lambda _step: None)
    (abandoned,) = _quarantine_ids(session_dir) - {review}
    with session_directory(root, "pruned") as (_, directory_fd):
        abandoned_record = recovery.read_recovery_record(directory_fd, abandoned)
    assert abandoned_record is not None and abandoned_record.state == "abandoned"

    committed = [_bad_row_repair(root, "pruned", home_lease) for _ in range(4)]
    orphan = session_dir / f"conversation.quarantine.{'0' * 32}.jsonl"
    orphan.write_bytes(b"orphan")
    committed.append(_bad_row_repair(root, "pruned", home_lease))

    assert _quarantine_ids(session_dir) == {review, *committed[-3:]}
    for removed in (abandoned, *committed[:2]):
        assert not list(session_dir.glob(f"conversation.*.{removed}.*"))
    assert not orphan.exists()
    with session_directory(root, "pruned") as (_, directory_fd):
        status = recovery.recovery_status(directory_fd, recovery.RepairTarget("pruned"))
        kept_review = recovery.read_recovery_record(directory_fd, review)
    assert status.latest_committed is not None
    assert status.latest_committed.quarantine_id == committed[-1]
    assert kept_review is not None and kept_review.memory_step == "review_pending"


@pytest.mark.parametrize(
    "step", ["prune_state_removed", "prune_manifest_removed", "prune_payload_removed"]
)
def test_prune_crash_points_reopen_and_converge_on_next_repair(
    tmp_path: Path,
    home_lease: WriterLease,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
) -> None:
    _sequenced_clock(monkeypatch)
    recovery = _recovery()
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id=step)
    store.append_message(_message("kept"))
    store.close()
    session_dir = root / step
    first = [_bad_row_repair(root, step, home_lease) for _ in range(3)]

    def crash_at(current: str) -> None:
        if current == step:
            raise _PruneCrash

    # test-contract: seam inject each durable prune boundary
    monkeypatch.setattr(recovery, "_after_recovery_step", crash_at)
    with pytest.raises(_PruneCrash):
        _bad_row_repair(root, step, home_lease)
    monkeypatch.setattr(recovery, "_after_recovery_step", lambda _step: None)

    reopened = ConversationStore(root, session_id=step)
    reopened.close()
    with session_directory(root, step) as (_, directory_fd):
        recovery.recovery_status(directory_fd, recovery.RepairTarget(step))

    last = _bad_row_repair(root, step, home_lease)
    kept = _quarantine_ids(session_dir)
    assert len(kept) == 3 and last in kept and first[0] not in kept
    assert {
        path.name.removeprefix("conversation.quarantine.").removesuffix(".jsonl")
        for path in session_dir.glob("conversation.quarantine.*.jsonl")
    } == kept
    assert {
        path.name.removeprefix("conversation.recovery.").removesuffix(".json")
        for path in session_dir.glob("conversation.recovery.*.json")
    } == kept


def test_recovery_status_skips_a_set_pruned_between_listing_and_read(
    tmp_path: Path, home_lease: WriterLease, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery = _recovery()
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="raced")
    store.append_message(_message("kept"))
    store.close()
    session_dir = root / "raced"
    _torn_tail_repair(root, "raced")
    (pruned,) = _quarantine_ids(session_dir)
    _torn_tail_repair(root, "raced")
    (survivor,) = _quarantine_ids(session_dir) - {pruned}

    real_read_record = recovery._read_record
    raced = False

    def prune_before_read(directory_fd: int, name: str):
        nonlocal raced
        if not raced and name == f"conversation.recovery.{pruned}.json":
            raced = True
            for suffix in ("recovery.{}.json", "quarantine.{}.json", "quarantine.{}.jsonl"):
                (session_dir / f"conversation.{suffix.format(pruned)}").unlink()
        return real_read_record(directory_fd, name)

    # test-contract: seam a concurrent prune lands between a reader's listing and read
    monkeypatch.setattr(recovery, "_read_record", prune_before_read)
    with session_directory(root, "raced") as (_, directory_fd):
        status = recovery.recovery_status(directory_fd, recovery.RepairTarget("raced"))
    assert raced
    assert status.quarantine_count == 1
    assert status.latest_committed is not None
    assert status.latest_committed.quarantine_id == survivor
