from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zeta.core.session_files import (
    child_directory,
    read_session_file,
    session_directory,
    write_json_file,
)
from zeta.core.store import ConversationStore, TranscriptLoadError
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)


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


def _strict_reopen(root: Path, session_id: str) -> None:
    reopened = ConversationStore(root, session_id=session_id, _read_only=True)
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
) -> None:
    root, damaged, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    target = recovery.RepairTarget("damaged")
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            target,
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged"),
            memory_review_required=True,
        )
        payload_before = (
            root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.jsonl"
        ).read_bytes()
        assert payload_before == damaged
        manifest = recovery.read_evidence_manifest(directory_fd, result.quarantine_id)
        record = recovery.read_recovery_record(directory_fd, result.quarantine_id)
        status_before = recovery.recovery_status(directory_fd, target)
        evidence_before = (
            root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.json"
        ).read_bytes()
        assert recovery.reconcile_pending(
            directory_fd,
            target,
            strict_reopen=lambda: _strict_reopen(root, "damaged"),
        ) == (record,)
        status_after = recovery.recovery_status(directory_fd, target)

    repaired = ConversationStore(root, session_id="damaged", _read_only=True)
    try:
        warning = repaired.entries[-1]
        assert warning.type == "warning"
        assert warning.data["kind"] == "transcript_recovery"
        assert warning.data["quarantine_id"] == result.quarantine_id
    finally:
        repaired.close()
    assert manifest.payload_digest == hashlib.sha256(damaged).hexdigest()
    assert record is not None and record.state == "committed"
    assert record.memory_step == "review_pending"
    assert status_before == status_after
    assert status_after.quarantine_count == 1
    assert status_after.quarantine_bytes == len(damaged)
    assert status_after.abandoned_evidence_count == 0
    assert (
        root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.json"
    ).read_bytes() == evidence_before
    assert (
        root / "damaged" / f"conversation.quarantine.{result.quarantine_id}.jsonl"
    ).read_bytes() == payload_before


def test_repair_retries_short_warning_write_and_strictly_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            strict_reopen=lambda: _strict_reopen(root, "damaged"),
            memory_review_required=False,
        )
    assert shortened
    _strict_reopen(root, "damaged")


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
    with session_directory(root, step) as (_, directory_fd):
        with pytest.raises(OSError, match=f"fault at {step}"):
            recovery.repair_transcript(
                directory_fd,
                target,
                diagnosis._source_facts,
                strict_reopen=lambda: _strict_reopen(root, step),
                memory_review_required=False,
            )
        active = (root / step / "conversation.jsonl").read_bytes()
        if active_replaced:
            assert active != damaged
            resolved = recovery.reconcile_pending(
                directory_fd,
                target,
                strict_reopen=lambda: _strict_reopen(root, step),
            )
            assert resolved[0].state == "committed"
        else:
            assert active == damaged
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
            assert payload_files[0].read_bytes() == damaged
            record = recovery.read_recovery_record(directory_fd, quarantine_id)
            if record is None:
                record = recovery.reconcile_pending(
                    directory_fd,
                    target,
                    strict_reopen=lambda: _strict_reopen(root, step),
                )[0]
            expected = "committed" if active_replaced else "abandoned"
            assert record.state == expected


def test_source_identity_change_refuses_before_creating_evidence(
    tmp_path: Path,
) -> None:
    root, damaged, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    transcript = root / "damaged" / "conversation.jsonl"
    changed = damaged[:-4] + b"null"
    assert len(changed) == len(damaged)
    transcript.write_bytes(changed)

    with session_directory(root, "damaged") as (_, directory_fd):
        with pytest.raises(recovery.TranscriptSourceChanged):
            recovery.repair_transcript(
                directory_fd,
                recovery.RepairTarget("damaged"),
                diagnosis._source_facts,
                strict_reopen=lambda: _strict_reopen(root, "damaged"),
                memory_review_required=False,
            )
        assert (
            recovery.recovery_status(
                directory_fd, recovery.RepairTarget("damaged")
            ).quarantine_count
            == 0
        )
    assert transcript.read_bytes() == changed


def test_nested_target_resolution_uses_public_identity_and_no_symlinks(
    tmp_path: Path,
) -> None:
    _diagnosed_store(tmp_path / "diagnosis-probe")
    recovery = _recovery()
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="root")
    store.close()
    with (
        session_directory(root, "root") as (_, root_fd),
        child_directory(root_fd, "agents", create=True) as agents_fd,
        child_directory(agents_fd, "1", create=True) as child_fd,
    ):
        write_json_file(
            child_fd,
            "agent_lifecycle.json",
            {"handle": "root:1"},
        )
    with recovery.repair_target_directory(
        root, recovery.RepairTarget("root", "root:1")
    ) as child_fd:
        assert (
            json.loads(read_session_file(child_fd, "agent_lifecycle.json"))["handle"]
            == "root:1"
        )

    lifecycle = root / "root" / "agents" / "1" / "agent_lifecycle.json"
    lifecycle.write_text('{"handle":"root:2"}\n', encoding="utf-8")
    with (
        pytest.raises(ValueError, match="identity"),
        recovery.repair_target_directory(root, recovery.RepairTarget("root", "root:1")),
    ):
        pass
    lifecycle.unlink()
    lifecycle.symlink_to(root / "root" / "conversation.jsonl")
    with (
        pytest.raises(OSError),
        recovery.repair_target_directory(root, recovery.RepairTarget("root", "root:1")),
    ):
        pass


@pytest.mark.parametrize(
    ("state", "memory_step"),
    [
        ("pending", "resumed"),
        ("abandoned", "review_pending"),
        ("abandoned", "resumed"),
    ],
)
def test_record_parser_rejects_illegal_state_memory_pairs(
    tmp_path: Path, state: str, memory_step: str
) -> None:
    root, _, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            recovery.RepairTarget("damaged"),
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged"),
            memory_review_required=False,
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


def test_record_parser_rejects_unknown_fields(tmp_path: Path) -> None:
    root, _, diagnosis = _diagnosed_store(tmp_path)
    recovery = _recovery()
    target = recovery.RepairTarget("damaged")
    with session_directory(root, "damaged") as (_, directory_fd):
        result = recovery.repair_transcript(
            directory_fd,
            target,
            diagnosis._source_facts,
            strict_reopen=lambda: _strict_reopen(root, "damaged"),
            memory_review_required=False,
        )
        value = recovery.read_evidence_manifest(
            directory_fd, result.quarantine_id
        ).to_dict()
    value["unexpected"] = True
    with pytest.raises(recovery.TranscriptRecoveryIntegrityError, match="shape"):
        recovery.parse_evidence_manifest(value)
