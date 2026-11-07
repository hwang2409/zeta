from pathlib import Path

from zeta.agent.recovery_lease import append_transcript_recovery_notification
from zeta.core.store import ConversationStore
from zeta.core.transcript_recovery import RepairResult, RepairTarget


def test_child_recovery_notification_deduplicates_by_quarantine_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    try:
        target = RepairTarget(store.session_id, f"{store.session_id}:1")
        first_result = RepairResult(
            "a" * 32, target, 0, "b" * 64, "c" * 64, False
        )
        first, appended = append_transcript_recovery_notification(
            store, first_result, child_session_path="/child"
        )
        again, repeated = append_transcript_recovery_notification(
            store, first_result, child_session_path="/child"
        )
        assert appended is True
        assert repeated is False
        assert first.id == again.id

        second_result = RepairResult(
            "d" * 32, target, 0, "b" * 64, "c" * 64, False
        )
        second, appended_new = append_transcript_recovery_notification(
            store, second_result, child_session_path="/child"
        )
        assert appended_new is True
        assert second.id != first.id
    finally:
        store.close()
