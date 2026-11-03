from __future__ import annotations

from dataclasses import replace

import pytest

from zeta.core.child_lifecycle import (
    ChildCleanupData,
    ChildOutcome,
    ChildProgress,
    ChildRunEvent,
    ChildRunFacts,
    ChildRunPhase,
    ChildRunRecord,
    ExplicitFinishReport,
    FinishGateData,
    parse_legacy_child_lifecycle,
    project_legacy_child_lifecycle,
    transition_child_run,
)

_TERMINAL = {
    ChildRunPhase.COMPLETED,
    ChildRunPhase.FAILED,
    ChildRunPhase.CANCELED,
}
_GATE = {
    ChildRunPhase.FINISH_GATE_MODEL,
    ChildRunPhase.FINISH_GATE_WAIT_WORK,
    ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY,
    ChildRunPhase.FINISH_GATE_POST_TASK_MODEL,
}
_ALLOWED = {
    ChildRunEvent.RUNTIME_STARTED: {ChildRunPhase.STARTING},
    ChildRunEvent.PROGRESS: {
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
        *_GATE,
    },
    ChildRunEvent.BECOME_IDLE: {ChildRunPhase.RUNNING},
    ChildRunEvent.WAKE: {ChildRunPhase.IDLE},
    ChildRunEvent.BEGIN_FINISH_GATE: {ChildRunPhase.RUNNING},
    ChildRunEvent.GATE_WAIT: {
        ChildRunPhase.FINISH_GATE_MODEL,
        ChildRunPhase.FINISH_GATE_POST_TASK_MODEL,
    },
    ChildRunEvent.GATE_WAIT_TASK_SAFETY: {
        ChildRunPhase.FINISH_GATE_MODEL,
        ChildRunPhase.FINISH_GATE_WAIT_WORK,
    },
    ChildRunEvent.GATE_DELIVER: {
        ChildRunPhase.FINISH_GATE_WAIT_WORK,
        ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY,
    },
    ChildRunEvent.GATE_TASKS_DRAINED: {
        ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY
    },
    ChildRunEvent.ACCEPT_FINISH: {ChildRunPhase.RUNNING, ChildRunPhase.IDLE},
    ChildRunEvent.FINALIZE_ACCEPTED_FINISH: {ChildRunPhase.FINISHING},
    ChildRunEvent.REQUEST_CANCEL: {
        ChildRunPhase.STARTING,
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
        *_GATE,
    },
    ChildRunEvent.CANCEL_CONFIRMED: {ChildRunPhase.CANCELING},
    ChildRunEvent.COMPLETE_UNREPORTED: {ChildRunPhase.RUNNING, *_GATE},
    ChildRunEvent.FAIL_UNREPORTED: {
        ChildRunPhase.STARTING,
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
        *_GATE,
    },
    ChildRunEvent.ADOPT: set(ChildRunPhase) - _TERMINAL,
    ChildRunEvent.ATTACH_CANONICAL_RECEIPT: _TERMINAL,
}


def _record(phase: ChildRunPhase) -> ChildRunRecord:
    state = phase.value if phase in _TERMINAL else "failed"
    outcome = (
        ChildOutcome(state, "result")  # type: ignore[arg-type]
        if phase in _TERMINAL
        else None
    )
    return ChildRunRecord(
        run_id="parent:1",
        original_parent_run_id=None,
        owner_run_id="parent",
        phase=phase,
        started_at="start",
        finished_at="finish" if phase in _TERMINAL else None,
        finish_evidence=(
            ExplicitFinishReport("failed", "summary")
            if phase is ChildRunPhase.FINISHING
            else None
        ),
        outcome=outcome,
        progress=ChildProgress("step", 1, 1),
        finish_gate=(FinishGateData(task_existed=True) if phase in _GATE else None),
        cleanup=ChildCleanupData(),
    )


def _facts(event: ChildRunEvent, phase: ChildRunPhase) -> ChildRunFacts:
    outcome = None
    if event is ChildRunEvent.CANCEL_CONFIRMED:
        outcome = ChildOutcome("canceled", "canceled")
    elif event is ChildRunEvent.COMPLETE_UNREPORTED:
        outcome = ChildOutcome("completed", "complete")
    elif event in {
        ChildRunEvent.FAIL_UNREPORTED,
        ChildRunEvent.FINALIZE_ACCEPTED_FINISH,
    }:
        outcome = ChildOutcome("failed", "failed")
    elif event is ChildRunEvent.ATTACH_CANONICAL_RECEIPT:
        outcome = _record(phase).outcome
    return ChildRunFacts(
        owner_run_id="new-parent",
        finished_at="finish",
        progress=ChildProgress("next", 2, 2),
        finish_gate=FinishGateData(task_existed=True),
        finish_evidence=ExplicitFinishReport("failed", "summary"),
        outcome=outcome,
    )


def test_transition_table_rejects_every_unnamed_edge() -> None:
    for phase in ChildRunPhase:
        for event in ChildRunEvent:
            if event is ChildRunEvent.SPAWN_RECORDED:
                allowed = False
            else:
                allowed = phase in _ALLOWED[event]
            if allowed:
                transition_child_run(_record(phase), event, _facts(event, phase))
            else:
                with pytest.raises(ValueError, match="unnamed child lifecycle edge"):
                    transition_child_run(_record(phase), event, _facts(event, phase))

    spawned = transition_child_run(
        None,
        ChildRunEvent.SPAWN_RECORDED,
        ChildRunFacts(
            run_id="parent:1",
            owner_run_id="parent",
            started_at="start",
            progress=ChildProgress("starting"),
        ),
    )
    assert spawned.phase is ChildRunPhase.STARTING
    for event in set(ChildRunEvent) - {ChildRunEvent.SPAWN_RECORDED}:
        with pytest.raises(ValueError, match="unnamed child lifecycle edge"):
            transition_child_run(None, event, ChildRunFacts())


@pytest.mark.parametrize(
    "state",
    ["running", "idle", "finishing", "completed", "canceled", "failed"],
)
def test_legacy_lifecycle_projects_identically(state: str) -> None:
    terminal = state in {"completed", "canceled", "failed"}
    lifecycle: dict[str, object] = {
        "handle": "parent:1",
        "state": state,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00" if terminal else None,
        "elapsed": 1.0,
        "turns_used": 2,
        "tool_calls": 3,
        "current_step": "done" if terminal else "working",
        "depth": 1,
        "agent_type": "worker",
        "description": "task",
        "selected_accounts": {"codex": "default"},
    }
    if terminal:
        lifecycle["final_result"] = "result"

    record = parse_legacy_child_lifecycle(lifecycle)

    assert project_legacy_child_lifecycle(record) == lifecycle
    assert replace(record).phase.value == state
