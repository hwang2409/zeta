"""Immutable child lifecycle values, transitions, and legacy projections."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal


class ChildRunPhase(Enum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    FINISH_GATE_MODEL = "finish_gate_model"
    FINISH_GATE_WAIT_WORK = "finish_gate_wait_work"
    FINISH_GATE_WAIT_TASK_SAFETY = "finish_gate_wait_task_safety"
    FINISH_GATE_POST_TASK_MODEL = "finish_gate_post_task_model"
    FINISHING = "finishing"
    CANCELING = "canceling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ChildRunEvent(Enum):
    SPAWN_RECORDED = "spawn_recorded"
    RUNTIME_STARTED = "runtime_started"
    PROGRESS = "progress"
    BECOME_IDLE = "become_idle"
    WAKE = "wake"
    BEGIN_FINISH_GATE = "begin_finish_gate"
    GATE_WAIT = "gate_wait"
    GATE_WAIT_TASK_SAFETY = "gate_wait_task_safety"
    GATE_DELIVER = "gate_deliver"
    GATE_TASKS_DRAINED = "gate_tasks_drained"
    ACCEPT_FINISH = "accept_finish"
    FINALIZE_ACCEPTED_FINISH = "finalize_accepted_finish"
    REQUEST_CANCEL = "request_cancel"
    CANCEL_CONFIRMED = "cancel_confirmed"
    COMPLETE_UNREPORTED = "complete_unreported"
    FAIL_UNREPORTED = "fail_unreported"
    ADOPT = "adopt"
    ATTACH_CANONICAL_RECEIPT = "attach_canonical_receipt"


@dataclass(frozen=True, slots=True)
class ChildProgress:
    current_step: str
    turns_used: int = 0
    tool_calls: int = 0


@dataclass(frozen=True, slots=True)
class FinishGateData:
    deadline: float | None = None
    task_existed: bool = False


@dataclass(frozen=True, slots=True)
class ExplicitFinishReport:
    status: Literal["done", "blocked", "failed"]
    summary: str
    head_sha: str | None = None
    verdict: Literal["pass", "fail", "inconclusive"] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"done", "blocked", "failed"} or not self.summary:
            raise ValueError("invalid explicit finish report")
        if self.verdict not in {None, "pass", "fail", "inconclusive"}:
            raise ValueError("invalid explicit finish verdict")


@dataclass(frozen=True, slots=True)
class InvalidFinishReport:
    row_id: str
    error: str

    def __post_init__(self) -> None:
        if not self.row_id or not self.error:
            raise ValueError("invalid finish report evidence")
        bounded = self.error.encode("utf-8")[:256].decode("utf-8", "ignore")
        object.__setattr__(self, "error", bounded)


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    state: Literal["completed", "failed", "canceled"]
    result: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChildCleanupData:
    cancellation_requested: bool = False
    error: str | None = None
    killed_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChildRunRecord:
    run_id: str
    original_parent_run_id: str | None
    owner_run_id: str | None
    phase: ChildRunPhase
    started_at: str
    finished_at: str | None
    finish_evidence: ExplicitFinishReport | InvalidFinishReport | None
    outcome: ChildOutcome | None
    progress: ChildProgress
    finish_gate: FinishGateData | None
    cleanup: ChildCleanupData
    _legacy_json: str | None = None


@dataclass(frozen=True, slots=True)
class ChildRunFacts:
    run_id: str | None = None
    original_parent_run_id: str | None = None
    owner_run_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    progress: ChildProgress | None = None
    finish_gate: FinishGateData | None = None
    finish_evidence: ExplicitFinishReport | InvalidFinishReport | None = None
    outcome: ChildOutcome | None = None
    cleanup: ChildCleanupData | None = None


_TERMINAL_PHASES = frozenset(
    {ChildRunPhase.COMPLETED, ChildRunPhase.FAILED, ChildRunPhase.CANCELED}
)
_FINISH_GATE_PHASES = frozenset(
    {
        ChildRunPhase.FINISH_GATE_MODEL,
        ChildRunPhase.FINISH_GATE_WAIT_WORK,
        ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY,
        ChildRunPhase.FINISH_GATE_POST_TASK_MODEL,
    }
)


def _require_record(current: ChildRunRecord | None) -> ChildRunRecord:
    if current is None:
        raise ValueError("child transition requires an existing record")
    return current


def _require_outcome(
    facts: ChildRunFacts, state: Literal["completed", "failed", "canceled"]
) -> ChildOutcome:
    outcome = facts.outcome
    if outcome is None or outcome.state != state:
        raise ValueError(f"child transition requires a {state} outcome")
    return outcome


def transition_child_run(
    current: ChildRunRecord | None,
    event: ChildRunEvent,
    facts: ChildRunFacts,
) -> ChildRunRecord:
    """Apply one named child transition or reject the edge."""

    if current is None:
        if event is not ChildRunEvent.SPAWN_RECORDED:
            raise ValueError("unnamed child lifecycle edge")
        if not facts.run_id or not facts.started_at or facts.progress is None:
            raise ValueError("spawn facts are incomplete")
        return ChildRunRecord(
            run_id=facts.run_id,
            original_parent_run_id=facts.original_parent_run_id,
            owner_run_id=facts.owner_run_id,
            phase=ChildRunPhase.STARTING,
            started_at=facts.started_at,
            finished_at=None,
            finish_evidence=None,
            outcome=None,
            progress=facts.progress,
            finish_gate=None,
            cleanup=facts.cleanup or ChildCleanupData(),
        )

    record = _require_record(current)
    phase = record.phase
    if event is ChildRunEvent.RUNTIME_STARTED and phase is ChildRunPhase.STARTING:
        return replace(record, phase=ChildRunPhase.RUNNING)
    if event is ChildRunEvent.PROGRESS and phase not in _TERMINAL_PHASES | {
        ChildRunPhase.STARTING,
        ChildRunPhase.CANCELING,
        ChildRunPhase.FINISHING,
    }:
        progress = facts.progress
        if (
            progress is None
            or not progress.current_step
            or progress.turns_used < record.progress.turns_used
            or progress.tool_calls < record.progress.tool_calls
        ):
            raise ValueError("invalid child progress")
        return replace(record, progress=progress)
    if event is ChildRunEvent.BECOME_IDLE and phase is ChildRunPhase.RUNNING:
        return replace(record, phase=ChildRunPhase.IDLE)
    if event is ChildRunEvent.WAKE and phase is ChildRunPhase.IDLE:
        return replace(record, phase=ChildRunPhase.RUNNING)
    if event is ChildRunEvent.BEGIN_FINISH_GATE and phase is ChildRunPhase.RUNNING:
        if facts.finish_gate is None:
            raise ValueError("finish gate facts are required")
        return replace(
            record,
            phase=ChildRunPhase.FINISH_GATE_MODEL,
            finish_gate=facts.finish_gate,
        )
    if event is ChildRunEvent.GATE_WAIT and phase in {
        ChildRunPhase.FINISH_GATE_MODEL,
        ChildRunPhase.FINISH_GATE_POST_TASK_MODEL,
    }:
        return replace(record, phase=ChildRunPhase.FINISH_GATE_WAIT_WORK)
    if event is ChildRunEvent.GATE_WAIT_TASK_SAFETY and phase in {
        ChildRunPhase.FINISH_GATE_MODEL,
        ChildRunPhase.FINISH_GATE_WAIT_WORK,
    }:
        if record.finish_gate is None or not record.finish_gate.task_existed:
            raise ValueError("task evidence is required")
        return replace(record, phase=ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY)
    if event is ChildRunEvent.GATE_DELIVER and phase in {
        ChildRunPhase.FINISH_GATE_WAIT_WORK,
        ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY,
    }:
        return replace(record, phase=ChildRunPhase.FINISH_GATE_MODEL)
    if (
        event is ChildRunEvent.GATE_TASKS_DRAINED
        and phase is ChildRunPhase.FINISH_GATE_WAIT_TASK_SAFETY
    ):
        return replace(record, phase=ChildRunPhase.FINISH_GATE_POST_TASK_MODEL)
    if event is ChildRunEvent.ACCEPT_FINISH and phase in {
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
    }:
        if not isinstance(facts.finish_evidence, ExplicitFinishReport):
            raise ValueError("accepted finish evidence is required")
        return replace(
            record,
            phase=ChildRunPhase.FINISHING,
            finish_evidence=facts.finish_evidence,
        )
    if (
        event is ChildRunEvent.FINALIZE_ACCEPTED_FINISH
        and phase is ChildRunPhase.FINISHING
    ):
        evidence = record.finish_evidence
        if evidence is None:
            raise ValueError("accepted finish evidence is required")
        failed = isinstance(evidence, InvalidFinishReport) or evidence.status == "failed"
        state: Literal["completed", "failed"] = "failed" if failed else "completed"
        if isinstance(evidence, InvalidFinishReport):
            outcome = ChildOutcome("failed", "finish report invalid", "finish_report_invalid")
        else:
            outcome = ChildOutcome(state, evidence.summary)
        return replace(
            record,
            phase=ChildRunPhase.FAILED if failed else ChildRunPhase.COMPLETED,
            finished_at=facts.finished_at,
            outcome=outcome,
            cleanup=facts.cleanup or record.cleanup,
        )
    if event is ChildRunEvent.REQUEST_CANCEL and phase in {
        ChildRunPhase.STARTING,
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
        *_FINISH_GATE_PHASES,
    }:
        return replace(record, phase=ChildRunPhase.CANCELING)
    if event is ChildRunEvent.CANCEL_CONFIRMED and phase is ChildRunPhase.CANCELING:
        return replace(
            record,
            phase=ChildRunPhase.CANCELED,
            finished_at=facts.finished_at,
            outcome=_require_outcome(facts, "canceled"),
            cleanup=facts.cleanup or record.cleanup,
        )
    if event is ChildRunEvent.COMPLETE_UNREPORTED and phase in {
        ChildRunPhase.RUNNING,
        *_FINISH_GATE_PHASES,
    }:
        return replace(
            record,
            phase=ChildRunPhase.COMPLETED,
            finished_at=facts.finished_at,
            outcome=_require_outcome(facts, "completed"),
        )
    if event is ChildRunEvent.FAIL_UNREPORTED and phase in {
        ChildRunPhase.STARTING,
        ChildRunPhase.RUNNING,
        ChildRunPhase.IDLE,
        *_FINISH_GATE_PHASES,
    }:
        return replace(
            record,
            phase=ChildRunPhase.FAILED,
            finished_at=facts.finished_at,
            outcome=_require_outcome(facts, "failed"),
            cleanup=facts.cleanup or record.cleanup,
        )
    if event is ChildRunEvent.ADOPT and phase not in _TERMINAL_PHASES:
        if not facts.owner_run_id:
            raise ValueError("adoption requires an owner")
        return replace(record, owner_run_id=facts.owner_run_id)
    if event is ChildRunEvent.ATTACH_CANONICAL_RECEIPT and phase in _TERMINAL_PHASES:
        outcome = facts.outcome
        if outcome is None or record.outcome is None or outcome.state != record.outcome.state:
            raise ValueError("canonical receipt conflicts with child outcome")
        return replace(record, outcome=outcome)
    raise ValueError("unnamed child lifecycle edge")


def parse_legacy_child_lifecycle(lifecycle: dict[str, object]) -> ChildRunRecord:
    """Parse one valid legacy lifecycle without changing its projection."""

    handle = lifecycle.get("handle")
    state = lifecycle.get("state")
    started_at = lifecycle.get("started_at")
    finished_at = lifecycle.get("finished_at")
    if type(handle) is not str or not handle or type(started_at) is not str:
        raise ValueError("invalid legacy child lifecycle identity")
    legacy_phases = {
        phase.value: phase
        for phase in (
            ChildRunPhase.RUNNING,
            ChildRunPhase.IDLE,
            ChildRunPhase.FINISHING,
            ChildRunPhase.COMPLETED,
            ChildRunPhase.CANCELED,
            ChildRunPhase.FAILED,
        )
    }
    if type(state) is not str or state not in legacy_phases:
        raise ValueError("invalid legacy child lifecycle state")
    phase = legacy_phases[state]
    if phase in _TERMINAL_PHASES:
        if type(finished_at) is not str:
            raise ValueError("terminal legacy lifecycle has no finish time")
        outcome = ChildOutcome(
            phase.value,  # type: ignore[arg-type]
            str(lifecycle.get("final_result", "")),
        )
    else:
        if finished_at is not None:
            raise ValueError("nonterminal legacy lifecycle has a finish time")
        outcome = None
    step = lifecycle.get("current_step", "unknown")
    turns = lifecycle.get("turns_used", 0)
    calls = lifecycle.get("tool_calls", 0)
    if type(step) is not str or type(turns) is not int or type(calls) is not int:
        raise ValueError("invalid legacy child progress")
    return ChildRunRecord(
        run_id=handle,
        original_parent_run_id=None,
        owner_run_id=None,
        phase=phase,
        started_at=started_at,
        finished_at=finished_at if type(finished_at) is str else None,
        finish_evidence=None,
        outcome=outcome,
        progress=ChildProgress(step, turns, calls),
        finish_gate=None,
        cleanup=ChildCleanupData(),
        _legacy_json=json.dumps(lifecycle, ensure_ascii=False, separators=(",", ":")),
    )


def project_legacy_child_lifecycle(record: ChildRunRecord) -> dict[str, object]:
    """Return the exact legacy value represented by a parsed record."""

    if record._legacy_json is None:
        raise ValueError("child record has no legacy projection")
    value = json.loads(record._legacy_json)
    if type(value) is not dict:
        raise ValueError("invalid legacy child projection")
    return value


def select_legacy_child_lifecycle(
    durable: object,
    live: dict[str, object] | None,
) -> object:
    """Select a read projection; a valid durable terminal value always wins."""

    if type(durable) is dict:
        try:
            record = parse_legacy_child_lifecycle(durable)
        except ValueError:
            pass
        else:
            if record.phase in _TERMINAL_PHASES:
                return project_legacy_child_lifecycle(record)
    selected = live if live is not None else durable
    if selected is None:
        return None
    return json.loads(json.dumps(selected, ensure_ascii=False))
