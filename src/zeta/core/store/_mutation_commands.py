"""Private typed commands for legacy conversation-store mutations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

_PATCH_FIELDS = (
    "cwd",
    "_active_approval_requests",
    "_active_approval_resolutions",
    "_active_tool_results",
    "_active_agent_call_ids",
    "_active_agent_receipts",
    "_notification_ids",
    "_active_notifications",
    "_active_completion_notifications",
    "_active_notification_presentations",
    "_pending_prompt_ids",
    "_active_pending_prompt_entries",
    "_active_pending_prompt_messages",
    "_active_notification_delivery_message",
    "_active_last_message",
    "_active_last_turn_end",
    "_task_notification_ids",
    "_sent_message_receipts",
    "_client_deliveries",
    "_client_delivery_outcomes",
    "_agent_counter",
    "_agent_children",
    "_agent_parent",
    "_agent_canceled",
    "_agent_lifecycle",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _StoreMutationCommand:
    """Shared reconciliation facts; each subclass fixes one storage operation."""

    baseline_count: int | None = None
    baseline_digest: bytes | None = None
    baseline_patch: Mapping[str, object] | None = None
    patch_fields: ClassVar[tuple[str, ...]] = _PATCH_FIELDS
    accept_external_generation: bool = False

    def apply(self, store: Any) -> Any:
        raise NotImplementedError

    def after_persisted(self) -> None:
        pass


@dataclass(frozen=True, slots=True, kw_only=True)
class _RefreshCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store.refresh()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendMessageCommand(_StoreMutationCommand):
    message: Any
    parent_id: str | None
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_message(self.message, parent_id=self.parent_id)

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendChildRunStartedCommand(_StoreMutationCommand):
    run_id: str
    parent_run_id: str | None
    name: str | None
    task: str
    started_at: str
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_child_run_started(
            self.run_id, self.parent_run_id, self.name, self.task, self.started_at
        )

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendApprovalHistoryCommand(_StoreMutationCommand):
    payload: Mapping[str, Any]

    def apply(self, store: Any) -> Any:
        return store.append_approval_history(self.payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendClientDeliveryCommand(_StoreMutationCommand):
    delivery_id: str
    method: Any
    status: Any
    outcome: dict[str, object]
    message: Any = None
    accepted_message: Any = None
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_client_delivery(
            self.delivery_id,
            self.method,
            self.status,
            self.outcome,
            message=self.message,
            accepted_message=self.accepted_message,
        )

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _DeliverClientSteeringCommand(_StoreMutationCommand):
    batch: Any

    def apply(self, store: Any) -> None:
        store._deliver_client_steering(self.batch)


@dataclass(frozen=True, slots=True, kw_only=True)
class _DropClientSteeringCommand(_StoreMutationCommand):
    queued: tuple[Any, ...]
    reason: Any

    def apply(self, store: Any) -> Any:
        return store._drop_client_steering(self.queued, self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendTurnEndCommand(_StoreMutationCommand):
    data: dict[str, Any]

    def apply(self, store: Any) -> None:
        store._append_row("turn_end", self.data)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordNotificationDeliveryCommand(_StoreMutationCommand):
    notification_id: str
    presented: bool
    acknowledged: bool

    def apply(self, store: Any) -> Any:
        return store.record_agent_notification_delivery(
            self.notification_id,
            presented=self.presented,
            acknowledged=self.acknowledged,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _PresentNotificationsCommand(_StoreMutationCommand):
    notification_ids: list[str]

    def apply(self, store: Any) -> None:
        store.mark_agent_notifications_presented_to_tui(self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AcknowledgeNotificationsCommand(_StoreMutationCommand):
    notification_ids: list[str]

    def apply(self, store: Any) -> None:
        store.acknowledge_agent_notifications(self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _WarmStepDeliveryCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store._warm_step_delivery_writer()


@dataclass(frozen=True, slots=True, kw_only=True)
class _CommitDeliveryInputCommand(_StoreMutationCommand):
    item: Any

    def apply(self, store: Any) -> Any:
        return store._commit_delivery_input(self.item)


@dataclass(frozen=True, slots=True, kw_only=True)
class _StageDeliveryRequestCommand(_StoreMutationCommand):
    notification_budget: int | None
    token_counter: Any
    notification_selector: Any
    notification_context: Any

    def apply(self, store: Any) -> Any:
        return store._stage_delivery_request(
            self.notification_budget,
            self.token_counter,
            self.notification_selector,
            self.notification_context,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _AcknowledgeStagedDeliveryCommand(_StoreMutationCommand):
    prompt_id: str | None
    notification_ids: tuple[str, ...]

    def apply(self, store: Any) -> Any:
        return store._acknowledge_staged_delivery(
            self.prompt_id, self.notification_ids
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _AdmitStepDeliveryTerminalCommand(_StoreMutationCommand):
    intent: Any

    def apply(self, store: Any) -> Any:
        return store._admit_step_delivery_terminal(self.intent)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AcceptExplicitFinishCommand(_StoreMutationCommand):
    report: dict[str, object]
    notification_ids: frozenset[str]
    prompt_id: str | None

    def apply(self, store: Any) -> Any:
        return store._accept_explicit_finish(
            self.report, self.notification_ids, self.prompt_id
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _CancelExplicitChildCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> bool:
        return store._cancel_explicit_child()


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExplicitFinishReportCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> Any:
        return store._explicit_finish_report()


@dataclass(frozen=True, slots=True, kw_only=True)
class _IdleNudgeEligibleCommand(_StoreMutationCommand):
    notification_ids: frozenset[str]

    def apply(self, store: Any) -> bool:
        return store._idle_nudge_eligible(self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResolveApprovalCommand(_StoreMutationCommand):
    request_id: str
    decision: str
    reason: str | None

    def apply(self, store: Any) -> None:
        store.resolve_approval(self.request_id, self.decision, reason=self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class _MarkAgentCanceledCommand(_StoreMutationCommand):
    tool_call_id: str

    def apply(self, store: Any) -> None:
        store.mark_agent_canceled(self.tool_call_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RegisterAgentChildCommand(_StoreMutationCommand):
    tool_call: Any
    child_session_path: str
    description: str
    agent_type: str | None
    background: bool
    child_instance_id: str | None
    accepts_follow_ups: bool
    explicit_finish: bool
    turns_used: int

    def apply(self, store: Any) -> Any:
        return store.register_agent_child(
            self.tool_call,
            child_session_path=self.child_session_path,
            description=self.description,
            agent_type=self.agent_type,
            background=self.background,
            child_instance_id=self.child_instance_id,
            accepts_follow_ups=self.accepts_follow_ups,
            explicit_finish=self.explicit_finish,
            turns_used=self.turns_used,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentChildCommand(_StoreMutationCommand):
    marker_key: str

    def apply(self, store: Any) -> None:
        store.finish_agent_child(self.marker_key)


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentParentCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store.finish_agent_parent()


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentLifecycleCommand(_StoreMutationCommand):
    state: str
    final_result: str
    turns_used: int | None
    killed_task_ids: list[str] | None
    killed_task_count: int | None
    killed_task_ids_truncated: bool | None
    canonical_receipt: bool

    def apply(self, store: Any) -> None:
        store.finish_agent_lifecycle(
            self.state,
            final_result=self.final_result,
            turns_used=self.turns_used,
            killed_task_ids=self.killed_task_ids,
            killed_task_count=self.killed_task_count,
            killed_task_ids_truncated=self.killed_task_ids_truncated,
            canonical_receipt=self.canonical_receipt,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendPendingPromptCommand(_StoreMutationCommand):
    text: str

    def apply(self, store: Any) -> Any:
        return store.append_pending_prompt(self.text)
