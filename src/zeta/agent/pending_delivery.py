"""Own pending-prompt sends, resume recovery, delivery, and acknowledgement."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from ..core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..core.store._pending_prompts import (
    PENDING_PROMPT_ID_METADATA,
    PendingPromptWriterReloadedError,
)
from ..core.store._step_delivery import (
    _MUTATION_ONLY_KINDS,
    DeliveryInput,
    DeliveryPublications,
    DeliverySource,
    NotificationInput,
    ProducerResult,
    PromptInput,
    StagedDelivery,
    TerminalIntent,
    validate_delivery_input,
)
from ..protocol.types import ErrorInfo, Message, MessageOrigin
from ._delivery_notifications import (
    _message_notification_ids,
    _select_notification_delivery,
)
from .conversation_channel import ConversationChannel, has_follow_up_loop

if TYPE_CHECKING:
    from ..core.context import ContextAssembler
    from ..core.context_parts.admission import AdmittedSessionRequest
    from ..core.store._client_delivery import ClientSteeringBatch

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0
SHUTDOWN_DELIVERY_TIMEOUT_SECONDS = 1.0
QUESTION_ANSWER_PATTERN = re.compile(r"^Re ([^:\s]+):(?:\s|$)")
_PREVIEW_WHITESPACE = re.compile(r"\s+")
_LOGGER = logging.getLogger(__name__)


def delivery_preview(text: str) -> str:
    """Return the shared bounded one-line delivery preview."""

    normalized = _PREVIEW_WHITESPACE.sub(" ", text).strip()
    if len(normalized) <= 120:
        return normalized
    return normalized[:119] + "…"


RequestCause = Literal["after_tools", "terminal_fallback", "idle_start", "retry"]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    messages: tuple[Message, ...]
    steering: ClientSteeringBatch | None
    notification_ids: tuple[str, ...] = ()
    prompt_id: str | None = None
    overflow_message: str | None = None
    deferred_low_capacity: bool = False
    low_capacity_notice: str | None = None
    admitted_request: AdmittedSessionRequest | None = None

    @property
    def has_active_delivery(self) -> bool:
        return bool(self.prompt_id or self.notification_ids)

    @property
    def contains_turn_context(self) -> bool:
        return any(message.metadata.get("turn_context") for message in self.messages)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    pending: bool
    closed_empty: bool


def complete_notification_context_staging(
    prepared: PreparedRequest,
    system_message: Message | None,
    notification_turn: bool,
    on_persisted: Callable[[], None] | None,
) -> tuple[Message | None, Callable[[], None] | None]:
    if not notification_turn or not prepared.contains_turn_context:
        return system_message, on_persisted
    if on_persisted is not None:
        on_persisted()
    return None, None


def delivery_disposition(
    flow: StepDeliveryFlow,
    prepared: PreparedRequest,
    *,
    notification_turn: bool,
    notice_sink: Callable[[str], None] | None,
) -> Literal["dispatch", "deferred"] | ErrorInfo:
    if prepared.low_capacity_notice is not None and notice_sink is not None:
        notice_sink(prepared.low_capacity_notice)
    if prepared.deferred_low_capacity and notification_turn:
        flow.release_active()
        return "deferred"
    if prepared.overflow_message is not None:
        flow.release_active()
        return ErrorInfo("context_length_exceeded", prepared.overflow_message)
    return "dispatch"


class StepDeliveryFlow:
    """Own production, request staging, and acknowledgement for one store."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        wake_sink: Callable[[DeliverySource], None] | None = None,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        self.store = store
        self._loop = asyncio.get_running_loop()
        self._wake_sink = wake_sink
        self._context_assembler = context_assembler
        self._edges: set[DeliverySource] = set()
        self._active: StagedDelivery | None = None
        self._active_steering: ClientSteeringBatch | None = None
        self._prompt_priority: str | None = None
        self._low_capacity_notifications = False
        self._low_capacity_notice_emitted = False
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Warm the isolated writer before this flow accepts timed input."""

        self._require_owner_loop()
        if self._initialized:
            return
        async with self._initialize_lock:
            if not self._initialized:
                await self.store._to_thread_durable(
                    self.store._warm_step_delivery_writer
                )
                self._initialized = True

    async def produce(self, item: DeliveryInput, /) -> ProducerResult:
        source = validate_delivery_input(item)
        self._require_owner_loop()
        await self.initialize()
        commit_window = (
            max(0.0, item.deadline - time.monotonic())
            if isinstance(item, PromptInput) and item.deadline is not None
            else None
        )
        try:
            result = await self.store._to_thread_durable(
                self.store._commit_delivery_input, item
            )
        except PendingPromptWriterReloadedError:
            if not isinstance(item, PromptInput) or item.deadline is None:
                raise
            assert commit_window is not None
            retry = replace(item, deadline=time.monotonic() + commit_window)
            result = await self.store._to_thread_durable(
                self.store._commit_delivery_input, retry
            )
        if result.changed and not (
            isinstance(item, NotificationInput)
            and (
                item.kind in _MUTATION_ONLY_KINDS
                or item.data.get("display_only") is True
            )
        ):
            self._arm_edge(source)
        return result

    def wake(self, source: DeliverySource) -> None:
        if source not in {"prompt", "notification"}:
            raise ValueError("unknown delivery source")
        try:
            owner_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            owner_loop = False
        if owner_loop:
            self._arm_edge(source)
        else:
            self._loop.call_soon_threadsafe(self._arm_edge, source)

    async def prepare_request(
        self,
        cause: RequestCause,
        assembled: Sequence[Message],
        steering: ClientSteeringBatch | None,
        tool_schemas: Sequence[Mapping[str, Any]] = (),
        *,
        notification_context: Message | None = None,
    ) -> PreparedRequest:
        """Stage one fixed suffix within the calibrated whole-request budget."""

        self._require_owner_loop()
        await self.initialize()
        assembler = self._context_assembler
        if cause == "retry" and self._active is not None:
            staged = self._active
            steering = self._active_steering
        else:
            staged = await self._stage_new_request(
                cause, assembler, steering, notification_context
            )
            if self._prompt_priority is not None:
                steering = None
            self._active_steering = steering
        if assembler is None:
            return self._compose(staged, assembled, steering)
        tool_tokens = await asyncio.to_thread(
            assembler.estimate_tool_schemas, tool_schemas
        )
        prepared = await self._prepare_budgeted(
            staged,
            assembled,
            steering,
            max(0, assembler.estimator_token_budget - tool_tokens),
            notification_context,
        )
        if prepared.overflow_message is None:
            from ..core.context import BudgetExceeded

            try:
                admitted = await assembler.admit_request(
                    prepared.messages, tool_schemas
                )
            except BudgetExceeded:
                request_tokens = await asyncio.to_thread(
                    assembler.estimate_request, prepared.messages, tool_schemas
                )
                return self._request_overflow(
                    staged, steering, request_tokens, assembler.estimator_token_budget
                )
            prepared = replace(prepared, admitted_request=admitted)
        return prepared

    async def _stage_new_request(
        self,
        cause: RequestCause,
        assembler: ContextAssembler | None,
        steering: ClientSteeringBatch | None,
        notification_context: Message | None,
    ) -> StagedDelivery:
        if cause not in {"after_tools", "terminal_fallback", "idle_start", "retry"}:
            raise ValueError("unknown request cause")
        self._edges.clear()
        if assembler is None:
            staged = await self.store._to_thread_durable(
                self.store._stage_delivery_request,
                None,
                None,
                _select_notification_delivery,
                notification_context,
            )
        else:
            staged = await self.store._to_thread_durable(
                self.store._stage_delivery_request,
                0,
                assembler.estimate_messages,
                _select_notification_delivery,
                None,
            )
        self._active = staged
        self._active_steering = steering
        if staged.more_pending:
            if staged.prompt_id is not None:
                self._edges.add("prompt")
            self._edges.add("notification")
        return staged

    async def _prepare_budgeted(
        self,
        staged: StagedDelivery,
        assembled: Sequence[Message],
        steering: ClientSteeringBatch | None,
        budget: int,
        notification_context: Message | None,
    ) -> PreparedRequest:
        assembler = self._context_assembler
        assert assembler is not None
        context = self._without_staged(assembled, staged)
        context_tokens, prompt_tokens, steering_tokens = await asyncio.gather(
            asyncio.to_thread(assembler.estimate_messages, context),
            asyncio.to_thread(
                assembler.estimate_messages,
                () if staged.prompt_message is None else (staged.prompt_message,),
            ),
            asyncio.to_thread(
                assembler.estimate_messages,
                () if steering is None else steering.messages,
            ),
        )
        if context_tokens + prompt_tokens + steering_tokens <= budget:
            return await self._add_notification(
                staged,
                context,
                steering,
                budget,
                context_tokens + prompt_tokens + steering_tokens,
                notification_context,
            )
        context = await self._reserved_context(prompt_tokens + steering_tokens)
        context = self._without_staged(context, staged)
        context_tokens = await asyncio.to_thread(assembler.estimate_messages, context)
        if context_tokens + prompt_tokens + steering_tokens <= budget:
            return await self._add_notification(
                staged,
                context,
                steering,
                budget,
                context_tokens + prompt_tokens + steering_tokens,
                notification_context,
            )
        return self._defer_fixed_suffix(
            staged,
            context,
            steering,
            context_tokens,
            prompt_tokens,
            steering_tokens,
            budget,
        )

    async def _reserved_context(self, reserved_tokens: int) -> list[Message]:
        from ..core.context import BudgetExceeded

        assembler = self._context_assembler
        assert assembler is not None
        try:
            return await assembler.assemble(
                force=True,
                emergency=True,
                reserved_estimator_tokens=reserved_tokens,
            )
        except BudgetExceeded:
            return await assembler.assemble(force=True, emergency=True)

    def _defer_fixed_suffix(
        self,
        staged: StagedDelivery,
        context: Sequence[Message],
        steering: ClientSteeringBatch | None,
        context_tokens: int,
        prompt_tokens: int,
        steering_tokens: int,
        budget: int,
    ) -> PreparedRequest:
        assembler = self._context_assembler
        assert assembler is not None
        if staged.prompt_id is not None:
            self._prompt_priority = staged.prompt_id
            self._edges.add("prompt")
        steering_only = self._compose(
            staged,
            context,
            steering,
            include_notification=False,
            include_prompt=False,
        )
        if steering is not None and context_tokens + steering_tokens <= budget:
            self._active = self._request_active(
                staged, notification=False, prompt=False
            )
            return steering_only
        self._active = self._request_active(staged, notification=False, prompt=False)
        self._active_steering = None
        if steering is None:
            overflow = (
                "the pending message was not sent: it does not fit in this model's "
                "context window "
                f"({prompt_tokens} of {assembler.estimator_token_budget} tokens). "
                "compact the "
                "context or switch to a larger model."
            )
        else:
            overflow = (
                "your message was not sent: it is larger than this model's context "
                "window "
                f"({steering_tokens} of {assembler.estimator_token_budget} tokens). "
                "shorten it and resend "
                "it, or switch to a larger model."
            )
        return PreparedRequest((), None, overflow_message=overflow)

    def _request_overflow(
        self,
        staged: StagedDelivery,
        steering: ClientSteeringBatch | None,
        request_tokens: int,
        budget: int,
    ) -> PreparedRequest:
        if staged.prompt_id is not None:
            self._prompt_priority = staged.prompt_id
            self._edges.add("prompt")
        self._active = self._request_active(staged, notification=False, prompt=False)
        self._active_steering = None
        subject = "your message" if steering is not None else "the pending message"
        action = (
            "shorten it and resend it, or switch to a larger model"
            if steering is not None
            else "compact the context or switch to a larger model"
        )
        return PreparedRequest(
            (),
            None,
            overflow_message=(
                f"{subject} was not sent: it is larger than this model's context "
                f"window ({request_tokens} of {budget} tokens). {action}."
            ),
        )

    @staticmethod
    def _without_staged(
        assembled: Sequence[Message], staged: StagedDelivery
    ) -> list[Message]:
        notification_ids = frozenset(staged.notification_ids)
        return [
            message
            for message in assembled
            if (
                staged.prompt_id is None
                or message.metadata.get(PENDING_PROMPT_ID_METADATA) != staged.prompt_id
            )
            and not notification_ids.intersection(_message_notification_ids(message))
            and not (
                staged.prompt_message is not None
                and staged.prompt_message.metadata.get(PENDING_PROMPT_ID_METADATA)
                is None
                and message == staged.prompt_message
            )
        ]

    async def _add_notification(
        self,
        staged: StagedDelivery,
        context: Sequence[Message],
        steering: ClientSteeringBatch | None,
        budget: int,
        without_tokens: int,
        notification_context: Message | None,
    ) -> PreparedRequest:
        assembler = self._context_assembler
        assert assembler is not None
        context_tokens = (
            await asyncio.to_thread(
                assembler.estimate_messages, (notification_context,)
            )
            if notification_context is not None
            else 0
        )
        capacity = min(
            assembler.delivery_notification_budget,
            max(0, budget - without_tokens - context_tokens),
        )
        selected = staged
        if capacity > 0:
            selected = await self.store._to_thread_durable(
                self.store._stage_delivery_request,
                capacity,
                assembler.estimate_messages,
                _select_notification_delivery,
                notification_context,
            )
        request = self._compose(
            selected,
            context,
            steering,
            include_notification=selected.notification_message is not None,
        )
        deferred = (
            "notification" in self._edges and selected.notification_message is None
        )
        notice = None
        if deferred:
            self._low_capacity_notifications = True
            if not self._low_capacity_notice_emitted:
                count = sum(
                    entry.data.get("display_only") is not True
                    for entry in self.store._active_notifications.values()
                )
                record_path = str(self.store.path)
                if len(record_path) > 240:
                    record_path = "…" + record_path[-239:]
                notice = (
                    f"{count} background updates waiting; context too small to show "
                    f"them; full records: {record_path}"
                )
                self._low_capacity_notice_emitted = True
        else:
            self._low_capacity_notifications = False
            self._low_capacity_notice_emitted = False
        if not selected.more_pending:
            self._edges.discard("notification")
            if selected.prompt_id is not None:
                self._edges.discard("prompt")
        self._active = selected
        return replace(
            request,
            deferred_low_capacity=deferred,
            low_capacity_notice=notice,
        )

    def _compose(
        self,
        staged: StagedDelivery,
        assembled: Sequence[Message],
        steering: ClientSteeringBatch | None,
        *,
        include_notification: bool = True,
        include_prompt: bool = True,
    ) -> PreparedRequest:
        context = self._without_staged(assembled, staged)
        if include_notification and staged.notification_message is not None:
            context.append(staged.notification_message)
        if include_prompt and staged.prompt_message is not None:
            context.append(staged.prompt_message)
        if steering is not None:
            context.extend(steering.messages)
        return PreparedRequest(
            tuple(context),
            steering,
            staged.notification_ids if include_notification else (),
            staged.prompt_id if include_prompt else None,
        )

    @staticmethod
    def _request_active(
        staged: StagedDelivery,
        *,
        notification: bool,
        prompt: bool = True,
    ) -> StagedDelivery:
        return StagedDelivery(
            staged.notification_message if notification else None,
            staged.notification_ids if notification else (),
            staged.prompt_message if prompt else None,
            staged.prompt_id if prompt else None,
            staged.more_pending or bool(staged.notification_ids and not notification),
            DeliveryPublications(
                staged.publications.notifications if notification else (),
                staged.publications.delivered_prompt if prompt else None,
            ),
        )

    @property
    def active_publications(self) -> DeliveryPublications:
        active = self._active
        return active.publications if active is not None else DeliveryPublications()

    async def acknowledge_active(self) -> DeliveryPublications:
        """Durably acknowledge the fixed suffix selected for provider evidence."""

        self._require_owner_loop()
        await self.initialize()
        active = self._active
        if active is None:
            return DeliveryPublications()
        publications = await self.store._to_thread_durable(
            self.store._acknowledge_staged_delivery,
            active.prompt_id,
            active.notification_ids,
        )
        if active.prompt_id is not None and active.prompt_id == self._prompt_priority:
            self._prompt_priority = None
        self._active = None
        self._active_steering = None
        return publications

    def release_active(self) -> None:
        """Re-arm a staged request that ended before provider acknowledgement."""

        self._require_owner_loop()
        staged = self._active
        if staged is None:
            return
        if staged.prompt_id is not None:
            self._edges.add("prompt")
        if staged.notification_ids:
            self._edges.add("notification")
        if self._prompt_priority is not None:
            self._edges.add("prompt")
        self._active = None
        self._active_steering = None

    async def admit_terminal(self, intent: TerminalIntent) -> TerminalResult:
        """Atomically decide whether terminal processing has durable input."""

        self._require_owner_loop()
        await self.initialize()
        if intent not in {
            "segment_end",
            "finish_gate_wait",
            "finish_gate_timeout",
            "foreground_end",
            "idle_wake",
            "finish",
            "nudge",
        }:
            raise ValueError("unknown terminal intent")
        pending, closed = await self.store._to_thread_durable(
            self.store._admit_step_delivery_terminal, intent
        )
        prompt_pending = bool(self.store._active_pending_prompt_entries)
        if self._low_capacity_notifications and not prompt_pending:
            pending = False
        if pending:
            self._edges.update(("prompt", "notification"))
        return TerminalResult(pending, closed)

    def initialize_resident_edges(self) -> None:
        """Seed level state from the already-loaded resident store."""

        if self.store.pending_prompt_queue._pending_unlocked():
            self._edges.add("prompt")
        if any(
            entry.data.get("display_only") is not True
            for entry in self.store._active_notifications.values()
        ):
            self._edges.add("notification")

    def capacity_changed(self) -> None:
        """Recheck deferred notifications after a model or account change."""

        self._require_owner_loop()
        self._low_capacity_notifications = False
        self._low_capacity_notice_emitted = False
        if any(
            entry.data.get("display_only") is not True
            for entry in self.store._active_notifications.values()
        ):
            self._arm_edge("notification")

    @property
    def has_prompt_edge(self) -> bool:
        return "prompt" in self._edges

    @property
    def has_pending_edge(self) -> bool:
        return "prompt" in self._edges or (
            "notification" in self._edges and not self._low_capacity_notifications
        )

    @property
    def active_notification_ids(self) -> frozenset[str]:
        return frozenset(self._active.notification_ids) if self._active else frozenset()

    @property
    def active_prompt_id(self) -> str | None:
        return self._active.prompt_id if self._active else None

    @property
    def prompt_priority(self) -> str | None:
        return self._prompt_priority

    def _require_owner_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("step delivery flow used from a non-owning event loop")

    def _arm_edge(self, source: DeliverySource) -> None:
        self._edges.add(source)
        if self._wake_sink is not None and not (
            source == "notification" and self._low_capacity_notifications
        ):
            self._wake_sink(source)


def _answered_question_notification_id(
    store: ConversationStore, child_instance_id: str, message: str
) -> str | None:
    match = QUESTION_ANSWER_PATTERN.match(message)
    if match is None:
        return None
    question_id = match.group(1)
    return next(
        (
            entry.id
            for entry in store.agent_notifications()
            if entry.data.get("kind") == "child_question"
            and entry.data.get("child_instance_id") == child_instance_id
            and entry.data.get("question_id") == question_id
        ),
        None,
    )


async def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    """Queue one follow-up through the child's resident delivery flow."""

    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    no_live_run = (
        f"no live run/child {child_instance_id!r}; it was canceled, finished, "
        "or never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    if not has_follow_up_loop(
        accepts_follow_ups=marker.get("accepts_follow_ups") is True,
        background=marker.get("background") is True,
    ):
        agent_type = marker.get("agent_type") or "general"
        reason = (
            "it is a foreground child"
            if marker.get("background") is not True
            else "its profile does not accept follow-ups"
        )
        return (
            f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
            f"{reason}"
        )
    try:
        await channel.produce_to(
            child_instance_id,
            PromptInput(
                message,
                MessageOrigin.AGENT_SEND,
                time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS,
            ),
        )
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except ValueError as exc:
        if str(exc) != "session store is closed":
            raise
        return no_live_run
    except (LookupError, PendingPromptsClosedError):
        return no_live_run
    match = QUESTION_ANSWER_PATTERN.match(message)
    if match is not None:
        await channel.produce_to_parent_of(
            child_instance_id,
            NotificationInput(
                "question_answer_ack",
                {
                    "child_instance_id": child_instance_id,
                    "question_id": match.group(1),
                },
            ),
        )
    return None
