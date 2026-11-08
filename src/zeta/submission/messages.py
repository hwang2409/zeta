"""Messages the submission pipeline consumer handles, one per transition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.approval import ApprovalDecision
from ..model_input import ModelInputEnvelope
from ..protocol.types import Message, ToolCall
from .model import Submission
from .notification_retry import NotificationOutcome


class _Message(Protocol):  # noqa: PYI046 - the pipeline queue item type
    pass


@dataclass(frozen=True, slots=True)
class _Submit:
    submission: Submission
    acknowledged: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _Retry:
    submission: Submission
    user_message: Message
    acknowledged: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _PreprocessingDone:
    submission: Submission
    model_input: ModelInputEnvelope | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _CommandDone:
    submission: Submission
    receipt: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _SlashDone:
    submission: Submission
    task: asyncio.Task[Any]


@dataclass(frozen=True, slots=True)
class _ProviderDone:
    submission: Submission
    task: asyncio.Task[NotificationOutcome]


@dataclass(frozen=True, slots=True)
class _DurableToolDone:
    request_id: str
    task: asyncio.Task[Any]


@dataclass(frozen=True, slots=True)
class _ApprovalLifecycle:
    submission: Submission
    kind: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class _ApprovalAction:
    decision: ApprovalDecision
    requested_key: str | None
    acknowledged: asyncio.Future[None] | None = None
    always: bool = False


@dataclass(frozen=True, slots=True)
class _AbortAction:
    submission_id: int | None


@dataclass(frozen=True, slots=True)
class _UndoAction:
    pass
