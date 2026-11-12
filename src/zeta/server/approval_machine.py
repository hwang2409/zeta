"""Value types for the server approval state machine.

The event dataclasses, the immutable machine state, and the small display-fact
helper are separated from the lifecycle engine so the engine module stays within
the module-size limit. These types carry no behavior beyond their data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..core.approval import ApprovalRequest
from ..core.store import ConversationStore
from ..protocol.types import StreamEvent
from .approval_settlement.approval_grants import ApprovalRecord
from .protocol import ApprovalDecision, EventDescriptor

ApprovalKey = str | tuple[str, str]


@dataclass(frozen=True, slots=True)
class SessionHealth:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ApprovalMachineState:
    """Immutable live approval projection for one loaded serve session."""

    generation: int = 0
    records: Mapping[ApprovalKey, ApprovalRecord] = MappingProxyType({})
    health: SessionHealth | None = None


@dataclass(frozen=True, slots=True)
class Register:
    request: ApprovalRequest
    store: ConversationStore
    child_instance_id: str


@dataclass(frozen=True, slots=True)
class Start:
    stream_event: StreamEvent
    session_id: str | None
    wire_event: EventDescriptor


@dataclass(frozen=True, slots=True)
class Decide:
    method: ApprovalDecision
    request_id: str
    scope: str = "once"


@dataclass(frozen=True, slots=True)
class ExternalEnd:
    key: ApprovalKey
    session_id: str | None
    data: Mapping[str, object]
    suppress_write_errors: bool = False


@dataclass(frozen=True, slots=True)
class Cancel:
    request_id: str
    child_instance_id: str | None = None
    reason: str = "turn_canceled"


@dataclass(frozen=True, slots=True)
class CleanupChild:
    child_instance_id: str


@dataclass(frozen=True, slots=True)
class Terminate:
    foreground_only: bool = False
    suppress_write_errors: bool = False
    reason: str = "controller_disconnected"


@dataclass(frozen=True, slots=True)
class BeginGrace:
    pass


@dataclass(frozen=True, slots=True)
class Reclaim:
    pass


@dataclass(frozen=True, slots=True)
class Ready:
    pass


@dataclass(frozen=True, slots=True)
class Expire:
    pass


@dataclass(frozen=True, slots=True)
class Replace:
    pass


@dataclass(frozen=True, slots=True)
class Close:
    pass


ApprovalEvent = (
    Register
    | Start
    | Decide
    | ExternalEnd
    | Cancel
    | CleanupChild
    | Terminate
    | BeginGrace
    | Reclaim
    | Ready
    | Expire
    | Replace
    | Close
)


class _Completed:
    def __init__(self, value: object = None) -> None:
        self._value = value

    def __await__(self):
        async def result() -> object:
            return self._value

        return result().__await__()


def approval_display_fields(request: ApprovalRequest) -> dict[str, object]:
    """Return immutable trusted display facts for an approval request."""

    fields: dict[str, object] = {}
    if request.action is not None:
        fields["approval_action"] = request.action
    display = request.audit_display()
    if display:
        fields["approval_display"] = display
    return fields

