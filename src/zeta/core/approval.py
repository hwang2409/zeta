"""Durable tool approval decisions."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..types import Message, MessageRole, ToolCall, ToolResult, ToolUseContent
from .abort import AbortSignal
from .store import ConversationStore


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ApprovalRule:
    """One allow/deny/ask entry: a tool name with an optional subject glob.

    ``read`` matches every ``read`` call. ``bash(git status*)`` matches a
    ``bash`` call whose declared subject argument (``command`` for ``bash``)
    satisfies ``fnmatch.fnmatchcase`` against the pattern. That is the only
    matching mode: a misread pattern is an unintended auto-approval, so
    predictability beats expressiveness here.
    """

    tool: str
    pattern: str | None = None

    def __str__(self) -> str:
        if self.pattern is None:
            return self.tool
        return f"{self.tool}({self.pattern})"


def parse_approval_rule(text: str) -> ApprovalRule:
    """Parse ``tool`` or ``tool(pattern)``; anything else is a ``ValueError``.

    The tool name may not be empty or contain whitespace or parentheses, and
    the pattern may not be empty. The pattern is taken verbatim.
    """

    if type(text) is not str:
        raise ValueError(
            f"invalid approval rule {text!r}: expected a string, "
            f"got {type(text).__name__}"
        )
    if "(" not in text:
        tool, pattern = text, None
    elif text.endswith(")"):
        tool, pattern = text[:-1].split("(", 1)
        if not pattern:
            raise ValueError(
                f"invalid approval rule {text!r}: empty argument pattern"
            )
    else:
        raise ValueError(
            f"invalid approval rule {text!r}: missing closing parenthesis; "
            "expected 'tool' or 'tool(pattern)'"
        )
    if not tool or "(" in tool or ")" in tool or any(ch.isspace() for ch in tool):
        raise ValueError(
            f"invalid approval rule {text!r}: tool name must be nonempty "
            "with no whitespace or parentheses"
        )
    return ApprovalRule(tool, pattern)


def _rule_set(rules: Iterable[str | ApprovalRule]) -> frozenset[ApprovalRule]:
    return frozenset(
        rule if isinstance(rule, ApprovalRule) else parse_approval_rule(rule)
        for rule in rules
    )


def _without_scoped(
    rules: frozenset[ApprovalRule], tool: str
) -> tuple[frozenset[ApprovalRule], list[ApprovalRule]]:
    dropped = [rule for rule in rules if rule.tool == tool and rule.pattern is not None]
    return rules - frozenset(dropped), dropped


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_call: ToolCall
    label: str | None = None
    child_instance_id: str | None = None

    @property
    def key(self) -> str | tuple[str, str]:
        if self.child_instance_id is None:
            return self.request_id
        return self.child_instance_id, self.request_id


class _AbortSignal(Protocol):
    def is_set(self) -> bool: ...

    async def wait(self) -> None: ...


class ApprovalPolicy:
    """Choose, persist, and resolve decisions for tool calls."""

    def __init__(
        self,
        *,
        always_allow: Iterable[str | ApprovalRule] = (),
        always_deny: Iterable[str | ApprovalRule] = (),
        always_ask: Iterable[str | ApprovalRule] = (),
        default: ApprovalDecision | str = ApprovalDecision.ASK,
        store: ConversationStore | None = None,
    ) -> None:
        self.always_allow = always_allow
        self.always_deny = always_deny
        self.always_ask = always_ask
        self.default = _decision(default)
        self._subjects: dict[str, str] = {}
        self._notices: list[str] = []
        self._store = store
        self._delegated: dict[
            tuple[str, str], tuple[ApprovalRequest, ConversationStore]
        ] = {}
        self._ephemeral: dict[str, tuple[ApprovalRequest, str | None]] = {}

    def bind_store(self, store: ConversationStore) -> None:
        self._store = store

    # The three rule sets accept rule text or parsed rules and always hold
    # parsed rules, so ``policy.always_ask = frozenset()`` keeps neutralising
    # every ask rule, bare or argument-scoped (headless relies on this).

    @property
    def always_allow(self) -> frozenset[ApprovalRule]:
        return self._always_allow

    @always_allow.setter
    def always_allow(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._always_allow = _rule_set(rules)

    @property
    def always_deny(self) -> frozenset[ApprovalRule]:
        return self._always_deny

    @always_deny.setter
    def always_deny(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._always_deny = _rule_set(rules)

    @property
    def always_ask(self) -> frozenset[ApprovalRule]:
        return self._always_ask

    @always_ask.setter
    def always_ask(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._always_ask = _rule_set(rules)

    @property
    def notices(self) -> tuple[str, ...]:
        """Loud configuration notices: rules dropped because they cannot apply."""

        return tuple(self._notices)

    def declare_subjects(
        self, subjects: Mapping[str, str | None]
    ) -> tuple[str, ...]:
        """Record which argument scopes each tool; returns new notices.

        The registry calls this for every tool it holds. A tool that declares
        no subject cannot be scoped by arguments, so an argument-scoped rule
        naming it is a configuration error: the rule is dropped from every
        tier and reported, never silently widened to a bare-name match.
        Declarations merge, so a child registry pushing a subset of its
        parent's tools cannot erase what the parent declared.
        """

        dropped: list[ApprovalRule] = []
        for tool, subject in subjects.items():
            if subject is not None:
                self._subjects[tool] = subject
                continue
            self._subjects.pop(tool, None)
            self._always_deny, removed = _without_scoped(self._always_deny, tool)
            dropped.extend(removed)
            self._always_ask, removed = _without_scoped(self._always_ask, tool)
            dropped.extend(removed)
            self._always_allow, removed = _without_scoped(self._always_allow, tool)
            dropped.extend(removed)
        notices = tuple(
            f"approval · dropped rule '{rule}': tool '{rule.tool}' "
            "declares no approval subject, so it cannot be scoped by arguments"
            for rule in sorted(dropped, key=str)
        )
        self._notices.extend(notices)
        return notices

    def decide(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ApprovalDecision:
        if self._matches(self._always_deny, tool_name, arguments, unreadable=True):
            return ApprovalDecision.DENY
        if self._matches(self._always_ask, tool_name, arguments, unreadable=True):
            return ApprovalDecision.ASK
        if self._matches(self._always_allow, tool_name, arguments, unreadable=False):
            return ApprovalDecision.ALLOW
        return self.default

    def _matches(
        self,
        rules: frozenset[ApprovalRule],
        tool_name: str,
        arguments: object,
        *,
        unreadable: bool,
    ) -> bool:
        """Report whether any rule in one tier fires for this call.

        A bare rule fires on the name alone. A scoped rule needs the tool's
        declared subject: with none declared it is inert. When the subject
        argument is missing or not a string the rule resolves to its safe
        side, given by ``unreadable``: an allow rule does not fire, while a
        deny or ask rule does (fail closed).
        """

        for rule in rules:
            if rule.tool != tool_name:
                continue
            if rule.pattern is None:
                return True
            subject = self._subjects.get(tool_name)
            if subject is None:
                continue
            value = arguments.get(subject) if isinstance(arguments, Mapping) else None
            if not isinstance(value, str):
                if unreadable:
                    return True
                continue
            if fnmatch.fnmatchcase(value, rule.pattern):
                return True
        return False

    def pending_requests(self) -> list[ApprovalRequest]:
        store = self._require_store()
        requests = [
            ApprovalRequest(request_id, tool_call)
            for request_id, tool_call in store.pending_approvals()
        ]
        for (child_id, request_id), (request, delegated_store) in list(
            self._delegated.items()
        ):
            state = delegated_store.approval_states().get(request_id)
            if state is None or state[1] is not None:
                self._delegated.pop((child_id, request_id), None)
            else:
                requests.append(request)
        requests.extend(
            request
            for request, decision in self._ephemeral.values()
            if decision is None
        )
        return requests

    def register_delegated(
        self,
        request: ApprovalRequest,
        store: ConversationStore,
        *,
        child_instance_id: str | None = None,
    ) -> None:
        """Expose a child request without writing it into this store."""

        child_id = child_instance_id or request.child_instance_id or store.session_id
        key = (child_id, request.request_id)
        if key in self._delegated:
            return
        self._delegated[key] = (
            ApprovalRequest(
                request.request_id,
                request.tool_call,
                request.label,
                child_id,
            ),
            store,
        )

    def cleanup_delegated(self, child_instance_id: str) -> None:
        """Resolve and remove all delegated requests owned by one child."""

        for key, (request, store) in list(self._delegated.items()):
            if key[0] != child_instance_id:
                continue
            try:
                store.resolve_approval(request.request_id, ApprovalDecision.DENY.value)
            except ValueError:
                pass
            finally:
                self._delegated.pop(key, None)

    def approve(
        self,
        request_id: str | tuple[str, str],
        *,
        child_instance_id: str | None = None,
    ) -> bool:
        return self.resolve(
            request_id,
            ApprovalDecision.ALLOW,
            child_instance_id=child_instance_id,
        )

    def deny(
        self,
        request_id: str | tuple[str, str],
        *,
        child_instance_id: str | None = None,
    ) -> bool:
        return self.resolve(
            request_id,
            ApprovalDecision.DENY,
            child_instance_id=child_instance_id,
        )

    def abort(
        self,
        request_id: str | tuple[str, str],
        *,
        child_instance_id: str | None = None,
    ) -> bool:
        delegated = self._delegated_entry(request_id, child_instance_id)
        if delegated is not None:
            return delegated[1].resolve_approval(
                delegated[0].request_id, "abort"
            )
        if isinstance(request_id, tuple):
            return False
        ephemeral = self._ephemeral.get(request_id)
        if ephemeral is not None:
            self._ephemeral[request_id] = (ephemeral[0], "abort")
            return True
        store = self._require_store()
        return store.resolve_approval(request_id, "abort")

    def abort_or_winner(
        self,
        request_id: str | tuple[str, str],
        *,
        child_instance_id: str | None = None,
    ) -> ApprovalDecision | None:
        delegated = self._delegated_entry(request_id, child_instance_id)
        if delegated is not None:
            delegated[1].resolve_approval(delegated[0].request_id, "abort")
            state = delegated[1].approval_states().get(delegated[0].request_id)
            return _resolved_decision(state[1] if state is not None else None)
        if isinstance(request_id, tuple):
            return None
        ephemeral = self._ephemeral.get(request_id)
        if ephemeral is not None:
            self._ephemeral[request_id] = (ephemeral[0], "abort")
            return None
        store = self._require_store()
        store.resolve_approval(request_id, "abort")
        state = store.approval_states().get(request_id)
        return _resolved_decision(state[1] if state is not None else None)

    def durable_decision(
        self,
        request_id: str | tuple[str, str],
        *,
        child_instance_id: str | None = None,
    ) -> str | None:
        delegated = self._delegated_entry(request_id, child_instance_id)
        if delegated is not None:
            state = delegated[1].approval_states().get(delegated[0].request_id)
            return None if state is None else state[1]
        if isinstance(request_id, tuple):
            return None
        state = self._require_store().approval_states().get(request_id)
        return None if state is None else state[1]

    def resolve(
        self,
        request_id: str | tuple[str, str],
        decision: ApprovalDecision | str,
        *,
        child_instance_id: str | None = None,
    ) -> bool:
        resolved = _decision(decision)
        if resolved is ApprovalDecision.ASK:
            raise ValueError("approval resolution must allow or deny")
        delegated = self._delegated_entry(request_id, child_instance_id)
        if delegated is not None:
            return delegated[1].resolve_approval(
                delegated[0].request_id, resolved.value
            )
        if isinstance(request_id, tuple):
            return False
        ephemeral = self._ephemeral.get(request_id)
        if ephemeral is not None:
            self._ephemeral[request_id] = (ephemeral[0], resolved.value)
            return True
        store = self._require_store()
        return store.resolve_approval(request_id, resolved.value)

    def _delegated_entry(
        self,
        request_id: str | tuple[str, str],
        child_instance_id: str | None,
    ) -> tuple[ApprovalRequest, ConversationStore] | None:
        if isinstance(request_id, tuple):
            return self._delegated.get(request_id)
        if child_instance_id is not None:
            return self._delegated.get((child_instance_id, request_id))
        matches = [
            delegated
            for (child_id, delegated_id), delegated in self._delegated.items()
            if delegated_id == request_id
        ]
        return matches[0] if len(matches) == 1 else None

    def prepare(self, tool_call: ToolCall) -> ApprovalRequest | None:
        """Build an ask request for atomic persistence with its assistant anchor."""

        store = self._require_store()
        state = store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(tool_call.name, tool_call.arguments) is ApprovalDecision.ASK:
            return ApprovalRequest(tool_call.id, tool_call)
        return None

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: _AbortSignal,
        *,
        persist_request: bool = True,
    ) -> ApprovalDecision | None:
        store = self._require_store()
        state = store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            if state[1] == ApprovalDecision.ALLOW.value:
                return ApprovalDecision.ALLOW
            if state[1] == ApprovalDecision.DENY.value:
                return ApprovalDecision.DENY
        else:
            decision = self.decide(tool_call.name, tool_call.arguments)
            if decision is not ApprovalDecision.ASK:
                return decision
            if persist_request:
                store.append_message_with_approval_requests(
                    Message(MessageRole.ASSISTANT, [ToolUseContent(tool_call)]),
                    [(tool_call.id, tool_call)],
                )
            else:
                self._ephemeral[tool_call.id] = (
                    ApprovalRequest(tool_call.id, tool_call),
                    None,
                )

        while True:
            ephemeral = self._ephemeral.get(tool_call.id)
            if ephemeral is not None and ephemeral[1] is not None:
                if ephemeral[1] == ApprovalDecision.ALLOW.value:
                    return ApprovalDecision.ALLOW
                if ephemeral[1] == ApprovalDecision.DENY.value:
                    return ApprovalDecision.DENY
                return None
            state = store.approval_states().get(tool_call.id)
            if state is not None and state[1] is not None:
                if state[1] == ApprovalDecision.ALLOW.value:
                    return ApprovalDecision.ALLOW
                if state[1] == ApprovalDecision.DENY.value:
                    return ApprovalDecision.DENY
                return None
            if abort_signal.is_set():
                return self._resolve_abort_or_winner(tool_call.id)
            abort_task = asyncio.create_task(abort_signal.wait())
            poll_task = asyncio.create_task(asyncio.sleep(0.05))
            try:
                done, pending = await asyncio.wait(
                    {abort_task, poll_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                abort_task.cancel()
                poll_task.cancel()
                await asyncio.gather(abort_task, poll_task, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if abort_task in done:
                return self._resolve_abort_or_winner(tool_call.id)

    def forget_ephemeral(self, request_id: str) -> None:
        """Remove one non-durable approval after its owner finishes."""

        self._ephemeral.pop(request_id, None)

    def _resolve_abort_or_winner(
        self,
        request_id: str,
    ) -> ApprovalDecision | None:
        return self.abort_or_winner(request_id)

    def _require_store(self) -> ConversationStore:
        if self._store is None:
            raise RuntimeError("approval policy requires a conversation store")
        return self._store


def _decision(value: ApprovalDecision | str) -> ApprovalDecision:
    try:
        return ApprovalDecision(value)
    except ValueError as exc:
        raise ValueError(f"invalid approval decision: {value}") from exc


def _resolved_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None


ApprovalHook = Callable[
    [str, dict[str, object]], bool | str | Awaitable[bool | str] | None
]
AdvanceGeneration = Callable[[AbortSignal], AbortSignal]


def canceled_result(tool_call_id: str) -> ToolResult:
    return ToolResult(tool_call_id, "tool execution canceled", True, is_canceled=True)


@dataclass(slots=True)
class ApprovalGate:
    """Run durable approval and the optional pre-execution hook."""

    policy: ApprovalPolicy | None
    hook: ApprovalHook | None

    async def run(
        self,
        tool_call: ToolCall,
        arguments: dict[str, object],
        signal: AbortSignal,
        advance_generation: AdvanceGeneration,
        lifecycle: Callable[[str], None] | None = None,
        *,
        skip_approval: bool = False,
        persist_request: bool = True,
    ) -> tuple[ToolResult | None, AbortSignal]:
        execution_signal = signal
        if self.policy is not None and not skip_approval:
            approval_started = (
                self.policy.durable_decision(tool_call.id) is None
                and self.policy.decide(tool_call.name, arguments)
                is ApprovalDecision.ASK
            )
            if approval_started and lifecycle is not None:
                lifecycle("approval_start")
            try:
                if persist_request:
                    decision = await self.policy.authorize(tool_call, signal)
                else:
                    decision = await self.policy.authorize(
                        tool_call, signal, persist_request=False
                    )
            except Exception as exc:  # noqa: BLE001 - report approval failures
                return ToolResult(tool_call.id, f"approval failed: {exc}", True), execution_signal
            finally:
                if approval_started and lifecycle is not None:
                    lifecycle("approval_end")
            if decision is None:
                return canceled_result(tool_call.id), execution_signal
            if signal.is_set():
                durable_decision = self.policy.durable_decision(tool_call.id)
                if durable_decision == ApprovalDecision.DENY.value:
                    return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
                if durable_decision != ApprovalDecision.ALLOW.value:
                    return canceled_result(tool_call.id), execution_signal
                execution_signal = advance_generation(signal)
                if execution_signal.is_set():
                    return canceled_result(tool_call.id), execution_signal
            if decision is ApprovalDecision.DENY:
                return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
        if self.hook is None:
            if signal.is_set() and execution_signal is signal:
                return canceled_result(tool_call.id), execution_signal
            return None, execution_signal
        try:
            allowed = self.hook(tool_call.name, arguments)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        except Exception as exc:  # noqa: BLE001 - report hook failures
            return ToolResult(tool_call.id, f"pre-execution hook failed: {exc}", True), execution_signal
        if isinstance(allowed, str):
            return ToolResult(tool_call.id, allowed, True), execution_signal
        if allowed is False:
            return ToolResult(tool_call.id, "tool execution denied by hook", True), execution_signal
        if signal.is_set() and execution_signal is signal:
            return canceled_result(tool_call.id), execution_signal
        return None, execution_signal
