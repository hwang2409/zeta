"""Durable tool approval decisions."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import stat
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..path_identity import same_physical_path
from ..protocol.types import (
    ASSISTANT_RESPONSE_SYNTHETIC,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from .abort import AbortSignal
from .store import ConversationStore


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ApprovedPathAbsent:
    """Approval-time proof that the target did not exist."""


@dataclass(frozen=True, slots=True)
class ApprovedPathExisting:
    """Approval-time identity of an existing target."""

    identity: tuple[int, int]


ApprovedPathState = ApprovedPathAbsent | ApprovedPathExisting


@dataclass(frozen=True, slots=True)
class ApprovedPathExecution:
    """Canonical target and stable directory root approved for a child call."""

    target: str
    root: str
    root_identity: tuple[int, int]
    target_state: ApprovedPathState


@dataclass(frozen=True, slots=True)
class ApprovedCwdExecution:
    """Canonical working directory approved for a child shell call."""

    cwd: str
    identity: tuple[int, int]


ApprovedExecution = ApprovedPathExecution | ApprovedCwdExecution


@dataclass(frozen=True, slots=True)
class ApprovalRule:
    """One allow/deny/ask entry with optional action and subject scope.

    ``read`` matches every ``read`` call. ``bash(git status*)`` matches a
    ``bash`` call whose declared subject argument (``command`` for ``bash``)
    satisfies ``fnmatch.fnmatchcase`` against the pattern. That is the only
    matching mode: a misread pattern is an unintended auto-approval, so
    predictability beats expressiveness here.
    """

    tool: str
    subject_pattern: str | None = None
    action: str | None = None

    @property
    def pattern(self) -> str | None:
        """Compatibility spelling for the subject glob."""

        return self.subject_pattern

    def __str__(self) -> str:
        if self.action is not None:
            scope = self.action
            if self.subject_pattern is not None:
                scope += f" {self.subject_pattern}"
            return f"{self.tool}({scope})"
        if self.subject_pattern is None:
            return self.tool
        return f"{self.tool}({self.subject_pattern})"


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
            raise ValueError(f"invalid approval rule {text!r}: empty argument pattern")
        if pattern.endswith(" ") and "(" not in pattern and ")" not in pattern:
            raise ValueError(f"invalid approval rule {text!r}: empty subject pattern")
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


def _canonical_path_pattern(pattern: str, parent_cwd: str) -> str:
    """Resolve a path glob's non-magic prefix without changing glob semantics."""

    expanded = os.path.expanduser(pattern)
    if not os.path.isabs(expanded):
        expanded = os.path.join(parent_cwd, expanded)
    magic = min(
        (expanded.find(character) for character in "*?[" if character in expanded),
        default=len(expanded),
    )
    prefix, suffix = expanded[:magic], expanded[magic:]
    trailing_separator = prefix.endswith(os.sep)
    canonical = os.path.realpath(prefix)
    if trailing_separator and not canonical.endswith(os.sep):
        canonical += os.sep
    return canonical + suffix


def _bind_approved_path(target: str) -> ApprovedPathExecution | None:
    """Capture the deepest existing canonical parent as the execution root."""

    root = os.path.dirname(target)
    while True:
        try:
            root_stat = os.stat(root, follow_symlinks=False)
        except FileNotFoundError:
            parent = os.path.dirname(root)
            if parent == root:
                return None
            root = parent
            continue
        except OSError:
            return None
        if not stat.S_ISDIR(root_stat.st_mode):
            return None
        try:
            target_stat = os.stat(target, follow_symlinks=False)
            target_state: ApprovedPathState = ApprovedPathExisting(
                (target_stat.st_dev, target_stat.st_ino)
            )
        except FileNotFoundError:
            target_state = ApprovedPathAbsent()
        except OSError:
            return None
        return ApprovedPathExecution(
            target=target,
            root=root,
            root_identity=(root_stat.st_dev, root_stat.st_ino),
            target_state=target_state,
        )


def _bind_approved_cwd(cwd: str) -> ApprovedCwdExecution | None:
    try:
        cwd_stat = os.stat(cwd, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISDIR(cwd_stat.st_mode):
        return None
    return ApprovedCwdExecution(cwd, (cwd_stat.st_dev, cwd_stat.st_ino))


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_call: ToolCall
    label: str | None = None
    child_instance_id: str | None = None
    effective_cwd: str | None = None
    resolved_path: str | None = None
    action: str | None = None
    capability_resolved: bool = True

    @property
    def key(self) -> str | tuple[str, str]:
        if self.child_instance_id is None:
            return self.request_id
        return self.child_instance_id, self.request_id

    def audit_display(self) -> dict[str, object]:
        """Return immutable presentation facts, never executable authority."""

        display: dict[str, object] = {}
        if self.effective_cwd is not None or self.resolved_path is not None:
            display.update(
                {
                    "effective_cwd": self.effective_cwd,
                    "resolved_path": self.resolved_path,
                }
            )
        return display

    def audit_facts(self) -> dict[str, object]:
        """Return registry-resolved authorization facts for durable audit."""

        return {"action": self.action}

    def audit_record(
        self,
    ) -> tuple[str, ToolCall, dict[str, object], dict[str, object]]:
        """Return the validated store input for this approval request."""

        return self.request_id, self.tool_call, self.audit_facts(), self.audit_display()

    def always_allow_rule(self) -> ApprovalRule:
        """Return the narrow persistent grant offered for this request."""

        if not self.capability_resolved:
            raise ValueError(
                f"cannot resolve persisted tool call: {self.tool_call.name}"
            )
        return ApprovalRule(self.tool_call.name, action=self.action)


class ApprovalCapability(Protocol):
    """Resolved authorization facts supplied by the tool registry."""

    tool: str
    action: str | None
    subject_field: str | None
    subject_value: object
    binding: object
    arguments: Mapping[str, object]


class _AbortSignal(Protocol):
    def is_set(self) -> bool: ...

    async def wait(self) -> None: ...


class ApprovalAbortPolicy(Protocol):
    """Resolve an abort against any decision that won concurrently."""

    def abort_or_winner(
        self,
        request_id: str,
        *,
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> ApprovalDecision | None: ...


class ApprovalPolicy(ApprovalAbortPolicy):
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
        self._rule_resolver: Callable[[str | ApprovalRule], ApprovalRule] | None = None
        self._rule_sources: dict[str, tuple[str | ApprovalRule, ...]] = {}
        self._notices: list[str] = []
        self.always_allow = always_allow
        self.always_deny = always_deny
        self.always_ask = always_ask
        self.default = _decision(default)
        self._store = store
        self._delegated: dict[
            tuple[str, str], tuple[ApprovalRequest, ConversationStore]
        ] = {}
        self._ephemeral: dict[str, tuple[ApprovalRequest, str | None]] = {}
        self._display_resolver: Callable[[ApprovalRequest], ApprovalRequest] | None = (
            None
        )
        self._capability_resolver: Callable[[ToolCall], ApprovalCapability] | None = None

    def bind_display_resolver(
        self, resolver: Callable[[ApprovalRequest], ApprovalRequest] | None
    ) -> None:
        self._display_resolver = resolver

    def bind_store(self, store: ConversationStore) -> None:
        self._store = store

    def bind_capability_resolver(
        self, resolver: Callable[[ToolCall], ApprovalCapability]
    ) -> None:
        """Use registry resolution when durable calls become approval requests."""

        self._capability_resolver = resolver

    def remember_allow(self, request: ApprovalRequest) -> ApprovalRule:
        """Persist and return the exact resolved capability grant."""

        rule = request.always_allow_rule()
        if self._rule_resolver is not None:
            rule = self._rule_resolver(rule)
        self.always_allow = self.always_allow | {rule}
        if rule not in self.always_allow:
            raise ValueError(f"approval rule was not accepted: {rule}")
        return rule

    # The three rule sets accept rule text or parsed rules and always hold
    # parsed rules, so ``policy.always_ask = frozenset()`` keeps neutralising
    # every ask rule, bare or argument-scoped (headless relies on this).

    def _set_rules(
        self, tier: str, rules: Iterable[str | ApprovalRule]
    ) -> None:
        source = tuple(rules)
        parsed = tuple(
            item if isinstance(item, ApprovalRule) else parse_approval_rule(item)
            for item in source
        )
        self._rule_sources[tier] = source
        resolved: set[ApprovalRule] = set()
        for item, rule in zip(source, parsed, strict=True):
            try:
                if self._rule_resolver is not None:
                    rule = self._rule_resolver(rule)
            except ValueError as exc:
                notice = f"approval · dropped rule {item!r}: {exc}"
                if notice not in self._notices:
                    self._notices.append(notice)
                continue
            resolved.add(rule)
        setattr(self, f"_{tier}", frozenset(resolved))

    def bind_rule_resolver(
        self, resolver: Callable[[str | ApprovalRule], ApprovalRule]
    ) -> None:
        """Resolve every configured and persisted rule through one registry seam."""

        self._rule_resolver = resolver
        for tier in ("always_deny", "always_ask", "always_allow"):
            self._set_rules(tier, self._rule_sources.get(tier, ()))

    @property
    def always_allow(self) -> frozenset[ApprovalRule]:
        return self._always_allow

    @always_allow.setter
    def always_allow(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._set_rules("always_allow", rules)

    @property
    def always_deny(self) -> frozenset[ApprovalRule]:
        return self._always_deny

    @always_deny.setter
    def always_deny(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._set_rules("always_deny", rules)

    @property
    def always_ask(self) -> frozenset[ApprovalRule]:
        return self._always_ask

    @always_ask.setter
    def always_ask(self, rules: Iterable[str | ApprovalRule]) -> None:
        self._set_rules("always_ask", rules)

    @property
    def notices(self) -> tuple[str, ...]:
        """Loud configuration notices for rejected rules."""

        return tuple(self._notices)

    def decide(self, capability: ApprovalCapability) -> ApprovalDecision:
        """Match one registry-resolved capability against parsed rules."""

        if self._matches(self._always_deny, capability, unreadable=True):
            return ApprovalDecision.DENY
        if self._matches(self._always_ask, capability, unreadable=True):
            return ApprovalDecision.ASK
        if self._matches(self._always_allow, capability, unreadable=False):
            return ApprovalDecision.ALLOW
        return self.default

    def decide_for_child(
        self,
        capability: ApprovalCapability,
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> ApprovalDecision:
        decision, _binding = self.decide_for_child_with_binding(
            capability, parent_cwd=parent_cwd, child_cwd=child_cwd
        )
        return decision

    def capture_child_binding(
        self,
        capability: ApprovalCapability,
        *,
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovedExecution | None, bool]:
        """Capture the object a human approval is about before display."""

        binding = str(capability.binding)
        if binding == "path":
            value = capability.subject_value
            if not isinstance(value, str):
                return None, True
            candidate = os.path.expanduser(value)
            if not os.path.isabs(candidate):
                candidate = os.path.join(os.fspath(child_cwd), candidate)
            return _bind_approved_path(os.path.realpath(candidate)), True
        if binding == "cwd":
            return _bind_approved_cwd(os.path.realpath(os.fspath(child_cwd))), True
        return None, False

    def decide_for_child_with_binding(
        self,
        capability: ApprovalCapability,
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        """Match a delegated call and bind scoped allows to canonical objects."""

        binding_kind = str(capability.binding)
        if binding_kind == "path":
            value = capability.subject_value
            resolved = (
                os.path.realpath(
                    os.path.join(os.fspath(child_cwd), os.path.expanduser(value))
                )
                if isinstance(value, str)
                else None
            )
            tiers = (
                (self._always_deny, ApprovalDecision.DENY, True),
                (self._always_ask, ApprovalDecision.ASK, True),
                (self._always_allow, ApprovalDecision.ALLOW, False),
            )
            for rules, decision, unreadable in tiers:
                rule = self._matching_child_path_rule(
                    rules,
                    capability,
                    resolved,
                    parent_cwd=os.fspath(parent_cwd),
                    unreadable=unreadable,
                )
                if rule is None:
                    continue
                binding = None
                if (
                    decision is ApprovalDecision.ALLOW
                    and rule.pattern is not None
                    and resolved is not None
                ):
                    binding = _bind_approved_path(resolved)
                    if binding is None:
                        return ApprovalDecision.ASK, None
                return decision, binding
            return self.default, None

        if binding_kind == "cwd":
            if self._matches(self._always_deny, capability, unreadable=True):
                return ApprovalDecision.DENY, None
            if self._matches(self._always_ask, capability, unreadable=True):
                return ApprovalDecision.ASK, None
            same_cwd = same_physical_path(parent_cwd, child_cwd)
            matching_rules = [
                rule
                for rule in self._always_allow
                if (rule.tool != capability.tool or rule.pattern is None or same_cwd)
                and self._matches(
                    frozenset({rule}), capability, unreadable=False
                )
            ]
            if matching_rules:
                if any(rule.pattern is None for rule in matching_rules):
                    return ApprovalDecision.ALLOW, None
                binding = _bind_approved_cwd(os.path.realpath(child_cwd))
                if binding is None:
                    return ApprovalDecision.ASK, None
                return ApprovalDecision.ALLOW, binding
            return self.default, None

        return self.decide(capability), None

    @staticmethod
    def _matching_child_path_rule(
        rules: frozenset[ApprovalRule],
        capability: ApprovalCapability,
        resolved_path: str | None,
        *,
        parent_cwd: str,
        unreadable: bool,
    ) -> ApprovalRule | None:
        for rule in sorted(rules, key=lambda candidate: candidate.pattern is not None):
            if rule.tool != capability.tool or (
                rule.action is not None and rule.action != capability.action
            ):
                continue
            if rule.pattern is None:
                return rule
            if resolved_path is None:
                if unreadable:
                    return rule
                continue
            pattern = _canonical_path_pattern(rule.pattern, parent_cwd)
            if fnmatch.fnmatchcase(resolved_path, pattern):
                return rule
        return None

    @staticmethod
    def _matches(
        rules: frozenset[ApprovalRule],
        capability: ApprovalCapability,
        *,
        unreadable: bool,
    ) -> bool:
        """Report whether one rule tier matches resolved authorization facts."""

        for rule in rules:
            if rule.tool != capability.tool or (
                rule.action is not None and rule.action != capability.action
            ):
                continue
            if rule.pattern is None:
                return True
            value = capability.subject_value
            if not isinstance(value, str):
                if unreadable:
                    return True
                continue
            if fnmatch.fnmatchcase(value, rule.pattern):
                return True
        return False

    def pending_requests(self) -> list[ApprovalRequest]:
        """Return strict pending state for decisions and server actions."""

        def strict_pending(store: ConversationStore) -> dict[str, ToolCall]:
            return {
                request_id: tool_call
                for request_id, (tool_call, decision) in store.approval_states().items()
                if decision is None
            }

        return self._collect_pending_requests(strict_pending)

    def pending_requests_for_display(self) -> list[ApprovalRequest]:
        """Return latency-tolerant pending state for TUI display and key filters."""

        # Display and key-filter callers only; decisions use pending_requests().
        return self._collect_pending_requests(
            lambda store: dict(store.pending_approvals())
        )

    def _collect_pending_requests(
        self,
        pending_for_store: Callable[[ConversationStore], dict[str, ToolCall]],
    ) -> list[ApprovalRequest]:
        store = self._require_store()
        requests: list[ApprovalRequest] = []
        for request_id, tool_call in pending_for_store(store).items():
            action = None
            capability_resolved = self._capability_resolver is None
            if self._capability_resolver is not None:
                try:
                    action = self._capability_resolver(tool_call).action
                    capability_resolved = True
                except (KeyError, TypeError, ValueError):
                    capability_resolved = False
            request = ApprovalRequest(
                request_id,
                tool_call,
                action=action,
                capability_resolved=capability_resolved,
            )
            requests.append(
                (self._display_resolver or (lambda current: current))(request)
            )
        delegated_pending: dict[ConversationStore, dict[str, ToolCall]] = {}
        for (child_id, request_id), (request, delegated_store) in list(
            self._delegated.items()
        ):
            if delegated_store not in delegated_pending:
                delegated_pending[delegated_store] = pending_for_store(delegated_store)
            pending = delegated_pending[delegated_store]
            if request_id not in pending:
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
                label=request.label,
                child_instance_id=child_id,
                effective_cwd=request.effective_cwd,
                resolved_path=request.resolved_path,
                action=request.action,
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
            return delegated[1].resolve_approval(delegated[0].request_id, "abort")
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
        request_id: str,
        *,
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> ApprovalDecision | None:
        del execution_token, capability
        delegated = self._delegated_entry(request_id, None)
        if delegated is not None:
            delegated[1].resolve_approval(delegated[0].request_id, "abort")
            state = delegated[1].approval_states().get(delegated[0].request_id)
            return _resolved_decision(state[1] if state is not None else None)
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
        if request_id in self._ephemeral:
            return None
        if self._store is not None and request_id in self._store.approval_states():
            return None
        matches = [
            delegated
            for (child_id, delegated_id), delegated in self._delegated.items()
            if delegated_id == request_id
        ]
        return matches[0] if len(matches) == 1 else None

    def prepare(
        self,
        tool_call: ToolCall,
        *,
        capability: ApprovalCapability,
    ) -> ApprovalRequest | None:
        """Build an ask request for atomic persistence with its assistant anchor."""

        store = self._require_store()
        state = store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(capability) is ApprovalDecision.ASK:
            return ApprovalRequest(tool_call.id, tool_call, action=capability.action)
        return None

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: _AbortSignal,
        *,
        persist_request: bool = True,
        capability: ApprovalCapability,
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
            action = capability.action
            decision = self.decide(capability)
            if decision is not ApprovalDecision.ASK:
                return decision
            if persist_request:
                store.append_message_with_approval_requests(
                    Message(
                        MessageRole.ASSISTANT,
                        [ToolUseContent(tool_call)],
                        metadata={"response_state": ASSISTANT_RESPONSE_SYNTHETIC},
                    ),
                    [
                        (
                            tool_call.id,
                            tool_call,
                            ApprovalRequest(
                                tool_call.id,
                                tool_call,
                                action=capability.action,
                            ).audit_facts(),
                            {},
                        )
                    ],
                )
            else:
                self._ephemeral[tool_call.id] = (
                    ApprovalRequest(tool_call.id, tool_call, action=action),
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
                return self._resolve_abort_or_winner(
                    tool_call.id, capability=capability
                )
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
                return self._resolve_abort_or_winner(
                    tool_call.id, capability=capability
                )

    def forget_ephemeral(self, request_id: str) -> None:
        """Remove one non-durable approval after its owner finishes."""

        self._ephemeral.pop(request_id, None)

    def _resolve_abort_or_winner(
        self,
        request_id: str,
        *,
        capability: ApprovalCapability,
    ) -> ApprovalDecision | None:
        return self.abort_or_winner(request_id, capability=capability)

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
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> tuple[ToolResult | None, AbortSignal]:
        execution_signal = signal
        if self.policy is not None and not skip_approval:
            approval_started = (
                self.policy.durable_decision(tool_call.id) is None
                and self.policy.decide(capability) is ApprovalDecision.ASK
            )
            if approval_started and lifecycle is not None:
                lifecycle("approval_start")
            try:
                authorize = self.policy.authorize
                parameters = inspect.signature(authorize).parameters
                kwargs: dict[str, object] = {}
                if not persist_request and "persist_request" in parameters:
                    kwargs["persist_request"] = False
                if execution_token is not None and "execution_token" in parameters:
                    kwargs["execution_token"] = execution_token
                if capability is not None and "capability" in parameters:
                    kwargs["capability"] = capability
                decision = await authorize(tool_call, signal, **kwargs)
            except Exception as exc:  # noqa: BLE001 - report approval failures
                return ToolResult(
                    tool_call.id, f"approval failed: {exc}", True
                ), execution_signal
            finally:
                if approval_started and lifecycle is not None:
                    lifecycle("approval_end")
            if decision is None:
                return canceled_result(tool_call.id), execution_signal
            if signal.is_set():
                durable_decision = self.policy.durable_decision(tool_call.id)
                if durable_decision == ApprovalDecision.DENY.value:
                    return ToolResult(
                        tool_call.id, "tool execution denied", True
                    ), execution_signal
                if durable_decision != ApprovalDecision.ALLOW.value:
                    return canceled_result(tool_call.id), execution_signal
                execution_signal = advance_generation(signal)
                if execution_signal.is_set():
                    return canceled_result(tool_call.id), execution_signal
            if decision is ApprovalDecision.DENY:
                message = "tool execution denied"
                denial_reason = getattr(self.policy, "denial_reason", None)
                if callable(denial_reason):
                    reason = denial_reason(tool_call.id)
                    if reason:
                        message = f"{message}: {reason}"
                return ToolResult(tool_call.id, message, True), execution_signal
        if self.hook is None:
            if signal.is_set() and execution_signal is signal:
                return canceled_result(tool_call.id), execution_signal
            return None, execution_signal
        try:
            allowed = self.hook(tool_call.name, arguments)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        except Exception as exc:  # noqa: BLE001 - report hook failures
            return ToolResult(
                tool_call.id, f"pre-execution hook failed: {exc}", True
            ), execution_signal
        if isinstance(allowed, str):
            return ToolResult(tool_call.id, allowed, True), execution_signal
        if allowed is False:
            return ToolResult(
                tool_call.id, "tool execution denied by hook", True
            ), execution_signal
        if signal.is_set() and execution_signal is signal:
            return canceled_result(tool_call.id), execution_signal
        return None, execution_signal
