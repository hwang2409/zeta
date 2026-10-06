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
            raise ValueError(f"invalid approval rule {text!r}: empty argument pattern")
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
    # Harness-owned, immutable display facts; never read from provider args by
    # frontends when present.
    project_id: str | None = None
    project_name: str | None = None
    filename: str | None = None
    content_bytes: int | None = None
    preview: str | None = None
    effective_cwd: str | None = None
    resolved_path: str | None = None

    @property
    def key(self) -> str | tuple[str, str]:
        if self.child_instance_id is None:
            return self.request_id
        return self.child_instance_id, self.request_id

    def audit_display(self) -> dict[str, object]:
        """Return immutable presentation facts, never executable authority."""

        display: dict[str, object] = {}
        if self.project_id is not None or self.filename is not None:
            display.update(
                {
                    "project_id": self.project_id,
                    "project_name": self.project_name,
                    "filename": self.filename,
                    "utf8_bytes": self.content_bytes,
                    "preview": self.preview,
                }
            )
        if self.effective_cwd is not None or self.resolved_path is not None:
            display.update(
                {
                    "effective_cwd": self.effective_cwd,
                    "resolved_path": self.resolved_path,
                }
            )
        return display


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
        self._subject_resolvers: dict[
            str, Callable[[Mapping[str, object]], str | None]
        ] = {}
        self._notices: list[str] = []
        self._store = store
        self._delegated: dict[
            tuple[str, str], tuple[ApprovalRequest, ConversationStore]
        ] = {}
        self._ephemeral: dict[str, tuple[ApprovalRequest, str | None]] = {}
        self._display_resolver: Callable[[ApprovalRequest], ApprovalRequest] | None = (
            None
        )

    def bind_display_resolver(
        self, resolver: Callable[[ApprovalRequest], ApprovalRequest] | None
    ) -> None:
        self._display_resolver = resolver

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

    def declare_subjects(self, subjects: Mapping[str, str | None]) -> tuple[str, ...]:
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

    def declare_subject_resolver(
        self, tool: str, resolver: Callable[[Mapping[str, object]], str | None]
    ) -> None:
        self._subject_resolvers[tool] = resolver

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

    def approval_subject(self, tool_name: str) -> str | None:
        """Return the declared subject argument for a registered tool."""

        return self._subjects.get(tool_name)

    def decide_for_child(
        self,
        tool_name: str,
        arguments: dict[str, object],
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> ApprovalDecision:
        """Match delegated calls in the parent's cwd and canonical path frame."""

        decision, _binding = self.decide_for_child_with_binding(
            tool_name,
            arguments,
            parent_cwd=parent_cwd,
            child_cwd=child_cwd,
        )
        return decision

    def capture_child_binding(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        *,
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovedExecution | None, bool]:
        """Capture the object a human approval is about before it is displayed.

        The boolean says that this tool has an object-scoped approval subject.
        Callers must retain the result until the decision is consumed; a durable
        ALLOW without these in-memory facts is therefore not executable.
        """
        subject = self._subjects.get(tool_name)
        if subject == "path":
            value = arguments.get("path")
            if not isinstance(value, str):
                return None, True
            candidate = os.path.expanduser(value)
            if not os.path.isabs(candidate):
                candidate = os.path.join(os.fspath(child_cwd), candidate)
            return _bind_approved_path(os.path.realpath(candidate)), True
        if subject == "command":
            return _bind_approved_cwd(os.path.realpath(os.fspath(child_cwd))), True
        return None, False

    def decide_for_child_with_binding(
        self,
        tool_name: str,
        arguments: dict[str, object],
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        """Match a delegated call and bind scoped allows to canonical objects."""

        subject = self._subjects.get(tool_name)
        if subject == "path":
            value = arguments.get("path")
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
                    tool_name,
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

        if subject == "command":
            if self._matches(
                self._always_deny, tool_name, arguments, unreadable=True
            ):
                return ApprovalDecision.DENY, None
            if self._matches(
                self._always_ask, tool_name, arguments, unreadable=True
            ):
                return ApprovalDecision.ASK, None
            canonical_parent = os.path.realpath(parent_cwd)
            canonical_child = os.path.realpath(child_cwd)
            same_cwd = canonical_parent == canonical_child
            allow_rules = frozenset(
                rule
                for rule in self._always_allow
                if rule.tool != tool_name or rule.pattern is None or same_cwd
            )
            matching_rules = [
                rule
                for rule in allow_rules
                if self._matches(
                    frozenset({rule}), tool_name, arguments, unreadable=False
                )
            ]
            if matching_rules:
                if any(rule.pattern is None for rule in matching_rules):
                    return ApprovalDecision.ALLOW, None
                binding = _bind_approved_cwd(canonical_child)
                if binding is None:
                    return ApprovalDecision.ASK, None
                return ApprovalDecision.ALLOW, binding
            return self.default, None

        return self.decide(tool_name, arguments), None

    @staticmethod
    def _matching_child_path_rule(
        rules: frozenset[ApprovalRule],
        tool_name: str,
        resolved_path: str | None,
        *,
        parent_cwd: str,
        unreadable: bool,
    ) -> ApprovalRule | None:
        for rule in sorted(rules, key=lambda candidate: candidate.pattern is not None):
            if rule.tool != tool_name:
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
            resolver = self._subject_resolvers.get(tool_name)
            if resolver is not None and isinstance(arguments, Mapping):
                value = resolver(arguments)
            else:
                subject = self._subjects.get(tool_name)
                if subject is None:
                    continue
                value = (
                    arguments.get(subject) if isinstance(arguments, Mapping) else None
                )
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
        requests = [
            (self._display_resolver or (lambda request: request))(
                ApprovalRequest(request_id, tool_call)
            )
            for request_id, tool_call in pending_for_store(store).items()
        ]
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
                request.label,
                child_id,
                request.project_id,
                request.project_name,
                request.filename,
                request.content_bytes,
                request.preview,
                request.effective_cwd,
                request.resolved_path,
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
                    Message(
                        MessageRole.ASSISTANT,
                        [ToolUseContent(tool_call)],
                        metadata={"response_state": ASSISTANT_RESPONSE_SYNTHETIC},
                    ),
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
        execution_token: str | None = None,
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
                authorize = self.policy.authorize
                parameters = inspect.signature(authorize).parameters
                kwargs: dict[str, object] = {}
                if not persist_request and "persist_request" in parameters:
                    kwargs["persist_request"] = False
                if execution_token is not None and "execution_token" in parameters:
                    kwargs["execution_token"] = execution_token
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
