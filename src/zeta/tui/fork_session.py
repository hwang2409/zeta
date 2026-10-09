"""Run attention discussion forks while the main TUI runtime stays live."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Protocol

from ..core.approval import ApprovalDecision, ApprovalRequest
from ..tools._shared.shell import trusted_macro_display
from .render import render_approval_card


class RuntimeResult(Enum):
    """Why a visible runtime released the terminal to its stack controller."""

    OPEN_CHILD = auto()
    RETURN_TO_PARENT = auto()
    EXIT = auto()


class ForkHost(Protocol):
    """The small interface a TUI runtime exposes to the fork stack."""

    async def run(self, *, resume_ui: bool = False) -> RuntimeResult: ...

    async def leave_fork(self) -> None: ...

    def take_open_fork(self) -> str | None: ...

    async def close(self) -> None: ...

    def set_fork_controller(self, controller: ForkStackController | None) -> None: ...

    @property
    def local_pending_approvals(self) -> tuple[ApprovalRequest, ...]: ...

    async def resolve_local_approval(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None: ...

    def sync_visible_approvals(self, approvals: tuple[OwnedApproval, ...]) -> None: ...

    def notification_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class OwnedApproval:
    """A pending request and its stable, owner-qualified command handle."""

    owner: ForkHost
    request: ApprovalRequest
    handle: str = ""


class ForkRuntimeMixin:
    """Adapt one TUI runtime to the controller's approval interface."""

    def _init_fork_runtime(self) -> None:
        self._fork_controller: ForkStackController | None = None
        self._approval_units: dict[tuple[int, object], object] = {}
        self._run_result = RuntimeResult.EXIT

    def set_fork_controller(self, controller: ForkStackController | None) -> None:
        self._fork_controller = controller

    @property
    def local_pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        return self._submissions.pending_approvals

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        if self._fork_controller is None:
            return self.local_pending_approvals
        return tuple(
            approval.request for approval in self._fork_controller.pending_approvals
        )

    @property
    def first_pending_approval_key(self) -> str | None:
        if self._fork_controller is not None:
            approvals = self._fork_controller.pending_approvals
            return approvals[0].handle if approvals else None
        approvals = self.local_pending_approvals
        return str(approvals[0].key) if approvals else None

    async def resolve_local_approval(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None:
        await self._submissions.approval_command(decision, requested_key, always=always)

    def _present_pending_approvals(self) -> None:
        if self._fork_controller is not None:
            self._fork_controller.approvals_changed()
            return
        self.sync_visible_approvals(
            tuple(
                OwnedApproval(self, request) for request in self.local_pending_approvals
            )
        )

    def sync_visible_approvals(self, approvals: tuple[OwnedApproval, ...]) -> None:
        current = {(id(item.owner), item.request.key) for item in approvals}
        for key in self._approval_units.keys() - current:
            unit = self._approval_units.pop(key)
            self._transcript.remove(unit, leading_blank=True)
        for index, item in enumerate(approvals):
            request = item.request
            key = (id(item.owner), request.key)
            if key in self._approval_units:
                continue
            unit = self._presenter.print_unit(
                render_approval_card(
                    request.tool_call.name,
                    request.tool_call.arguments,
                    label=request.label,
                    key=item.handle or str(request.key),
                    shortcut=index == 0,
                    trusted_display=trusted_macro_display(request.tool_call.id),
                    project_display=(
                        request.project_id,
                        request.project_name,
                        request.filename,
                        request.content_bytes,
                        request.preview,
                    )
                    if request.filename is not None
                    and request.content_bytes is not None
                    and request.preview is not None
                    else None,
                    execution_display=(request.effective_cwd, request.resolved_path)
                    if request.effective_cwd is not None
                    or request.resolved_path is not None
                    else None,
                )
            )
            if unit is not None:
                self._approval_units[key] = unit
        self._invalidate_prompt()


class ForkStackController:
    """Own the visible runtime, suspension state, approvals, and shutdown order."""

    def __init__(
        self,
        main_app: ForkHost,
        build_fork: Callable[[str], Awaitable[ForkHost]],
    ) -> None:
        self._stack = [main_app]
        self._build_fork = build_fork
        self._resuming: set[int] = set()
        self._main_notification_baselines: dict[int, int] = {}
        self._approval_handles: dict[tuple[int, object], str] = {}
        self._next_approval_handle = count(1)
        main_app.set_fork_controller(self)

    @property
    def visible(self) -> ForkHost:
        return self._stack[-1]

    def is_visible(self, runtime: ForkHost) -> bool:
        return self.visible is runtime

    @property
    def pending_approvals(self) -> tuple[OwnedApproval, ...]:
        pending = [
            (owner, request)
            for owner in reversed(self._stack)
            for request in owner.local_pending_approvals
        ]
        active = {(id(owner), request.key) for owner, request in pending}
        self._approval_handles = {
            key: handle
            for key, handle in self._approval_handles.items()
            if key in active
        }
        approvals = []
        for owner, request in pending:
            key = (id(owner), request.key)
            handle = self._approval_handles.get(key)
            if handle is None:
                handle = f"approval-{next(self._next_approval_handle)}"
                self._approval_handles[key] = handle
            approvals.append(OwnedApproval(owner, request, handle))
        return tuple(approvals)

    async def resolve_approval(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None:
        approval = self._find_approval(requested_key)
        if approval is None:
            await self.visible.resolve_local_approval(
                decision, requested_key, always=always
            )
            return
        await approval.owner.resolve_local_approval(
            decision, str(approval.request.key), always=always
        )
        self.approvals_changed()

    def approvals_changed(self) -> None:
        self.visible.sync_visible_approvals(self.pending_approvals)

    def main_activity_pending(self, runtime: ForkHost) -> bool:
        baseline = self._main_notification_baselines.get(id(runtime))
        if baseline is None:
            return False
        return self._stack[0].notification_count() > baseline

    async def run(self) -> None:
        try:
            while self._stack:
                current = self.visible
                resume_ui = id(current) in self._resuming
                self._resuming.discard(id(current))
                result = await current.run(resume_ui=resume_ui)
                if result is RuntimeResult.OPEN_CHILD:
                    target = current.take_open_fork()
                    if target is None:
                        raise RuntimeError("runtime requested a child without a target")
                    fork = await self._build_fork(target)
                    fork.set_fork_controller(self)
                    self._main_notification_baselines[id(fork)] = self._stack[
                        0
                    ].notification_count()
                    self._stack.append(fork)
                    self.approvals_changed()
                    continue
                if result is RuntimeResult.EXIT:
                    await self.close()
                    return
                if result is not RuntimeResult.RETURN_TO_PARENT:
                    raise RuntimeError(f"unknown runtime result: {result!r}")
                if len(self._stack) == 1:
                    raise RuntimeError("main runtime cannot return to a parent")
                await current.leave_fork()
                current.set_fork_controller(None)
                self._main_notification_baselines.pop(id(current), None)
                self._stack.pop()
                self._resuming.add(id(self.visible))
                self.approvals_changed()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        first_error: BaseException | None = None
        for runtime in reversed(self._stack):
            try:
                await runtime.close()
            except BaseException as exc:  # noqa: BLE001 - finish every runtime
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _find_approval(self, requested_key: str | None) -> OwnedApproval | None:
        approvals = self.pending_approvals
        if requested_key is None:
            return approvals[0] if approvals else None
        for approval in approvals:
            if approval.handle == requested_key:
                return approval
        raw_matches = [
            approval
            for approval in approvals
            if str(approval.request.key) == requested_key
        ]
        if len(raw_matches) > 1:
            raise ValueError(
                f"ambiguous approval key {requested_key!r}; use the approval handle"
            )
        return raw_matches[0] if raw_matches else None


async def run_fork_stack(
    main_app: ForkHost,
    build_fork: Callable[[str], Awaitable[ForkHost]],
) -> None:
    """Run one main runtime and any discussion runtimes opened above it."""

    await ForkStackController(main_app, build_fork).run()


__all__ = [
    "ForkHost",
    "ForkRuntimeMixin",
    "ForkStackController",
    "OwnedApproval",
    "RuntimeResult",
    "run_fork_stack",
]
