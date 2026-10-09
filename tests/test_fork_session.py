from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalRequest
from zeta.protocol.types import ToolCall
from zeta.tui import fork_session
from zeta.tui.fork_session import ForkStackController, OwnedApproval, run_fork_stack


class _LegacyRuntimeResult(Enum):
    OPEN_CHILD = auto()
    RETURN_TO_PARENT = auto()
    EXIT = auto()


RuntimeResult = getattr(fork_session, "RuntimeResult", _LegacyRuntimeResult)


@dataclass
class _Resource:
    alive: bool = True


@dataclass
class _FakeApp:
    name: str
    script: list[str | object]
    is_fork: bool = False
    child: _Resource = field(default_factory=_Resource)
    task: _Resource = field(default_factory=_Resource)
    turn: _Resource = field(default_factory=_Resource)
    notifications: int = 0
    runs: int = 0
    closed: bool = False
    left: bool = False
    _pending_fork: str | None = None
    controller: ForkStackController | None = None
    approvals: list[ApprovalRequest] = field(default_factory=list)
    resolved: list[tuple[ApprovalDecision, str | None, bool]] = field(
        default_factory=list
    )
    approval_views: list[tuple[str, ...]] = field(default_factory=list)
    run_gate: asyncio.Event | None = None
    close_order: list[str] | None = None
    close_error: BaseException | None = None
    resume_values: list[bool] = field(default_factory=list)

    async def run(self, *, resume_ui: bool = False) -> object:
        self.runs += 1
        self.resume_values.append(resume_ui)
        if self.run_gate is not None:
            await self.run_gate.wait()
        action = self.script.pop(0) if self.script else RuntimeResult.EXIT
        if isinstance(action, str):
            self._pending_fork = action
            return RuntimeResult.OPEN_CHILD
        if action in {RuntimeResult.RETURN_TO_PARENT, RuntimeResult.EXIT}:
            self.closed = True
        return action

    async def leave_fork(self) -> None:
        self.left = True

    async def close(self) -> None:
        if self.close_order is not None:
            self.close_order.append(self.name)
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    def take_open_fork(self) -> str | None:
        target, self._pending_fork = self._pending_fork, None
        return target

    def set_fork_controller(self, controller: ForkStackController | None) -> None:
        self.controller = controller

    @property
    def local_pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self.approvals)

    async def resolve_local_approval(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None:
        self.resolved.append((decision, requested_key, always))
        self.approvals = [
            request for request in self.approvals if str(request.key) != requested_key
        ]

    def sync_visible_approvals(self, approvals: tuple[OwnedApproval, ...]) -> None:
        self.approval_views.append(
            tuple(
                getattr(approval, "handle", str(approval.request.key))
                for approval in approvals
            )
        )

    def notification_count(self) -> int:
        return self.notifications

    @property
    def main_activity_pending(self) -> bool:
        return self.controller is not None and self.controller.main_activity_pending(
            self
        )


def _approval(key: str, tool_name: str = "read") -> ApprovalRequest:
    arguments = {"path": key} if tool_name == "read" else {"command": key}
    return ApprovalRequest(key, ToolCall(key, tool_name, arguments))


def test_opening_discussion_keeps_main_runtime_live_and_untouched() -> None:
    main = _FakeApp("main", script=["fork-1", RuntimeResult.EXIT])
    fork = _FakeApp("fork", script=[RuntimeResult.RETURN_TO_PARENT], is_fork=True)
    built: list[str] = []

    async def build_fork(target: str) -> _FakeApp:
        built.append(target)
        assert (main.child.alive, main.task.alive, main.turn.alive) == (True,) * 3
        assert main.closed is False
        return fork

    asyncio.run(run_fork_stack(main, build_fork))

    assert built == ["fork-1"]
    assert (main.child.alive, main.task.alive, main.turn.alive) == (True,) * 3
    assert main.runs == 2
    assert main.resume_values == [False, True]
    assert (fork.left, fork.closed) == (True, True)
    assert main.left is False
    assert main.closed is True


def test_main_notification_during_discussion_is_seen_by_the_fork() -> None:
    async def driver() -> None:
        gate = asyncio.Event()
        main = _FakeApp("main", script=["fork-1", RuntimeResult.EXIT])
        fork = _FakeApp(
            "fork",
            script=[RuntimeResult.RETURN_TO_PARENT],
            is_fork=True,
            run_gate=gate,
        )

        async def build_fork(target: str) -> _FakeApp:
            return fork

        task = asyncio.create_task(run_fork_stack(main, build_fork))
        await asyncio.sleep(0)
        main.notifications += 1
        assert fork.main_activity_pending is True
        gate.set()
        await task

    asyncio.run(driver())


@pytest.mark.parametrize(
    ("main_keys", "fork_keys", "expected"),
    [
        ((), ("fork",), ("fork",)),
        (("main",), (), ("main",)),
        (("main",), ("fork",), ("fork", "main")),
    ],
)
def test_visible_approvals_include_each_owning_runtime(
    main_keys: tuple[str, ...],
    fork_keys: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    async def driver() -> None:
        gate = asyncio.Event()
        main = _FakeApp(
            "main",
            ["fork-1", RuntimeResult.EXIT],
            approvals=[_approval(k) for k in main_keys],
        )
        fork = _FakeApp(
            "fork",
            [RuntimeResult.RETURN_TO_PARENT],
            is_fork=True,
            approvals=[_approval(k) for k in fork_keys],
            run_gate=gate,
        )

        async def build_fork(target: str) -> _FakeApp:
            return fork

        task = asyncio.create_task(run_fork_stack(main, build_fork))
        await asyncio.sleep(0)
        approvals = fork.controller.pending_approvals
        assert tuple(str(item.request.key) for item in approvals) == expected
        assert fork.approval_views[-1] == tuple(item.handle for item in approvals)
        for approval in approvals:
            await fork.controller.resolve_approval(
                ApprovalDecision.ALLOW, approval.handle
            )
        assert [item[1] for item in fork.resolved] == list(fork_keys)
        assert [item[1] for item in main.resolved] == list(main_keys)
        gate.set()
        await task

    asyncio.run(driver())


def test_forwarded_approval_is_invalidated_after_external_resolution_and_return() -> (
    None
):
    async def driver() -> None:
        gate = asyncio.Event()
        main = _FakeApp(
            "main",
            ["fork-1", RuntimeResult.EXIT],
            approvals=[_approval("main")],
        )
        fork = _FakeApp(
            "fork", [RuntimeResult.RETURN_TO_PARENT], is_fork=True, run_gate=gate
        )

        async def build_fork(target: str) -> _FakeApp:
            return fork

        task = asyncio.create_task(run_fork_stack(main, build_fork))
        await asyncio.sleep(0)
        assert fork.approval_views[-1] == (
            fork.controller.pending_approvals[0].handle,
        )
        main.approvals.clear()
        main.controller.approvals_changed()
        assert fork.approval_views[-1] == ()
        gate.set()
        await task
        assert main.approval_views[-1] == ()

    asyncio.run(driver())


def test_exit_from_visible_fork_closes_stack_without_resuming_main() -> None:
    order: list[str] = []
    main = _FakeApp("main", ["fork-1"], close_order=order)
    fork = _FakeApp(
        "fork", [RuntimeResult.EXIT], is_fork=True, close_order=order
    )

    async def build_fork(target: str) -> _FakeApp:
        return fork

    asyncio.run(run_fork_stack(main, build_fork))

    assert order == ["fork", "main"]
    assert main.runs == 1
    assert main.resume_values == [False]
    assert fork.left is False
    assert main.closed and fork.closed


def test_return_to_parent_ends_fork_and_resumes_main() -> None:
    main = _FakeApp("main", ["fork-1", RuntimeResult.EXIT])
    fork = _FakeApp("fork", [RuntimeResult.RETURN_TO_PARENT], is_fork=True)

    async def build_fork(target: str) -> _FakeApp:
        return fork

    asyncio.run(run_fork_stack(main, build_fork))

    assert fork.left is True
    assert main.runs == 2
    assert main.resume_values == [False, True]


def test_same_raw_approval_keys_have_owner_qualified_handles() -> None:
    async def driver() -> None:
        gate = asyncio.Event()
        main = _FakeApp(
            "main", ["fork-1", RuntimeResult.EXIT], approvals=[_approval("same")]
        )
        fork = _FakeApp(
            "fork",
            [RuntimeResult.RETURN_TO_PARENT],
            is_fork=True,
            approvals=[_approval("same")],
            run_gate=gate,
        )

        async def build_fork(target: str) -> _FakeApp:
            return fork

        task = asyncio.create_task(run_fork_stack(main, build_fork))
        await asyncio.sleep(0)
        approvals = fork.controller.pending_approvals
        handles = tuple(getattr(item, "handle", str(item.request.key)) for item in approvals)
        assert len(handles) == 2
        assert len(set(handles)) == 2
        assert fork.approval_views[-1] == handles

        for approval, handle in zip(approvals, handles, strict=True):
            await fork.controller.resolve_approval(ApprovalDecision.ALLOW, handle)
            assert approval.owner.resolved[-1][1] == "same"
        assert len(fork.resolved) == 1
        assert len(main.resolved) == 1
        gate.set()
        await task

    asyncio.run(driver())


def test_replaced_approval_gets_new_handle_and_rejects_stale_handle() -> None:
    async def driver() -> None:
        request = _approval("same")
        main = _FakeApp("main", [], approvals=[request])
        controller = ForkStackController(main, pytest.fail)

        controller.approvals_changed()
        old_handle = controller.pending_approvals[0].handle
        assert main.approval_views[-1] == (old_handle,)

        replacement = _approval("same", "bash")
        main.approvals = [replacement]
        controller.approvals_changed()
        new_handle = controller.pending_approvals[0].handle

        assert new_handle != old_handle
        assert main.approval_views[-1] == (new_handle,)
        assert old_handle not in {
            approval.handle for approval in controller.pending_approvals
        }
        with pytest.raises(
            ValueError, match=rf"unknown or expired approval handle {old_handle!r}"
        ):
            await controller.resolve_approval(ApprovalDecision.ALLOW, old_handle)
        assert main.resolved == []
        assert main.approvals == [replacement]

    asyncio.run(driver())


def test_ambiguous_raw_approval_key_is_rejected() -> None:
    async def driver() -> None:
        gate = asyncio.Event()
        main = _FakeApp(
            "main", ["fork-1", RuntimeResult.EXIT], approvals=[_approval("same")]
        )
        fork = _FakeApp(
            "fork",
            [RuntimeResult.RETURN_TO_PARENT],
            is_fork=True,
            approvals=[_approval("same")],
            run_gate=gate,
        )

        async def build_fork(target: str) -> _FakeApp:
            return fork

        task = asyncio.create_task(run_fork_stack(main, build_fork))
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="ambiguous approval key"):
            await fork.controller.resolve_approval(ApprovalDecision.ALLOW, "same")
        assert fork.resolved == []
        assert main.resolved == []
        gate.set()
        await task

    asyncio.run(driver())


def test_early_fork_close_failure_still_closes_main_and_reraises() -> None:
    order: list[str] = []

    class ExitApp(_FakeApp):
        async def run(self, *, resume_ui: bool = False) -> object:
            if self.is_fork:
                raise KeyboardInterrupt
            return await super().run(resume_ui=resume_ui)

    main = ExitApp("main", ["fork-1"], close_order=order)
    failure = RuntimeError("fork close failed")
    fork = ExitApp("fork", [], is_fork=True, close_order=order, close_error=failure)

    async def build_fork(target: str) -> _FakeApp:
        return fork

    with pytest.raises(RuntimeError, match="fork close failed"):
        asyncio.run(run_fork_stack(main, build_fork))

    assert order == ["fork", "main"]
    assert main.closed and fork.closed


def test_forks_can_stack_and_unwind_newest_first() -> None:
    main = _FakeApp("main", script=["fork-1", RuntimeResult.EXIT])
    inner = _FakeApp(
        "inner", script=["fork-2", RuntimeResult.RETURN_TO_PARENT], is_fork=True
    )
    deepest = _FakeApp(
        "deepest", script=[RuntimeResult.RETURN_TO_PARENT], is_fork=True
    )
    built = {"fork-1": inner, "fork-2": deepest}

    async def build_fork(target: str) -> _FakeApp:
        return built[target]

    asyncio.run(run_fork_stack(main, build_fork))

    assert (inner.left, inner.closed) == (True, True)
    assert (deepest.left, deepest.closed) == (True, True)
    assert main.closed is True
    assert (main.runs, inner.runs, deepest.runs) == (2, 2, 1)
