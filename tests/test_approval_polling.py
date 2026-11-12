from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from tests.support.owned import await_or_fail
from zeta.core import approval as approval_module
from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.tools._action_metadata import ApprovalBinding, ResolvedCapability
from zeta.tools.agent.approval import ChildApprovalPolicy


class _ControlledSignal:
    def __init__(self) -> None:
        self.released = asyncio.Event()
        self.wait_entered = asyncio.Event()
        self.wait_finished = asyncio.Event()

    def is_set(self) -> bool:
        return self.released.is_set()

    async def wait(self) -> None:
        self.wait_entered.set()
        try:
            await asyncio.wait_for(self.released.wait(), 10)
        finally:
            self.wait_finished.set()

    def abort(self) -> None:
        self.released.set()


class _SlowCancelSignal(_ControlledSignal):
    def __init__(self) -> None:
        super().__init__()
        self.drain_entered = asyncio.Event()
        self.release_drain = asyncio.Event()

    async def wait(self) -> None:
        self.wait_entered.set()
        try:
            await asyncio.wait_for(self.released.wait(), 10)
        except asyncio.CancelledError:
            self.drain_entered.set()
            await asyncio.wait_for(self.release_drain.wait(), 10)
            raise
        finally:
            self.wait_finished.set()


class _ControlledTick:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.finished = asyncio.Event()

    async def __call__(self, _delay: float) -> None:
        self.entered.set()
        try:
            await asyncio.wait_for(self.released.wait(), 10)
        finally:
            self.finished.set()


class _AdmissionControl:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.finished = asyncio.Event()
        self.aborts: list[tuple[str, str | None, str]] = []

    def register_delegated(self, *args: object) -> None:
        pass

    async def cleanup_delegated(self, _child_instance_id: str) -> None:
        pass

    async def request_approval_abort_gate(
        self,
        request_id: str,
        *,
        child_instance_id: str | None = None,
        reason: str = "turn_canceled",
    ) -> dict[str, str]:
        self.aborts.append((request_id, child_instance_id, reason))
        self.entered.set()
        try:
            await asyncio.wait_for(self.released.wait(), 10)
            return {"decision": "abort", "reason": reason}
        finally:
            self.finished.set()


@dataclass
class _Harness:
    kind: Literal["root", "nested"]
    policy: ApprovalPolicy | ChildApprovalPolicy
    root: ApprovalPolicy
    store: ConversationStore
    stores: list[ConversationStore]
    call: ToolCall
    capability: ResolvedCapability

    async def authorize(
        self, signal: _ControlledSignal | AbortSignal, *, token: str = "execution"
    ) -> ApprovalDecision | None:
        if isinstance(self.policy, ChildApprovalPolicy):
            return await self.policy.authorize(
                self.call,
                signal,
                execution_token=token,
                capability=self.capability,
            )
        return await self.policy.authorize(
            self.call, signal, capability=self.capability
        )

    def resolve(self, decision: str) -> bool:
        return self.store.resolve_approval(self.call.id, decision)

    async def decision(self) -> str | None:
        state = await self.store.approval_state(self.call.id)
        return None if state is None else state[1]

    def persist(self) -> None:
        self.store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(self.call)]),
            [(self.call.id, self.call)],
        )

    def assert_allow_binding_transferred_once(self) -> None:
        if isinstance(self.policy, ChildApprovalPolicy):
            assert self.policy.consume_execution_binding("execution") is not None
            assert self.policy.consume_execution_binding("execution") is None

    def assert_pending_binding_cleared(self) -> None:
        if isinstance(self.policy, ChildApprovalPolicy):
            assert self.call.id not in self.policy._pending_bindings

    def close(self) -> None:
        for store in reversed(self.stores):
            store.close()


def _harness(tmp_path: Path, kind: Literal["root", "nested"]) -> _Harness:
    root_store = ConversationStore(tmp_path / "root", cwd=tmp_path)
    root = ApprovalPolicy(store=root_store)
    call = ToolCall(
        "approval", "bash", {"command": "pwd", "cwd": str(tmp_path)}
    )
    capability = ResolvedCapability(
        "bash",
        None,
        True,
        "command",
        "pwd",
        ApprovalBinding.CWD,
        None,
        call.arguments,
    )
    if kind == "root":
        return _Harness(kind, root, root, root_store, [root_store], call, capability)
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    child = ChildApprovalPolicy(
        root, child_store, "child", "innermost", child_cwd=tmp_path
    )
    return _Harness(
        kind, child, root, child_store, [root_store, child_store], call, capability
    )


_CASES = [
    "decision_before_wait",
    "abort_already_set",
    "abort_before_winner",
    "winner_before_abort",
    "winner_after_tick",
    "abort_and_tick_together",
    "duplicate_same",
    "duplicate_conflicting",
    "cancel_wait",
    "cancel_drain",
    "cancel_serve_admission",
    "delegation_gone",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["root", "nested"])
@pytest.mark.parametrize("case", _CASES)
async def test_authorize_polling_interleavings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["root", "nested"],
    case: str,
) -> None:
    harness = _harness(tmp_path, kind)
    signal = _SlowCancelSignal() if case == "cancel_drain" else _ControlledSignal()
    tick = _ControlledTick()
    monkeypatch.setattr(approval_module.asyncio, "sleep", tick)
    task: asyncio.Task[ApprovalDecision | None] | None = None
    try:
        if case == "decision_before_wait":
            append = harness.store.append_message_with_approval_requests

            def persist_winner(*args: object, **kwargs: object) -> object:
                result = append(*args, **kwargs)  # type: ignore[arg-type]
                assert harness.resolve("allow")
                return result

            async def reject_abort(*args: object, **kwargs: object) -> None:
                pytest.fail("terminal approval must be read before abort arbitration")

            monkeypatch.setattr(
                harness.store, "append_message_with_approval_requests", persist_winner
            )
            # test-contract: seam prove current-state read precedes abort arbitration
            monkeypatch.setattr(harness.policy, "_resolve_abort_or_winner", reject_abort)
            signal.abort()
            assert await harness.authorize(signal) is ApprovalDecision.ALLOW
            assert await harness.decision() == "allow"
            harness.assert_allow_binding_transferred_once()
            assert not signal.wait_entered.is_set()
            return

        if case == "abort_already_set":
            signal.abort()
            assert await harness.authorize(signal) is None
            assert await harness.decision() == "abort"
            harness.assert_pending_binding_cleared()
            if kind == "nested":
                assert await harness.store.approval_resolution(harness.call.id) == {
                    "request_id": harness.call.id,
                    "decision": "abort",
                    "reason": "child_cleanup",
                }
            return

        if case in {"winner_before_abort", "abort_and_tick_together"}:
            original = harness.store.resolve_approval

            def allow_before_abort(
                request_id: str, decision: str, *, reason: str | None = None
            ) -> bool:
                if decision == "abort":
                    assert original(request_id, "allow")
                return original(request_id, decision, reason=reason)

            monkeypatch.setattr(harness.store, "resolve_approval", allow_before_abort)

        task = asyncio.create_task(harness.authorize(signal))
        await await_or_fail(signal.wait_entered.wait(), what="approval abort waiter")
        await await_or_fail(tick.entered.wait(), what="approval poll tick")

        if case == "abort_before_winner":
            signal.abort()
            assert await await_or_fail(task, what="aborted approval") is None
            assert await harness.decision() == "abort"
            harness.assert_pending_binding_cleared()
        elif case == "winner_before_abort":
            signal.abort()
            assert await await_or_fail(task, what="concurrent approval winner") is ApprovalDecision.ALLOW
            assert await harness.decision() == "allow"
            harness.assert_allow_binding_transferred_once()
        elif case == "winner_after_tick":
            assert harness.resolve("allow")
            tick.released.set()
            assert await await_or_fail(task, what="post-tick approval winner") is ApprovalDecision.ALLOW
            harness.assert_allow_binding_transferred_once()
        elif case == "abort_and_tick_together":
            signal.abort()
            tick.released.set()
            assert await await_or_fail(task, what="simultaneous abort and tick") is ApprovalDecision.ALLOW
            assert await harness.decision() == "allow"
            harness.assert_allow_binding_transferred_once()
        elif case in {"duplicate_same", "duplicate_conflicting"}:
            assert harness.resolve("allow")
            second = "allow" if case == "duplicate_same" else "deny"
            assert not harness.resolve(second)
            tick.released.set()
            assert await await_or_fail(task, what="duplicate approval resolution") is ApprovalDecision.ALLOW
            assert await harness.decision() == "allow"
            harness.assert_allow_binding_transferred_once()
        elif case == "cancel_wait":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, what="canceled approval wait")
            await await_or_fail(signal.wait_finished.wait(), what="canceled abort waiter")
            await await_or_fail(tick.finished.wait(), what="canceled poll tick")
            assert await harness.decision() is None
        elif case == "cancel_drain":
            assert isinstance(signal, _SlowCancelSignal)
            tick.released.set()
            await await_or_fail(signal.drain_entered.wait(), what="pending-task drain")
            task.cancel()
            signal.release_drain.set()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, what="canceled approval drain")
            await await_or_fail(signal.wait_finished.wait(), what="drained abort waiter")
            await await_or_fail(tick.finished.wait(), what="completed poll tick")
            assert await harness.decision() is None
        elif case == "cancel_serve_admission":
            control = _AdmissionControl()
            harness.root.bind_serve_control(control)
            signal.abort()
            await await_or_fail(control.entered.wait(), what="serve abort admission")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, what="canceled serve admission")
            control.released.set()
            await await_or_fail(control.finished.wait(), what="owned serve settlement")
            expected_child = "innermost" if kind == "nested" else None
            expected_reason = "child_cleanup" if kind == "nested" else "turn_canceled"
            assert control.aborts == [
                (harness.call.id, expected_child, expected_reason)
            ]
        elif case == "delegation_gone":
            if kind == "nested":
                assert harness.root.cleanup_delegated("innermost") is None
                tick.released.set()
                assert await await_or_fail(task, what="removed delegation") is ApprovalDecision.DENY
                assert (await harness.root.pending_requests()) == []
                harness.assert_pending_binding_cleared()
            else:
                assert harness.resolve("deny")
                tick.released.set()
                assert await await_or_fail(task, what="root terminal state") is ApprovalDecision.DENY
        else:
            raise AssertionError(case)
    finally:
        tick.released.set()
        signal.abort()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), 10
            )
        harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [2, 3])
async def test_nested_authorize_polling_preserves_innermost_abort_context(
    tmp_path: Path,
    depth: int,
) -> None:
    root_store = ConversationStore(tmp_path / "root", cwd=tmp_path)
    root = ApprovalPolicy(store=root_store)
    control = _AdmissionControl()
    control.released.set()
    root.bind_serve_control(control)
    stores = [root_store]
    policy: ApprovalPolicy | ChildApprovalPolicy = root
    for level in range(1, depth + 1):
        store = ConversationStore(tmp_path / f"child-{level}", cwd=tmp_path)
        stores.append(store)
        policy = ChildApprovalPolicy(
            policy,
            store,
            f"child {level}",
            f"child-{level}",
            child_cwd=tmp_path,
        )
    call = ToolCall("nested", "echo", {"text": "hello"})
    capability = ResolvedCapability(
        "echo", None, True, "text", "hello", ApprovalBinding.NONE, None
    )
    signal = AbortSignal()
    signal.abort()
    try:
        assert isinstance(policy, ChildApprovalPolicy)
        assert await policy.authorize(call, signal, capability=capability) is None
        assert control.aborts == [("nested", f"child-{depth}", "child_cleanup")]
    finally:
        for store in reversed(stores):
            store.close()
