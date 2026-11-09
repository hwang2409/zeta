"""Inbox name grants and approval labels through the real AgentLoop turn."""

from __future__ import annotations

import ast
import asyncio
import contextlib
from pathlib import Path

import pytest
from rich.text import Text

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail, owned_app
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.project_errors import ProjectRegistryBusyError
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import (
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.server.approval_lifecycle import ApprovalLifecycle, Close, Decide, Start
from zeta.server.connections import Admission
from zeta.server.protocol import ApprovalDecision, EventDescriptor
from zeta.skills import SkillCatalog
from zeta.storage_format import open_writer
from zeta.tools.registry import ToolRegistry
from zeta.tui.app import TUIApp

SRC = Path(__file__).parents[1] / "src" / "zeta"


@pytest.fixture(autouse=True)
def _retain_async_test_writer(tmp_path: Path, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("asyncio") is None:
        yield
        return
    with open_writer(tmp_path / ".zeta"):
        yield


def _home(tmp_path: Path):
    home = tmp_path / ".zeta"
    projects = ProjectRegistry(home / "projects")
    alpha = projects.create_project("alpha", "alpha")
    beta = projects.create_project("beta", "beta")
    store = ConversationStore(home / "sessions", session_id="a" * 32, cwd=tmp_path)
    return home, projects, alpha, beta, store


def _send(call_id: str) -> ToolCall:
    arguments = {"action": "send", "project": "beta", "kind": "info", "title": call_id, "body": "b"}
    return ToolCall(call_id, "inbox", arguments)


def _loop(tmp_path: Path, projects, alpha, store, policy, backend) -> AgentLoop:
    registry = ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        project_id=alpha.project_id,
        project_registry=projects,
        inbox_enabled=True,
    )
    return AgentLoop(
        backend,
        store,
        registry=registry,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
        skip_mcp_mount=True,
    )


def _persisted_calls(store: ConversationStore, call_id: str) -> list[ToolCall]:
    return [
        block.tool_call
        for message in store.messages()
        if message.role is MessageRole.ASSISTANT
        for block in message.content
        if isinstance(block, ToolUseContent) and block.tool_call.id == call_id
    ]


def _results(store: ConversationStore, call_id: str) -> list:
    return [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None and message.tool_result.tool_call_id == call_id
    ]


async def _turn(loop: AgentLoop, on_prompt=None) -> list[str]:
    prompts: list[str] = []
    async for event in loop.run_turn("send it", origin=MessageOrigin.USER):
        if event.type is StreamEventType.TOOL_APPROVAL_START and event.tool_call is not None:
            prompts.append(event.tool_call.id)
            if on_prompt is not None:
                on_prompt(event.tool_call)
    return prompts


@pytest.mark.asyncio
async def test_name_grant_sends_first_inbox_call_through_real_turn(tmp_path: Path) -> None:
    home, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(always_allow={"inbox(send beta)"}, default="ask", store=store)
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_send("send-beta")]), ScriptedTurn(content=[TextContent("ok")])]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    try:
        # A prompt would block the turn, so deny it at once and let the asserts report it.
        prompts = await asyncio.wait_for(
            _turn(loop, lambda call: policy.deny(call.id)), 30
        )
        assert prompts == [], "configured name grant asked for approval"
        assert "send-beta" not in store.approval_states()
        assert [call.arguments["project"] for call in _persisted_calls(store, "send-beta")] == [
            beta.project_id
        ]
        results = _results(store, "send-beta")
        assert len(results) == 1 and not results[0].is_error, results
        inbox = ProjectInbox(projects, sessions_root=home / "sessions")
        delivered = (await asyncio.to_thread(inbox.list, beta.project_id))["new"]
        assert [message["title"] for message in delivered] == ["send-beta"]
        assert _call_state(loop, policy, "send-beta") == []
    finally:
        await loop.close()
        store.close()


def _call_state(loop: AgentLoop, policy: ApprovalPolicy, call_id: str) -> list[str]:
    """Name every per-call container on the registry or policy that still holds the call."""
    return [
        f"{type(owner).__name__}.{name}"
        for owner in (loop.tool_registry, policy)
        for name, value in vars(owner).items()
        if isinstance(value, (dict, set, frozenset)) and call_id in value
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny", "abort"])
async def test_settled_inbox_approval_leaves_no_call_state(tmp_path: Path, decision: str) -> None:
    _home_dir, projects, alpha, _beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_send("send-label")]), ScriptedTurn(content=[TextContent("ok")])]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)

    def settle(call: ToolCall) -> None:
        if decision == "allow":
            policy.approve(call.id)
        elif decision == "deny":
            policy.deny(call.id)
        else:
            loop.abort()

    try:
        prompts = await asyncio.wait_for(_turn(loop, settle), 30)
        assert prompts == ["send-label"]
        assert _call_state(loop, policy, "send-label") == [], "label outlived its call"
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel_durable_append", "prepare_approval_raises"])
async def test_prepared_call_that_never_dispatches_leaves_no_call_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _home_dir, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend([ScriptedTurn(tool_calls=[_send("send-fail")])])
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    reached: list[str] = []

    def append(message, approval_requests):
        # The durable append is synchronous, so a cancellation here surfaces from inside it.
        reached.extend(call.arguments["project"] for _id, call, *_rest in approval_requests)
        raise asyncio.CancelledError

    def prepare_approval(call: ToolCall):
        reached.append(call.arguments["project"])
        raise RuntimeError("prepare_approval failed")

    if failure == "cancel_durable_append":
        monkeypatch.setattr(store, "append_message_with_approval_requests", append)
    else:
        monkeypatch.setattr(loop.tool_registry, "prepare_approval", prepare_approval)
    try:
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await asyncio.wait_for(_turn(loop), 30)
        assert reached == [beta.project_id], "the failure must come after call preparation"
        assert _call_state(loop, policy, "send-fail") == [], "label outlived its call"
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_pending_inbox_approval_names_beta_in_tui_card_and_serve_event(
    tmp_path: Path,
) -> None:
    _home_dir, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_send("send-shown")]), ScriptedTurn(content=[TextContent("ok")])]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    admitted: list[dict[str, object]] = []

    def admit(event: object, session_id: object, **fields: object) -> Admission:
        admitted.append(fields)
        return Admission("admitted")

    async def offload(job):
        return job()

    lifecycle = ApprovalLifecycle(
        admit=admit,
        request_turn_failure=lambda: None,
        policy=lambda: policy,
        session_id=lambda: "session",
        parent_store=lambda: None,
        offload=offload,
        is_closing=lambda: False,
    )
    app = TUIApp(loop, provider="codex", model="offline", approval_policy=policy)
    cards: list[str] = []

    async def show(event: StreamEvent) -> None:
        await await_or_fail(
            lifecycle.transition(Start(event, "session", EventDescriptor.APPROVAL_REQUEST)),
            what="serve approval start",
        )
        app._present_pending_approvals()
        # getattr: the lookup tasks do not exist on the pre-fix base, which must reach the assert.
        lookups = getattr(app, "_approval_label_tasks", ())
        await await_or_fail(asyncio.gather(*lookups), what="tui label lookup")
        cards.append(Text.from_ansi(app._transcript.render(120)).plain)
        policy.deny(event.tool_call.id)

    async def drive() -> None:
        async for event in loop.run_turn("send it", origin=MessageOrigin.USER):
            if event.type is StreamEventType.TOOL_APPROVAL_START and event.tool_call is not None:
                await show(event)

    try:
        async with owned_app(app):
            app._full_screen_active = lambda: True
            await asyncio.wait_for(drive(), 30)
            label = f"beta ({beta.project_id})"
            assert [fields.get("label") for fields in admitted] == [label]
            assert len(cards) == 1 and f"allow {label}?" in cards[0], cards
            assert _call_state(loop, policy, "send-shown") == []
    finally:
        await await_or_fail(lifecycle.transition(Close()), what="serve approval close")
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_registry_error_in_label_lookup_keeps_the_approval_usable(
    tmp_path: Path,
) -> None:
    # The label is display-only: a busy or corrupt registry during the lookup must
    # not fail the serve turn, wedge the decision, or leak a TUI task exception.
    _home_dir, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_send("send-busy")]), ScriptedTurn(content=[TextContent("ok")])]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    admitted: list[dict[str, object]] = []

    def admit(event: object, session_id: object, **fields: object) -> Admission:
        admitted.append(fields)
        return Admission("admitted")

    async def offload(job):
        return job()

    lifecycle = ApprovalLifecycle(
        admit=admit,
        request_turn_failure=lambda: None,
        policy=lambda: policy,
        session_id=lambda: "session",
        parent_store=lambda: None,
        offload=offload,
        is_closing=lambda: False,
    )
    app = TUIApp(loop, provider="codex", model="offline", approval_policy=policy)
    cards: list[str] = []
    outcomes: dict[str, object] = {}
    original_show = projects.show_project

    def busy(**_kwargs: object):
        raise ProjectRegistryBusyError()

    async def show(event: StreamEvent) -> None:
        # Preparation already rewrote the project to its id; now break the label read.
        projects.show_project = busy
        try:
            start_error: list[BaseException] = []
            try:
                await asyncio.wait_for(
                    lifecycle.transition(
                        Start(event, "session", EventDescriptor.APPROVAL_REQUEST)
                    ),
                    10,
                )
            except BaseException as exc:  # noqa: BLE001 - the test asserts on the escape
                start_error.append(exc)
            outcomes["start_error"] = start_error
            outcomes["admitted_labels"] = [fields.get("label") for fields in admitted]

            app._present_pending_approvals()
            lookups = getattr(app, "_approval_label_tasks", ())
            outcomes["label_task_results"] = await asyncio.wait_for(
                asyncio.gather(*lookups, return_exceptions=True), 10
            )
            cards.append(Text.from_ansi(app._transcript.render(120)).plain)

            request_id = admitted[-1]["request_id"] if admitted else event.tool_call.id
            try:
                decision, _key = await asyncio.wait_for(
                    lifecycle.transition(Decide(ApprovalDecision.APPROVE, request_id)), 10
                )
                outcomes["decide"] = decision
            except BaseException as exc:  # noqa: BLE001 - APPROVAL_UNAVAILABLE on the base
                outcomes["decide"] = exc
                loop.abort()  # the wedged record never resolves the turn; unblock it
        finally:
            projects.show_project = original_show  # the approved send uses the real registry

    async def drive() -> None:
        async for event in loop.run_turn("send it", origin=MessageOrigin.USER):
            if event.type is StreamEventType.TOOL_APPROVAL_START and event.tool_call is not None:
                await show(event)

    try:
        async with owned_app(app):
            app._full_screen_active = lambda: True
            await asyncio.wait_for(drive(), 30)
            outcomes["call_state"] = _call_state(loop, policy, "send-busy")
            # On the fixed code the accepted decision already drained the record, so
            # the lifecycle holds nothing. On the base the wedged "starting" record is
            # the bug itself and never drains, so do not await Close(): assert instead.
            outcomes["records"] = len(lifecycle._state.records)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(loop.close(), 10)
        store.close()

    assert outcomes["start_error"] == [], (
        f"serve start raised on a registry read error: {outcomes['start_error']!r}"
    )
    assert outcomes["admitted_labels"] == [None], outcomes["admitted_labels"]
    assert not any(
        isinstance(result, BaseException) for result in outcomes["label_task_results"]
    ), outcomes["label_task_results"]
    assert len(cards) == 1, cards
    assert "allow inbox?" in cards[0], cards[0]
    assert f"beta ({beta.project_id})" not in cards[0], cards[0]
    decide = outcomes["decide"]
    assert not isinstance(decide, BaseException), decide
    assert decide["accepted"] is True, decide
    assert outcomes["call_state"] == []
    assert outcomes["records"] == 0, "the decided approval left a record behind"


def test_only_the_inbox_definition_builds_an_inbox_approval_subject(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty(), project_id="p")
    try:
        resolved = [d.name for d in registry.definitions if d.approval_subject_resolver is not None]
    finally:
        asyncio.run(registry.close())
    assert resolved == ["inbox"]
    owners = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            forwarded = isinstance(node, ast.keyword) and isinstance(node.value, ast.Name) and (
                node.value.id == "approval_subject_resolver"
            )
            if isinstance(node, ast.keyword) and node.arg == "approval_subject_resolver" and not forwarded:
                owners.append(path.relative_to(SRC).as_posix())
            if isinstance(node, ast.FunctionDef) and node.name == "_approval_subject":
                owners.append(f"{path.relative_to(SRC).as_posix()}:def")
    assert sorted(owners) == ["tools/inbox/__init__.py", "tools/inbox/__init__.py:def"]



@pytest.mark.asyncio
async def test_depth_two_worker_has_no_inbox_so_child_grants_never_apply(tmp_path: Path) -> None:
    # Children never get inbox (agent/presets.py hard-excludes it), so a nested
    # ChildApprovalPolicy never prepares an inbox call. Pin that through real loops.
    home, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(always_allow={"inbox(send beta)", "agent"}, default="deny", store=store)
    agent = {"prompt": "go", "description": "lead"}
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("lead", "agent", dict(agent))]),
            ScriptedTurn(tool_calls=[ToolCall("worker", "agent", dict(agent))]),
            ScriptedTurn(tool_calls=[_send("nested-send")]),
            ScriptedTurn(content=[TextContent("worker done")]),
            ScriptedTurn(content=[TextContent("lead done")]),
            ScriptedTurn(content=[TextContent("ok")]),
        ]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    try:
        prompts = await asyncio.wait_for(_turn(loop), 30)
        assert prompts == []
        worker_request = backend.calls[2]
        assert "inbox" not in {tool["name"] for tool in worker_request[1]}
        worker_result = backend.calls[3][0][-1].tool_result
        assert worker_result is not None and worker_result.tool_call_id == "nested-send"
        assert worker_result.is_error and worker_result.content == "unknown tool: inbox"
        nested = backend.calls[4][0][-1].tool_result.structured_content
        assert nested["depth"] == 2
        grandchild_dir = Path(nested["child_session_path"])
        grandchild = ConversationStore(grandchild_dir.parent, session_id=grandchild_dir.name)
        try:
            assert grandchild.approval_states() == {}
        finally:
            grandchild.close()
        assert policy.pending_requests() == []
        inbox = ProjectInbox(projects, sessions_root=home / "sessions")
        assert (await asyncio.to_thread(inbox.list, beta.project_id))["new"] == []
    finally:
        await loop.close()
        store.close()
