"""Inbox name grants and approval labels through the real AgentLoop turn."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import (
    MessageOrigin,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.storage_format import open_writer
from zeta.tools.registry import ToolRegistry

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
        call = _persisted_calls(store, "send-beta")[0]
        assert policy.request_from_store(call.id, call).label is None
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny", "abort"])
async def test_settled_inbox_approval_drops_its_display_label(
    tmp_path: Path, decision: str
) -> None:
    _home_dir, projects, alpha, beta, store = await asyncio.to_thread(_home, tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_send("send-label")]), ScriptedTurn(content=[TextContent("ok")])]
    )
    loop = _loop(tmp_path, projects, alpha, store, policy, backend)
    pending_labels: list[str | None] = []

    def settle(call: ToolCall) -> None:
        pending_labels.append(policy.request_from_store(call.id, call).label)
        if decision == "allow":
            policy.approve(call.id)
        elif decision == "deny":
            policy.deny(call.id)
        else:
            loop.abort()

    try:
        prompts = await asyncio.wait_for(_turn(loop, settle), 30)
        assert prompts == ["send-label"]
        assert pending_labels == [f"beta ({beta.project_id})"]
        call = _persisted_calls(store, "send-label")[0]
        assert policy.request_from_store(call.id, call).label is None, "label outlived its call"
    finally:
        await loop.close()
        store.close()


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
