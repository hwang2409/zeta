from __future__ import annotations

import json

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, StreamEventType, TextContent, ToolCall
from zeta.runtime.loop import nested_tools
from zeta.runtime.loop.agent import AgentLoop
from zeta.server.event_contract import tool_end
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")


async def run_script(tmp_path, script, *, policy=None, configure=None, on_event=None):
    skills = SkillCatalog.empty()
    store = ConversationStore(tmp_path / "session")
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=skills,
                            approval_policy=policy, code_mode_enabled=True)
    if configure is not None:
        configure(registry)
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall("outer", "code", {"script": script})]),
        ScriptedTurn(content=[TextContent("finished")]),
    ])
    loop = AgentLoop(backend, store, registry=registry, skill_catalog=skills, skip_mcp_mount=True)
    events = []

    async def collect():
        async for event in loop.run_turn("run the script", origin=MessageOrigin.USER):
            events.append(event)
            if on_event is not None:
                on_event(event)

    try:
        await await_or_fail(collect(), what="code mode turn settles")
        outer = store.tool_result("outer")
        assert outer is not None
        return outer, events, backend.calls
    finally:
        await await_or_fail(loop.close(), what="close code mode loop")
        store.close()


@pytest.mark.parametrize("inject_snapshot", [False, True], ids=["production", "executor"])
async def test_builtin_origin_check(tmp_path, monkeypatch, inject_snapshot):
    hits = []

    async def replacement(arguments, signal):
        hits.append(arguments)
        return "replacement ran"

    def configure(registry):
        registry.register("bash", replacement)

    if inject_snapshot:
        monkeypatch.setattr(nested_tools, "authority_snapshot", lambda registry, _: {
            "bash": registry.definition_reference("bash"),
        })
    result, events, _ = await run_script(tmp_path, '''
        let state = 'ran';
        try { await tools.call('bash', {command: 'printf original'}); }
        catch (error) { state = error.message; }
        output(state);
        ''', configure=configure)
    assert result.content == json.dumps("tool 'bash' is not exposed to code mode: active definition is not a packaged built-in")
    assert not hits
    assert not [event for event in events if event.data.get("code_parent_id") == "outer"]


async def test_dependent_direct_approvals(tmp_path):
    (tmp_path / "pointer").write_text("target")
    (tmp_path / "target").write_text("selected")
    policy = ApprovalPolicy()
    requests = []

    def approve(event):
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            requests.append(event.tool_call)
            assert policy.resolve(event.tool_call.id, "allow")

    result, events, calls = await run_script(tmp_path, '''
        const first = await tools.read({path: 'pointer'});
        const second = await tools.read({path: first.structuredContent.text ?? first.content.trim()});
        output(second.content);
        ''', policy=policy, on_event=approve)
    assert 'selected' in result.content, result.content
    assert [r.arguments['path'] for r in requests] == ['pointer', 'target']
    ends = [tool_end(event) for event in events if event.type is StreamEventType.TOOL_EXECUTION_END]
    assert [row['data'].get('code_call_index') for row in ends] == [1, 2, None]
    messages, _ = calls[1]
    assert ':code:' not in json.dumps([message.to_dict() for message in messages])
