from __future__ import annotations

import asyncio
import inspect
import io
import json

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,

    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,

)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.runtime.loop import nested_tools
from zeta.runtime.loop.agent import AgentLoop
from zeta.server.event_contract import tool_end
from zeta.server.handlers.history_projection import _history_group, history_projection
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy
from zeta.tui.checkpoints import render_replayed_message
from zeta.tui.transcript.transcript import TranscriptWidget
from zeta.tui.transcript.transcript_presenter import TranscriptPresenter

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")


async def run_script(tmp_path, script, *, policy=None, delegated=False, configure=None, on_event=None):
    skills = SkillCatalog.empty()
    store = ConversationStore(tmp_path / "session", session_id="test")
    if delegated:
        policy = ChildApprovalPolicy(policy, store, "proof", "code-child", child_cwd=tmp_path)
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
                reply = on_event(event)
                if inspect.isawaitable(reply):
                    await await_or_fail(reply, what="test event handler")

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


@pytest.mark.parametrize("delegated", [False, True], ids=["direct", "delegated"])
async def test_dependent_approvals(tmp_path, delegated):
    (tmp_path / "pointer").write_text("target")
    (tmp_path / "target").write_text("selected")
    policy = ApprovalPolicy()
    requests = []

    def approve(event):
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            requests.append(event.tool_call)
            assert policy.resolve(event.tool_call.id, "allow", child_instance_id="code-child" if delegated else None)

    result, events, calls = await run_script(tmp_path, '''
        const first = await tools.read({path: 'pointer'});
        const second = await tools.read({path: first.structuredContent.text ?? first.content.trim()});
        output(second.content);
        ''', policy=policy, delegated=delegated, on_event=approve)
    assert 'selected' in result.content, result.content
    assert [r.arguments['path'] for r in requests] == ['pointer', 'target']
    ends = [tool_end(event) for event in events if event.type is StreamEventType.TOOL_EXECUTION_END]
    assert [row['data'].get('code_call_index') for row in ends] == [1, 2, None]
    messages, _ = calls[1]
    assert ':code:' not in json.dumps([message.to_dict() for message in messages])
    anthropic = build_messages_payload(messages, [], model="claude", max_tokens=16384)
    codex = build_responses_payload(messages, [], model="gpt")
    for payload in (anthropic, codex):
        wire = json.dumps(payload)
        assert ':code:' not in wire and 'selected' in wire
        assert 'code_mode' not in wire

    console = Console(file=io.StringIO(), width=100, no_color=True, force_terminal=False)
    live = TranscriptPresenter(TranscriptWidget(), console, lambda: False, console.print)
    for event in events:
        live.handle_tool_event(event, aborted=False)
    live.commit_tool_region()
    live_text = console.file.getvalue()
    assert '2 calls' in live_text and 'const first' not in live_text, live_text

    reopened = ConversationStore(tmp_path / "session", session_id=next((tmp_path / "session").glob("*/conversation.jsonl")).parent.name)
    try:
        history = history_projection(reopened, fidelity=True)
        rows = []
        for i in range(len(history.entry_groups)):
            group, _, _, _ = _history_group(history, i, end=i + 1, include_delivery_id=False,
                                             fidelity=True, content_read=False, approval_origin=False,
                                             backward=False, validator=None)
            rows.extend(group)
        assert {row['code_call_index'] for row in rows if 'code_call_index' in row} == {1, 2}
        assert any((row.get('tool_result') or {}).get('structured_content', {}).get('code_mode')
                   for row in rows if (row.get('tool_result') or {}).get('structured_content'))

        console = Console(file=io.StringIO(), width=100, no_color=True, force_terminal=False)
        replay = TranscriptPresenter(TranscriptWidget(), console, lambda: False, console.print)
        tools = {}
        for entry in map(reopened.entry, reopened.branch_refs()):
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            render_replayed_message(message, presenter=replay, print_unit=console.print,
                                    tool_calls=tools, replay_tool_results=True)
        replay_text = console.file.getvalue()
        assert '2 calls' in replay_text and 'const first' not in replay_text, replay_text
        assert 'pointer' in live_text and 'pointer' in replay_text
    finally:
        reopened.close()


@pytest.mark.parametrize("delegated", [False, True], ids=["direct", "delegated"])
async def test_inner_deny_is_catchable(tmp_path, delegated):
    policy = ApprovalPolicy()
    target = tmp_path / "blocked"

    def on_event(event):
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            assert policy.resolve(event.tool_call.id, "deny", child_instance_id="code-child" if delegated else None)

    result, events, _ = await run_script(tmp_path, """
        let value = 'unexpected success';
        try { await tools.write({path: 'blocked', content: 'denied'}); }
        catch (error) { value = error.name; }
        output(value);
        """, policy=policy, delegated=delegated, on_event=on_event)
    assert result.content == '"ToolDenied"'
    assert not target.exists()
    assert sum(event.type is StreamEventType.TOOL_EXECUTION_END and
               event.tool_call.id == "outer:code:1" for event in events) == 1


async def test_unexposed_controls(tmp_path):
    names = ["finish", "ask_parent", "agent", "agent_status", "agent_output", "agent_send",
             "agent_cancel", "agent_handoff", "request_attention", "resolve_attention",
             "list_attention", "inbox", "session_push", "task_input", "task_kill", "todo",
             "skill", "mcp_discover", "code", "unknown", "read"]

    def configure(registry):
        registry.unregister("read")

    result, events, _ = await run_script(tmp_path, """
        const errors = [];
        for (const name of NAMES) {
          try { await tools.call(name, {}); errors.push('unexpected success'); }
          catch (error) { errors.push(error.message); }
        }
        output(errors);
        """.replace("NAMES", json.dumps(names)), configure=configure)
    assert json.loads(result.content) == [f"tool {name!r} is not exposed to code mode" for name in names]
    assert not [event for event in events if event.data.get("code_parent_id") == "outer"]


async def test_restart_does_not_resume_inner_call(tmp_path):
    root = tmp_path / "session"
    store = ConversationStore(root, session_id="interrupted")
    outer = ToolCall("outer", "code", {"script": "output(null)"})
    inner = ToolCall("outer:code:1", "write", {"path": "never", "content": "changed"})
    try:
        store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(outer)]))
        store.append_approval_request(inner.id, inner)
    finally:
        store.close()
    store = ConversationStore(root, session_id="interrupted")
    skills = SkillCatalog.empty()
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=skills,
                            approval_policy=ApprovalPolicy(), code_mode_enabled=True)
    loop = AgentLoop(FakeBackend([]), store, registry=registry, skill_catalog=skills, skip_mcp_mount=True)
    try:
        await await_or_fail(loop.activate(), what="recover interrupted code invocation")
        assert store.approval_state_now(inner.id)[1] == "abort"
        assert store.tool_result(inner.id).is_canceled
        assert (await await_or_fail(loop.resume_pending_tool(inner.id), what="refuse standalone nested resume")).is_canceled
        assert not (tmp_path / "never").exists()
        messages = await await_or_fail(loop.context_assembler.assemble(), what="assemble recovered conversation")
        assert inner.id not in json.dumps([message.to_dict() for message in messages])
    finally:
        await await_or_fail(loop.close(), what="close recovered owner")
        store.close()


@pytest.mark.parametrize("delegated", [False, True], ids=["direct", "delegated"])
async def test_cancel_pending_inner_approval(tmp_path, delegated):
    policy = ApprovalPolicy()
    awaiting_approval = asyncio.Event()
    never = asyncio.Event()

    async def hold(event):
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            awaiting_approval.set()
            await await_or_fail(never.wait(), what="hold unsettled approval until cancellation")

    task = asyncio.create_task(run_script(tmp_path, "await tools.write({path: 'never', content: 'never'});",
                                         policy=policy, delegated=delegated, on_event=hold))
    try:
        await await_or_fail(awaiting_approval.wait(), what="inner approval is durable")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(task, what="cancel code invocation and join guest")
    store = ConversationStore(tmp_path / "session", session_id="test")
    try:
        assert store.approval_state_now("outer:code:1")[1] == "abort"
        assert store.tool_result("outer:code:1").is_canceled
        assert not (tmp_path / "never").exists()
    finally:
        store.close()
