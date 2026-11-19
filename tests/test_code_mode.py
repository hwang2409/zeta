from __future__ import annotations

import asyncio
import inspect
import io
import json
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.core.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRule,
)
from zeta.core.store import ConversationStore
from zeta.core.tool_dispatch import ToolDispatchCoordinator
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
from zeta.runtime.execution import NestedToolParent
from zeta.runtime.loop import nested_tools
from zeta.runtime.loop.agent import AgentLoop
from zeta.server.event_contract import tool_end
from zeta.server.handlers.history_projection import _history_group, history_projection
from zeta.skills import SkillCatalog
from zeta.submission.pipeline import SubmissionPipeline
from zeta.tools import ToolRegistry
from zeta.tools import registry as registry_module
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
        if configure is not None:
            configure(registry)
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


@pytest.mark.parametrize("packaged", [False, True], ids=["dynamic", "packaged"])
async def test_unlisted_tool_has_no_authority(tmp_path, monkeypatch, packaged):
    hits = []
    name = "slice_probe"

    async def handler(arguments, signal):
        hits.append(arguments)
        return "ran"

    def register(registry):
        registry.register(name, handler)

    if packaged:
        fake = ModuleType("zeta.tools.slice_probe")
        fake.register = register
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
        found = registry_module._discover_tool_modules()
        # test-contract: seam inject one more packaged module without changing the allow-list.
        monkeypatch.setattr(registry_module, "_discover_tool_modules", lambda: [*found, fake.__name__])

    def configure(registry):
        if not packaged:
            register(registry)
        origin = registry.definition_reference(name).origin
        assert origin is (registry_module.ToolOrigin.PACKAGED_BUILTIN if packaged
                          else registry_module.ToolOrigin.DYNAMIC)
        assert any(schema["name"] == name for schema in registry.schemas)

    result, events, _ = await run_script(tmp_path, """
        let value = 'ran';
        try { await tools.call('slice_probe', {}); }
        catch (error) { value = error.message; }
        output(value);
        """, configure=configure)
    assert result.content == json.dumps("tool 'slice_probe' is not exposed to code mode")
    assert not hits
    assert not [event for event in events if event.data.get("code_parent_id") == "outer"]


@asynccontextmanager
async def nested_owner(tmp_path, *, delegated=False):
    policy = ApprovalPolicy()
    store = ConversationStore(tmp_path / "session", session_id="table")
    outer = ToolCall("outer", "code", {"script": "output(null)"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(outer)]))
    tools_policy = ChildApprovalPolicy(policy, store, "proof", "code-child", child_cwd=tmp_path) if delegated else policy
    skills = SkillCatalog.empty()
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=skills,
                            approval_policy=tools_policy, code_mode_enabled=True)
    loop = AgentLoop(FakeBackend([]), store, registry=registry, skill_catalog=skills, skip_mcp_mount=True)
    try:
        yield loop, policy, outer
    finally:
        await await_or_fail(loop.close(), what="close section-4 owner")
        store.close()


def assert_settled_once(store, call, decision, *, canceled=False):
    entries = [store.entry(ref) for ref in store.branch_refs()]
    decisions = [entry.data["decision"] for entry in entries
                 if entry.type == "approval_resolution" and entry.data["request_id"] == call.id]
    results = [result for entry in entries if entry.type == "message"
               if (result := Message.from_dict(entry.data["message"]).tool_result) is not None
               and result.tool_call_id == call.id]
    assert decisions == [decision]
    assert len(results) == 1 and results[0].is_canceled is canceled


@pytest.mark.parametrize("mode", ["direct-always", "delegated-always", "parent-grant"])
async def test_nested_grants(tmp_path, mode):
    async with nested_owner(tmp_path, delegated=mode != "direct-always") as (loop, policy, outer):
        ask = asyncio.Event()
        events = []
        messages = []
        host = SimpleNamespace(loop=loop, _approval_policy=policy, _print_system=messages.append,
                               _present_pending_approvals=lambda: None, _invalidate_prompt=lambda: None)
        pipeline = SubmissionPipeline(host)
        registry = loop.tool_registry

        def publish(event):
            events.append(event.type)
            if event.type is StreamEventType.TOOL_APPROVAL_START:
                ask.set()

        if mode == "parent-grant":
            policy.remember_allow(ApprovalRequest("prior", ToolCall("prior", "write", {"path": "other"})))
        execute = nested_tools.LoopNestedToolExecutor(loop, outer, registry.abort_signal, publish, None,
                                                      coordinator=ToolDispatchCoordinator(8))
        call = ToolCall("outer:code:1", "write", {"path": "first", "content": "written"})
        task = asyncio.create_task(execute(call, NestedToolParent(outer.id, 1)))
        try:
            if mode != "parent-grant":
                await await_or_fail(ask.wait(), what="inner write approval")
                await await_or_fail(pipeline.approval_command(ApprovalDecision.ALLOW, None, always=True),
                                    what="resolve the /always command")
                if mode == "delegated-always":
                    assert policy.session_allow == frozenset()
                    assert any("cannot always allow delegated approvals" in message for message in messages)
                    assert loop.store.approval_state_now(call.id)[1] is None
                    assert policy.approve(call.id, child_instance_id="code-child")
            assert not (await await_or_fail(task, what="inner write settles")).is_error
            assert (tmp_path / "first").read_text() == "written"
            if mode == "parent-grant":
                assert StreamEventType.TOOL_APPROVAL_START not in events
                assert loop.store.approval_state_now(call.id) is None
            else:
                assert_settled_once(loop.store, call, "allow")
            if mode == "direct-always":
                assert policy.session_allow == {ApprovalRule("write")}
                assert policy.decide(registry.resolve_call("read", {"path": "first"})) is ApprovalDecision.ASK
                again = ToolCall("outer:code:2", "write", {"path": "second", "content": "written"})
                assert not (await await_or_fail(execute(again, NestedToolParent(outer.id, 2)),
                                                what="repeat write uses its inner grant")).is_error
                assert events.count(StreamEventType.TOOL_APPROVAL_START) == 1
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(task, what="drain inner grant task")
            await await_or_fail(pipeline.close(), what="close grant command pipeline")


@pytest.mark.parametrize("case", ["duplicate", "admission", "settlement"])
async def test_nested_overlap(tmp_path, monkeypatch, case):
    async with nested_owner(tmp_path) as (loop, policy, outer):
        registry = loop.tool_registry
        first_ask, second_entered = asyncio.Event(), asyncio.Event()
        settling, release = asyncio.Event(), asyncio.Event()
        in_handler, release_handler = asyncio.Event(), asyncio.Event()
        entered = []
        events = []
        original_abort = registry.abort_approval

        async def pause_settlement(call):
            settling.set()
            await await_or_fail(release.wait(), what="allow durable settlement to continue")
            return await original_abort(call)

        if case == "settlement":
            # test-contract: seam hold settlement at the real registry's durable abort boundary.
            monkeypatch.setattr(registry, "abort_approval", pause_settlement)

        async def hook(name, arguments):
            entered.append((name, arguments["path"]))
            if case == "duplicate":
                in_handler.set()
                await await_or_fail(release_handler.wait(), what="let admitted call finish")
            return True

        def publish(event):
            events.append((event.type, event.tool_call.id))
            if event.type is StreamEventType.TOOL_APPROVAL_START:
                first_ask.set()

        registry.set_pre_execute_hook(hook)
        execute = nested_tools.LoopNestedToolExecutor(loop, outer, registry.abort_signal, publish, None,
                                                      coordinator=ToolDispatchCoordinator(8))
        first = ToolCall("outer:code:1", "write", {"path": "first", "content": "written"})
        second = ToolCall("outer:code:2", "write", {"path": "second", "content": "written"})

        async def queued():
            second_entered.set()
            return await execute(second, NestedToolParent(outer.id, 2))

        task = asyncio.create_task(execute(first, NestedToolParent(outer.id, 1)))
        waiting = None
        duplicate = None
        try:
            await await_or_fail(first_ask.wait(), what="first call holds admission through its approval")
            waiting = asyncio.create_task(queued())
            await await_or_fail(second_entered.wait(), what="second call reaches dispatch queue")
            if case == "duplicate":
                duplicate = asyncio.create_task(execute(second, NestedToolParent(outer.id, 2)))
                assert (await await_or_fail(duplicate, what="reject duplicate of queued call")).content == "duplicate_tool_call"
                assert (await await_or_fail(execute(first, NestedToolParent(outer.id, 1)),
                                            what="reject duplicate while approval holds admission")).content == "duplicate_tool_call"
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(waiting, what="drain canceled admission waiter")
            assert loop.store.approval_state_now(second.id) is None
            assert not any(call_id == second.id for _, call_id in events)
            assert not (tmp_path / "second").exists()
            if case == "settlement":
                task.cancel()
                await await_or_fail(settling.wait(), what="canceled admitted call reaches settlement")
                task.cancel()
                # Ready callbacks run in order: give the second cancellation its turn before releasing settlement.
                delivered = asyncio.Event()
                asyncio.get_running_loop().call_soon(delivered.set)
                await await_or_fail(delivered.wait(), what="deliver cancellation while settlement is held")
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(task, what="wait for shielded settlement before cancellation returns")
                assert_settled_once(loop.store, first, "abort", canceled=True)
                assert entered == []
                return
            assert policy.approve(first.id)
            if case == "duplicate":
                await await_or_fail(in_handler.wait(), what="call is running after approval")
                assert (await await_or_fail(execute(first, NestedToolParent(outer.id, 1)),
                                            what="reject duplicate during execution")).content == "duplicate_tool_call"
                release_handler.set()
            assert not (await await_or_fail(task, what="first call completes after admission test")).is_error
            if case == "duplicate":
                assert (await await_or_fail(execute(first, NestedToolParent(outer.id, 1)),
                                            what="reject duplicate of completed call")).content == "duplicate_tool_call"
            assert entered == [("write", "first")]
            assert events.count((StreamEventType.TOOL_EXECUTION_START, first.id)) == 1
            assert_settled_once(loop.store, first, "allow")
        finally:
            release.set()
            release_handler.set()
            for pending in (task, waiting, duplicate):
                if pending is not None and not pending.done():
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await await_or_fail(pending, what="drain overlap task")


@pytest.mark.parametrize("failure", [
    StreamEventType.TOOL_APPROVAL_START, StreamEventType.TOOL_APPROVAL_END,
    StreamEventType.TOOL_EXECUTION_START, StreamEventType.TOOL_EXECUTION_UPDATE,
    StreamEventType.TOOL_EXECUTION_END,
])
async def test_nested_publication_failure(tmp_path, failure):
    async with nested_owner(tmp_path) as (loop, policy, outer):
        registry = loop.tool_registry
        call = ToolCall("outer:code:1", "bash", {"command": "printf x >> receipt; printf progress", "cwd": str(tmp_path)})
        asked = asyncio.Event()
        events = []

        def publish(event):
            events.append(event.type)
            if event.type is StreamEventType.TOOL_APPROVAL_START:
                asked.set()
            if event.type is failure:
                raise RuntimeError(f"{failure.value} failed")
            if event.type is StreamEventType.TOOL_EXECUTION_END:
                raise RuntimeError("later failure must not replace the first")

        execute = nested_tools.LoopNestedToolExecutor(loop, outer, registry.abort_signal, publish, None,
                                                      coordinator=ToolDispatchCoordinator(8))
        task = asyncio.create_task(execute(call, NestedToolParent(outer.id, 1)))
        try:
            await await_or_fail(asked.wait(), what="approval was published before the injected error")
            assert policy.approve(call.id)
            result = await await_or_fail(task, what="settle the call after publication failure")
            assert_settled_once(loop.store, call, "allow")
            assert (tmp_path / "receipt").read_text() == "x"
            assert events.count(StreamEventType.TOOL_EXECUTION_END) == 1
            assert result.is_error and result.content == f"tool lifecycle publication failed: {failure.value} failed"
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(task, what="drain publication-fault call")


async def test_parallel_limit_and_exclusive_fence(tmp_path):
    first_wave, ninth, exclusive = (asyncio.Event() for _ in range(3))
    release_first, release_reads, release_write = (asyncio.Event() for _ in range(3))
    entered = []
    completed_reads = set()
    predecessors_at_write = []
    following_at_write_end = []

    async def hook(name, arguments):
        if name == 'code':
            return True
        path = str(arguments['path']).rsplit('/', 1)[-1]
        entered.append(path)
        if name == 'write':
            predecessors_at_write.extend(completed_reads)
            exclusive.set()
            await await_or_fail(release_write.wait(), what='hold the exclusive write')
            following_at_write_end.extend(entered)
        elif path != 'after':
            if len(entered) == 8:
                first_wave.set()
            if path == 'r9':
                ninth.set()
            gate = release_first if path == 'r1' else release_reads
            await await_or_fail(gate.wait(), what=f'release parallel read {path}')
            completed_reads.add(path)
        return True

    paths = [f'r{i}' for i in range(1, 10)] + ['after']
    for path in paths:
        (tmp_path / path).write_text(path)
    task = asyncio.create_task(run_script(tmp_path, '''
        const replies = await Promise.all([
            ...Array.from({length: 9}, (_, i) => tools.read({path: `r${i+1}`})),
            tools.write({path: 'fence', content: 'exclusive'}),
            tools.read({path: 'after'}),
        ]);
        output(replies.map(r => r.structuredContent?.text ?? r.content));
    ''', configure=lambda registry: registry.set_pre_execute_hook(hook)))
    try:
        await await_or_fail(first_wave.wait(), what='eight reads run at the same time')
        assert sorted(entered) == sorted(paths[:8])
        release_first.set()
        await await_or_fail(ninth.wait(), what='ninth read starts after one slot is released')
        # If the write bypasses its predecessors, it starts as soon as another slot opens.
        release_reads.set()
        await await_or_fail(exclusive.wait(), what='exclusive write starts after reads finish')
        assert set(predecessors_at_write) == set(paths[:9])
        release_write.set()
        result, events, _ = await await_or_fail(task, what='settle all calls in the parallel batch')
        assert not result.is_error, result.content
        assert 'after' not in following_at_write_end
        replies = json.loads(result.content)
        assert all(path in replies[index] for index, path in enumerate(paths[:9]))
        assert 'after' in replies[10]
        ends = [tool_end(event) for event in events if event.type is StreamEventType.TOOL_EXECUTION_END
                and event.data.get('code_parent_id') == 'outer']
        assert {row['data']['code_call_index'] for row in ends} == set(range(1, 12))
        assert len(ends) == 11
        console = Console(file=io.StringIO(), width=120, no_color=True, force_terminal=False)
        presenter = TranscriptPresenter(TranscriptWidget(), console, lambda: False, console.print)
        for event in events:
            presenter.handle_tool_event(event, aborted=False)
        presenter.commit_tool_region()
        rendered = console.file.getvalue()
        assert '11 calls' in rendered and 'fence' in rendered and 'after' in rendered
    finally:
        for gate in (release_first, release_reads, release_write):
            gate.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, what='drain the parallel code invocation')


async def test_exclusive_inner_call_completes_in_outer_slot(tmp_path):
    import threading

    try:
        result, _, _ = await run_script(tmp_path, "await tools.write({path: 'receipt', content: 'ok'}); output(true);")
        assert result.content == 'true'
        assert (tmp_path / 'receipt').read_text() == 'ok'
    finally:
        assert not any(thread.name == 'zeta-code-guest' for thread in threading.enumerate())


@pytest.mark.parametrize('delegated', [False, True], ids=['direct', 'delegated'])
async def test_parallel_asks_are_independent(tmp_path, delegated):
    policy = ApprovalPolicy()
    requests = []
    (tmp_path / 'deny').write_text('private')
    (tmp_path / 'allow').write_text('public')

    def decide(event):
        if event.type is not StreamEventType.TOOL_APPROVAL_START:
            return
        requests.append(event.tool_call)
        if len(requests) == 2:
            child = 'code-child' if delegated else None
            for call in requests:
                decision = 'deny' if call.id == 'outer:code:1' else 'allow'
                assert policy.resolve(call.id, decision, child_instance_id=child)

    result, events, _ = await run_script(tmp_path, '''
        const replies = await Promise.allSettled([tools.read({path: 'deny'}), tools.read({path: 'allow'})]);
        output([replies[0].reason.name, replies[1].value.content]);
    ''', policy=policy, delegated=delegated, on_event=decide)
    assert not result.is_error, result.content
    denied, allowed = json.loads(result.content)
    assert denied == 'ToolDenied' and 'public' in allowed
    assert sorted(call.id for call in requests) == ['outer:code:1', 'outer:code:2']
    started = [event.tool_call.id for event in events if event.type is StreamEventType.TOOL_EXECUTION_START]
    assert started == ['outer', 'outer:code:2']
    if delegated:
        with ConversationStore(tmp_path / 'session', session_id='test') as store:
            denied_call, allowed_call = sorted(requests, key=lambda call: call.id)
            assert_settled_once(store, denied_call, 'deny')
            assert_settled_once(store, allowed_call, 'allow')


async def test_abort_parallel_asks_settles_started_calls_only(tmp_path):
    policy = ApprovalPolicy()
    requests = []
    registry = None
    head = "tools.read({path: 'first'}), tools.read({path: 'second'})"

    def configure(owner):
        nonlocal registry
        registry = owner

    def abort(event):
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            requests.append(event.tool_call)
            if len(requests) == 2:
                registry.abort()

    result, events, _ = await run_script(tmp_path, f'''
        await Promise.allSettled([{head}, tools.write({{path: 'never', content: 'no'}}),
                                  tools.read({{path: 'later'}})]);
        output(true);
    ''', policy=policy, delegated=True, configure=configure, on_event=abort)
    assert result.is_canceled
    assert sorted(call.id for call in requests) == ['outer:code:1', 'outer:code:2']
    assert sorted(event.tool_call.id for event in events if event.type is StreamEventType.TOOL_EXECUTION_END
                  and event.data.get('code_parent_id') == 'outer') == ['outer:code:1', 'outer:code:2']
    with ConversationStore(tmp_path / 'session', session_id='test') as store:
        for call in requests:
            assert_settled_once(store, call, 'abort', canceled=True)
        assert store.approval_state_now('outer:code:3') is None
    assert not (tmp_path / 'never').exists()


@pytest.mark.parametrize('invariant', ['projection', 'history-index'])
async def test_code_replay_retains_audit_not_projection(tmp_path, invariant):
    import re

    from tests.support.serve_close import close_serve
    from tests.test_server import _connect, _request
    from zeta.server import ZetaServer

    sentinel = 'UNSELECTED_REPLAY_FIXTURE_' + 'x' * 2022
    (tmp_path / 'unselected').write_text(sentinel)
    (tmp_path / 'selected').write_text('selected')
    (tmp_path / 'settings.toml').write_text('[code_mode]\nenabled = true\n')
    script = '''await tools.read({path: 'unselected'});
const selected = await tools.read({path: 'selected'});
output(selected.content);'''
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall('outer', 'code', {'script': script})]),
        ScriptedTurn(content=[TextContent('finished')]),
    ])
    server = ZetaServer(home=tmp_path, port=0, provider='codex',
                        backend_factory=lambda provider, model, home: (backend, model or 'offline'))
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, 'hello', {'protocol_version': '1.2', 'features': [
            'session_history_fidelity', 'session_history_backward', 'session_cwd',
        ]})
        session = (await _request(reader, writer, 2, 'new_session', {'provider': 'codex', 'cwd': str(tmp_path)}))[-1]['result']['session']
        loop = server.runtime.loop
        store = loop.store
        policy = loop.tool_registry.approval_policy

        async def run():
            async for event in loop.run_turn('read both', origin=MessageOrigin.USER):
                if event.type is StreamEventType.TOOL_APPROVAL_START:
                    assert policy.resolve(event.tool_call.id, 'allow')

        await await_or_fail(run(), what='complete the scripted code turn with real approvals')
        outer = store.tool_result('outer')
        assert not outer.is_error and outer.content == '"selected"'
        console = Console(file=io.StringIO(), width=100, no_color=True, force_terminal=False)
        widget = TranscriptWidget()
        replay = TranscriptPresenter(widget, console, lambda: True, console.print)
        tool_calls = {}
        for ref in store.branch_refs():
            entry = store.entry(ref)
            if entry.type == 'message':
                render_replayed_message(Message.from_dict(entry.data['message']), presenter=replay,
                                        print_unit=console.print, tool_calls=tool_calls)
        def render():
            return re.sub(r'\x1b\[[0-9;]*m', '', widget.render(100))

        collapsed = render()
        assert 'const selected' not in collapsed and '2 calls' in collapsed
        assert widget.toggle_latest_agent() and 'const selected' in render()
        assert 'UNSELECTED_REPLAY_FIXTURE_' in collapsed  # durable approved read, not the outer projection
        page = (await _request(reader, writer, 3, 'session_history', {'session_id': session['session_id'], 'before': None}))[-1]['result']
        results = {r['tool_result']['tool_call_id']: r for r in page['messages'] if r.get('tool_result')}
        outer_wire = results['outer']['tool_result']
        assert outer_wire['content'] == outer.content
        assert sentinel in results['outer:code:1']['tool_result']['content']
        if invariant == 'projection':
            for projection in (outer.structured_content, outer_wire['structured_content']):
                assert 'UNSELECTED_REPLAY_FIXTURE_' not in json.dumps(projection), 'raw inner output in outer projection'
        else:
            indexed = [results['outer:code:1'], results['outer:code:2']]
            assert [r.get('code_call_index') for r in indexed] == [1, 2], 'serve lost nested call indexes'
            assert [r.get('code_parent_id') for r in indexed] == ['outer', 'outer']
    finally:
        await close_serve(server, writer)
