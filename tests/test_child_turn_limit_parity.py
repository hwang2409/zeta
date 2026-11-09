from __future__ import annotations

import asyncio
import hashlib
import json
import re
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

import zeta.agent.runner as agent_runner_module
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import owned_app
from zeta.agent.presets import AgentPreset
from zeta.agent.runner import run_finish_gate as real_run_finish_gate
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.server.ergonomics import _history_entry_rows
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent import _read_agent_status
from zeta.tui.app import TUIApp


async def _unset_child_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    nested_started = asyncio.Event()
    release_nested = asyncio.Event()

    class NestedBackend(FakeBackend):
        @owned_completion
        async def complete(self, messages, tool_schemas):
            nested_started.set()
            await asyncio.wait_for(release_nested.wait(), timeout=10)
            async for event in FakeBackend.complete.__wrapped__(
                self, messages, tool_schemas
            ):
                yield event

    child_backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "todo-1",
                        "todo",
                        {"items": [{"content": "step", "status": "completed"}]},
                    )
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "nested-1",
                        "agent",
                        {
                            "prompt": "nested work",
                            "description": "nested child",
                            "agent_type": "nested",
                            "background": True,
                        },
                    )
                ]
            ),
            ScriptedTurn([]),
            ScriptedTurn([TextContent("child final")]),
            ScriptedTurn([TextContent("finish gate final")]),
            ScriptedTurn([TextContent("settled final")]),
        ]
    )
    nested_backend = NestedBackend([ScriptedTurn([TextContent("nested final")])])

    def resolve_child_backend(parent_loop, _model):
        if parent_loop.agent_instance_id is None:
            return child_backend, None
        return nested_backend, None

    monkeypatch.setattr(
        agent_runner_module, "resolve_child_backend", resolve_child_backend
    )
    finish_gate_calls = 0

    async def run_finish_gate(*args, **kwargs):
        nonlocal finish_gate_calls
        finish_gate_calls += 1
        release_nested.set()
        return await real_run_finish_gate(*args, **kwargs)

    monkeypatch.setattr(agent_runner_module, "run_finish_gate", run_finish_gate)
    catalog = AgentCatalog(
        (
            AgentPreset(
                name="unset",
                preamble="unset",
                selection_guidance="unset child",
                tool_names=None,
            ),
            AgentPreset(
                name="nested",
                preamble="nested",
                selection_guidance="nested child",
                tool_names=None,
            ),
        )
    )
    store = ConversationStore(tmp_path / "session", session_id="root", cwd=tmp_path)
    registry = ToolRegistry(
        store.cwd,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )

    call = ToolCall(
        "agent-1",
        "agent",
        {
            "prompt": "child work",
            "description": "unset child",
            "agent_type": "unset",
            "background": False,
        },
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )
    try:
        await asyncio.wait_for(
            asyncio.create_task(
                _collect(loop.run_turn("start", origin=MessageOrigin.USER))
            ),
            timeout=10,
        )
        assert nested_started.is_set()
        child_path = store.session_dir / "agents" / "1"
        child = ConversationStore(store.session_dir / "agents", session_id="1")
        try:
            result = store.tool_result("agent-1")
            assert result is not None
            lifecycle = child.agent_lifecycle()
            assert lifecycle is not None
            status = _read_agent_status(store, requested_handle=f"{store.session_id}:1")[0]
            tui_output = StringIO()
            app = TUIApp(
                AgentLoop(
                    FakeBackend([]), store, skill_catalog=SkillCatalog.empty()
                ),
                provider="codex",
                model="offline",
                console=Console(file=tui_output, force_terminal=False),
            )
            async with owned_app(app):
                app._rebuild_transcript()
            rendered = Text.from_ansi(tui_output.getvalue()).plain
            serve_rows = [
                row
                for entry in store.replay()
                if entry.type in {"message", "notification"}
                for row in _history_entry_rows(entry, {}, fidelity=True)
            ]
            child_messages = [
                {
                    "role": message.role.value,
                    "text": "".join(
                        block.text
                        for block in message.content
                        if isinstance(block, TextContent)
                    ),
                    "tool": message.tool_result.content
                    if message.tool_result is not None
                    else None,
                }
                for message in child.messages()
            ]
            def stable(value: object) -> object:
                text = json.dumps(value)
                text = text.replace(str(tmp_path), "<TMP>")
                text = re.sub(r'\b[0-9a-f]{32}\b', "<ID>", text)
                text = re.sub(r'\b\d+\.\d+s\b', "<ELAPSED>", text)
                text = re.sub(r'"elapsed": \d+(?:\.\d+)?', '"elapsed": "<ELAPSED>"', text)
                text = re.sub(r'\d+\.\d+', "<FLOAT>", text)
                return json.loads(text)

            def digest(value: object) -> str:
                payload = json.dumps(stable(value), sort_keys=True).encode()
                return hashlib.sha256(payload).hexdigest()

            stable_messages = stable(child_messages)
            assert isinstance(stable_messages, list)
            return stable({
                "child_calls": len(child_backend.calls),
                "finish_gate_calls": finish_gate_calls,
                "message_roles": [message["role"] for message in stable_messages],
                "messages_sha256": digest(child_messages),
                "result": {
                    "content": result.content,
                    "is_error": result.is_error,
                    "structured": {
                        key: value
                        for key, value in (result.structured_content or {}).items()
                        if key != "child_session_path"
                    },
                },
                "lifecycle": {
                    key: lifecycle.get(key)
                    for key in (
                        "state",
                        "turns_used",
                        "tool_calls",
                        "current_step",
                        "depth",
                        "agent_type",
                        "description",
                        "final_result",
                        "final_result_is_receipt",
                    )
                },
                "status": {
                    key: value
                    for key, value in status.items()
                    if key
                    not in {
                        "child_session_path",
                        "cwd",
                        "elapsed",
                        "finished_at",
                        "started_at",
                    }
                },
                "notice_count": sum(
                    message.metadata.get("zeta_event") == "child_turn_limit"
                    for message in child.messages()
                ),
                "marker": "[zeta:turn-limit" in result.content,
                "handoff": (child_path / "turn-limit-handoff.md").exists(),
                "tui_sha256": digest(rendered),
                "serve_sha256": digest(serve_rows),
            })
        finally:
            child.close()
    finally:
        release_nested.set()
        await loop.close()
        store.close()


async def _collect(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_child_without_max_turns_matches_origin_main_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = await _unset_child_snapshot(tmp_path, monkeypatch)
    expected = {
        "child_calls": 6,
        "finish_gate_calls": 1,
        "handoff": False,
        "lifecycle": {
            "agent_type": "unset",
            "current_step": "turn 6: thinking",
            "depth": 1,
            "description": "unset child",
            "final_result": "settled final · 6 turns · <ELAPSED> · 0 tool calls · error=false · canceled=false",
            "final_result_is_receipt": True,
            "state": "completed",
            "tool_calls": 0,
            "turns_used": 6,
        },
        "marker": False,
        "message_roles": [
            "user",
            "assistant",
            "tool_result",
            "assistant",
            "tool_result",
            "assistant",
            "system",
            "assistant",
            "system",
            "assistant",
            "system",
            "assistant",
        ],
        "messages_sha256": "5f8600eefe5eb754bd2ba0e86495a4f7d0d0ca051b14da210b53083a654e953d",
        "notice_count": 0,
        "result": {
            "content": "settled final · 6 turns · <ELAPSED> · 0 tool calls · error=false · canceled=false",
            "is_error": False,
            "structured": {
                "agent_type": "unset",
                "child_instance_id": "root:1",
                "turns_used": 6,
            },
        },
        "serve_sha256": "11137fd732b7899222543e2beb7cf1354156765f2aa2f60febe6d5c35d4e8cbb",
        "status": {
            "agent_type": "unset",
            "current_step": "turn 6: thinking",
            "depth": 1,
            "description": "unset child",
            "final_result": "settled final · 6 turns · <ELAPSED> · 0 tool calls · error=false · canceled=false",
            "handle": "root:1",
            "state": "completed",
            "tool_calls": 0,
            "turns_used": 6,
        },
        "tui_sha256": "ec0e0e2a3956aab826a0ed5026da33ec665b4030182a8577ee290421e9a1dfcc",
    }
    assert snapshot == expected
