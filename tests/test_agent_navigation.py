from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.approval import ApprovalPolicy, ApprovalRequest
from zeta.core.fake import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tui import agent_card
from zeta.tui.agent_card import (
    MAX_AGENT_VIEW_LINES,
    AgentNavigation,
    read_agent_transcript,
)
from zeta.tui.app import TUIApp
from zeta.types import StreamEvent, StreamEventType, ToolCall


def _child(
    store: ConversationStore,
    number: int,
    *,
    description: str,
    agent_type: str = "general",
    state: str = "completed",
) -> Path:
    child = ConversationStore(store.session_dir / "agents", session_id=str(number))
    child.agent_lifecycle_path.write_text(
        json.dumps(
            {
                "description": description,
                "agent_type": agent_type,
                "state": state,
            }
        )
    )
    return child.session_dir


def _message(path: Path, role: str, blocks: list[dict[str, object]]) -> None:
    path.joinpath("conversation.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "data": {"message": {"role": role, "content": blocks}},
            }
        )
        + "\n"
    )


def test_list_is_quiet_without_children(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")

    navigation = AgentNavigation(store)

    assert not navigation.list_visible
    assert navigation.entries == []


def test_list_shows_child_state_and_recursive_breadcrumb(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore", agent_type="research")
    grandchild_store = ConversationStore(child / "agents", session_id="1")
    grandchild_store.agent_lifecycle_path.write_text(
        json.dumps(
            {
                "description": "Inspect",
                "agent_type": "code",
                "state": "failed",
            }
        )
    )
    grandchild = grandchild_store.session_dir

    navigation = AgentNavigation(store)

    assert [(entry.label, entry.state) for entry in navigation.entries] == [
        ("main", "running"),
        ("Explore", "completed"),
    ]

    navigation.selected_index = 1
    navigation.open_selected()
    assert navigation._breadcrumb_labels == ["main", "Explore"]
    assert [(entry.label, entry.state) for entry in navigation.entries] == [
        ("Explore", "completed"),
        ("Inspect", "failed"),
    ]

    navigation.selected_index = 1
    navigation.open_selected()
    assert navigation._breadcrumb_labels == ["main", "Explore", "Inspect"]
    assert navigation.current_path == grandchild


def test_child_transcript_excludes_nested_child_calls_and_is_bounded(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    nested_store = ConversationStore(child / "agents", session_id="1")
    nested_store.agent_lifecycle_path.write_text(
        json.dumps({"description": "Nested", "agent_type": "general", "state": "completed"})
    )
    nested = nested_store.session_dir
    _message(
        child,
        "assistant",
        [
            {"type": "text", "text": "direct output"},
            {
                "type": "tool_use",
                "tool_call": {"name": "websearch", "arguments": {"query": "direct"}},
            },
        ],
    )
    _message(
        nested,
        "assistant",
        [
            {
                "type": "tool_use",
                "tool_call": {"name": "websearch", "arguments": {"query": "nested"}},
            }
        ],
    )
    with child.joinpath("conversation.jsonl").open("a") as handle:
        for index in range(MAX_AGENT_VIEW_LINES + 20):
            handle.write(
                json.dumps(
                    {
                        "type": "message",
                        "data": {
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": f"line {index}"}],
                            }
                        },
                    }
                )
                + "\n"
            )

    lines = read_agent_transcript(child)

    assert len(lines) == MAX_AGENT_VIEW_LINES + 1
    assert "query=nested" not in "\n".join(lines)
    assert lines[0] == "[older lines omitted]"


def test_child_transcript_scans_only_a_bounded_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    row = {
        "type": "message",
        "data": {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "tail"}],
            }
        },
    }
    path = child / "conversation.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(10_000)))
    parsed_rows = 0
    load_session_json = agent_card.load_session_json

    def count_rows(raw_line: bytes) -> object:
        nonlocal parsed_rows
        parsed_rows += 1
        return load_session_json(raw_line)

    monkeypatch.setattr(agent_card, "load_session_json", count_rows)
    lines = read_agent_transcript(child)

    assert lines[-1] == "assistant: tail"
    assert parsed_rows < 10_000


def test_transcript_control_scrolls_with_bounded_content(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    _message(child, "assistant", [{"type": "text", "text": "one\ntwo\nthree"}])
    navigation = AgentNavigation(store)
    navigation.selected_index = 1
    navigation.open_selected()

    content = navigation.transcript_control.create_content(80, 2)
    assert content.line_count == 3
    navigation.child_top()
    assert navigation.transcript_control.offset == 0
    navigation.child_bottom()
    assert navigation.transcript_control.offset == 1


def test_nested_tool_events_stay_out_of_the_parent_transcript(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skip_mcp_mount=True, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
    )
    calls: list[object] = []
    app._presenter.handle_tool_event = lambda *args, **kwargs: calls.append(args)  # type: ignore[method-assign]

    handled = app._handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=ToolCall("nested", "read", {"path": "README.md"}),
            data={"agent_instance_id": "root:1"},
        )
    )

    assert not handled
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("open_child", [False, True])
async def test_child_approval_surfaces_when_view_is_closed_or_open(
    tmp_path: Path, open_child: bool
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    child_store = ConversationStore(child.parent, session_id=child.name)
    call = ToolCall("approval-1", "danger", {})
    child_store.append_approval_request(call.id, call)
    child_instance_id = "root:1"
    policy = ApprovalPolicy(default="ask", store=store)
    policy.register_delegated(
        ApprovalRequest(call.id, call, child_instance_id=child_instance_id),
        child_store,
        child_instance_id=child_instance_id,
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            store,
            approval_policy=policy,
            skip_mcp_mount=True,
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        approval_policy=policy,
        console=Console(file=output, force_terminal=False),
    )
    if open_child:
        app._agent_navigation.selected_index = 1
        app._agent_navigation.open_selected()

    app._handle_background_event(
        StreamEvent(
            StreamEventType.TOOL_APPROVAL_START,
            tool_call=call,
            data={"agent_instance_id": child_instance_id},
        )
    )

    assert app.pending_approvals
    assert "danger" in output.getvalue()
    await app._handle_approval_input(f"approve {app.pending_approvals[0].key}")
    assert not app.pending_approvals
    assert child_store.approval_states()[call.id][1] == "allow"
    await app.close()
