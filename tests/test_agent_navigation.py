from __future__ import annotations

import json
from io import BytesIO, StringIO
from itertools import product
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from zeta.core.approval import ApprovalPolicy, ApprovalRequest
from zeta.core.fake import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tui import agent_card
from zeta.tui.agent_card import (
    MAX_AGENT_SCAN_BYTES,
    MAX_AGENT_VIEW_LINES,
    AgentNavigation,
    read_agent_transcript,
)
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.checkpoints import render_replayed_message
from zeta.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


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


def test_child_view_keeps_main_route_visible(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    _child(store, 1, description="Leaf")
    navigation = AgentNavigation(store)

    class Layout:
        def __init__(self) -> None:
            self.focused: object | None = None

        def focus(self, control: object) -> None:
            self.focused = control

        def has_focus(self, control: object) -> bool:
            return self.focused is control

    layout = Layout()
    navigation.bind_layout(layout, object())
    navigation.selected_index = 1
    navigation.open_selected()

    assert navigation.list_visible
    assert [entry.label for entry in navigation.entries] == ["main"]
    assert layout.focused is navigation.transcript_window
    navigation.focus_child_list()
    assert layout.focused is navigation.list_window


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
        ("main", "running"),
        ("Inspect", "failed"),
    ]

    navigation.selected_index = 1
    navigation.open_selected()
    assert navigation._breadcrumb_labels == ["main", "Explore", "Inspect"]
    assert navigation.current_path == grandchild


def test_main_row_is_first_and_is_a_back_route(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    grandchild_store = ConversationStore(child / "agents", session_id="1")
    grandchild_store.agent_lifecycle_path.write_text(
        json.dumps({"description": "Inspect", "state": "completed"})
    )
    grandchild = grandchild_store.session_dir
    navigation = AgentNavigation(store)

    assert navigation.entries[0].label == "main"
    navigation.open_selected()
    assert navigation.current_path == child
    navigation.move_selection(-1)
    assert navigation.entries[navigation.selected_index].label == "main"
    navigation.open_selected()
    assert navigation.current_path == store.session_dir
    assert navigation.selected_index == 1
    assert grandchild.exists()


def test_child_view_reuses_markdown_and_tool_card_rendering(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    read = ToolCall("read-1", "read", {"path": "app.py"})
    edit = ToolCall(
        "edit-1",
        "edit",
        {"path": "app.py", "old_string": "before", "new_string": "after"},
    )
    child_store = ConversationStore(child.parent, session_id=child.name)
    for index in range(MAX_AGENT_VIEW_LINES):
        child_store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"older {index}")])
        )
    child_store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("**markdown answer**"), ToolUseContent(read)],
        )
    )
    child_store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(read.id, "print('safe')"),
        )
    )
    child_store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(edit)])
    )
    child_store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(edit.id, "<!DOCTYPE HTML>" + " dump" * 10_000),
        )
    )

    navigation = AgentNavigation(store)
    navigation.open_selected()
    rendered = Text.from_ansi(
        navigation.transcript_control.transcript.render(120)
    ).plain

    assert navigation.transcript_control.lines[0].endswith("older lines omitted]")
    assert "markdown answer" in rendered
    assert "read app.py" in rendered
    assert "edit app.py" in rendered
    assert "tool result:" not in rendered
    assert "<!DOCTYPE HTML>" not in rendered
    assert any(
        type(unit).__name__ == "_ToolUnit"
        for unit in navigation.transcript_control.transcript.units
    )


def test_child_replay_bounds_thoughts_and_text_before_rendering(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    child_store = ConversationStore(child.parent, session_id=child.name)
    long_text = "x" * 3_000
    child_store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [
                ThinkingContent("\n".join(f"thought {index}" for index in range(400))),
            ],
        )
    )
    child_store.append_message(Message(MessageRole.ASSISTANT, [TextContent(long_text)]))

    navigation = AgentNavigation(store)
    navigation.open_selected()
    rendered_lines = navigation.transcript_control.transcript.lines(120)
    rendered = "\n".join(rendered_lines)

    assert len(rendered_lines) < MAX_AGENT_VIEW_LINES + 32
    assert "thought 0" not in rendered
    assert "thought 399" in rendered
    assert long_text not in rendered
    assert "x" * 2_001 not in rendered
    assert rendered.count("x") <= 2_000


def test_child_replay_caps_rendered_rows_at_multiple_widths(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Wide thought")
    child_store = ConversationStore(child.parent, session_id=child.name)
    child_store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("\n".join(f"thought {index} " + "x" * 1_980 for index in range(100)))],
        )
    )

    navigation = AgentNavigation(store)
    navigation.open_selected()

    for width in (80, 120):
        rendered_lines = navigation.transcript_control.transcript.lines(width)
        rendered = "\n".join(rendered_lines)
        assert len(rendered_lines) == MAX_AGENT_VIEW_LINES
        assert "thought 0" not in rendered
        assert "thought 99" in rendered


def test_child_replay_sanitizes_markdown_and_thought_controls() -> None:
    controls = "\x9b31mCSI\x9b0m \x9dOSC\x9c \x90DCS\x9c"
    control = agent_card.AgentTranscriptControl()
    control.presenter.console = Console(
        file=StringIO(), force_terminal=True, color_system="truecolor"
    )
    transcript = control.transcript
    presenter = control.presenter
    message = Message(
        MessageRole.ASSISTANT,
        [ThinkingContent(controls), TextContent(controls)],
    )

    render_replayed_message(
        message,
        presenter=presenter,
        print_unit=presenter.print_unit,
        tool_calls={},
        include_thoughts=True,
    )
    rendered = transcript.render(80)

    assert "\x9b" not in rendered
    assert "\x9d" not in rendered
    assert "\x90" not in rendered
    assert "OSC" not in rendered
    assert "DCS" not in rendered


def test_prefix_accounting_guards_each_row_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = (
        json.dumps(
            {
                "type": "message",
                "data": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "row"}],
                    }
                },
            }
        ).encode()
        + b"\n"
    )

    class GuardedReader(BytesIO):
        def readline(self, size: int = -1) -> bytes:
            assert size >= 0
            return super().readline(size)

    handle = GuardedReader(row)

    assert agent_card._count_rendered_lines_before(handle, len(row)) == 1
    monkeypatch.setattr(agent_card, "MAX_AGENT_SCAN_BYTES", len(row) - 1)
    assert agent_card._count_rendered_lines_before(GuardedReader(row), len(row)) is None


def test_child_replay_renders_incomplete_tool_start(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    child_store = ConversationStore(child.parent, session_id=child.name)
    child_store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("pending-1", "read", {"path": "app.py"}))],
        )
    )

    navigation = AgentNavigation(store)
    navigation.open_selected()
    rendered = Text.from_ansi(
        navigation.transcript_control.transcript.render(120)
    ).plain

    assert "read app.py" in rendered
    assert any(
        type(unit).__name__ == "_ToolUnit"
        for unit in navigation.transcript_control.transcript.units
    )


def test_truncated_tool_result_keeps_card_and_sanitizes_output(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Shell")
    call = ToolCall("bash-1", "bash", {"command": "printf output"})
    child_store = ConversationStore(child.parent, session_id=child.name)
    child_store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    child_store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(
                call.id,
                "\n".join(f"\x1b[41mline-{index}\x1b[0m" for index in range(300)),
            ),
        )
    )

    navigation = AgentNavigation(store)
    navigation.open_selected()
    rendered = navigation.transcript_control.transcript.render(120)
    plain = Text.from_ansi(rendered).plain

    assert "bash" in plain
    assert "tool result:" not in plain
    assert "\x1b[41m" not in rendered
    assert any(
        type(unit).__name__ == "_ToolUnit"
        for unit in navigation.transcript_control.transcript.units
    )


def test_replay_defaults_keep_main_transcript_presentation() -> None:
    call = ToolCall("read-1", "read", {"path": "main.py"})
    assistant = Message(
        MessageRole.ASSISTANT,
        [ThinkingContent("private plan"), TextContent("visible answer"), ToolUseContent(call)],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        [],
        tool_result=ToolResult(call.id, "output"),
    )
    units: list[object] = []
    tool_calls: dict[str, ToolCall] = {}

    render_replayed_message(
        assistant,
        presenter=object(),
        print_unit=units.append,
        tool_calls=tool_calls,
    )
    render_replayed_message(
        result,
        presenter=object(),
        print_unit=units.append,
        tool_calls=tool_calls,
    )

    plain = "\n".join(getattr(unit, "plain", "") for unit in units)
    assert "visible answer" in plain
    assert "private plan" not in plain
    assert len(units) == 2


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
    assert lines[0] == "[22 older lines omitted]"


def test_child_transcript_exact_fit_has_no_truncation_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Exact")
    with child.joinpath("conversation.jsonl").open("w") as handle:
        for index in range(MAX_AGENT_VIEW_LINES):
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

    assert len(lines) == MAX_AGENT_VIEW_LINES
    assert lines[0] == "assistant: line 0"


def test_child_transcript_reports_boundary_byte_omission(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Boundary")
    row_size = MAX_AGENT_SCAN_BYTES // MAX_AGENT_VIEW_LINES
    with child.joinpath("conversation.jsonl").open("w") as handle:
        for index in range(MAX_AGENT_VIEW_LINES + 132):
            message = {
                "type": "message",
                "data": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"line {index}"}],
                    }
                },
            }
            encoded = json.dumps(message)
            message["data"]["message"]["content"][0]["text"] += "x" * (
                row_size - len(encoded.encode()) - 1
            )
            encoded = json.dumps(message)
            assert len(encoded.encode()) + 1 == row_size
            handle.write(encoded + "\n")

    lines = read_agent_transcript(child)

    assert lines[0] == "[132 older lines omitted]"
    assert lines[1].startswith("assistant: line 132")


def test_child_transcript_reports_overflow_count(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Overflow")
    with child.joinpath("conversation.jsonl").open("w") as handle:
        for index in range(MAX_AGENT_VIEW_LINES + 132):
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

    assert lines[0] == "[132 older lines omitted]"
    assert lines[-1] == f"assistant: line {MAX_AGENT_VIEW_LINES + 131}"


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


_TRUNCATION_SWEEP_CASES = tuple(
    product(
        (False, True),
        (1, 2, 5),
        ("uniform", "mixed"),
        ("below", "at", "above"),
        ("start", "boundary", "mid-row"),
        ("single", "mixed-pair"),
    )
)


@pytest.mark.parametrize(
    "header,line_count,byte_sizes,total_size,landing,message_layout",
    _TRUNCATION_SWEEP_CASES,
)
def test_child_transcript_truncation_accounting_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: bool,
    line_count: int,
    byte_sizes: str,
    total_size: str,
    landing: str,
    message_layout: str,
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Sweep")
    limit = MAX_AGENT_VIEW_LINES
    group_lines = line_count if message_layout == "single" else line_count * 2 + 1
    if total_size == "below":
        message_count = max(1, (limit - group_lines) // group_lines)
    elif total_size == "at":
        message_count = max(1, limit // group_lines)
    else:
        message_count = limit // group_lines + 132

    rows: list[bytes] = []
    if header:
        rows.append(
            (
                json.dumps(
                    {"type": "header", "data": {"schema": "test"}},
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
    rendered: list[str] = []
    for index in range(message_count):
        parts = (
            ((line_count, ""),)
            if message_layout == "single"
            else ((line_count, "a"), (line_count + 1, "b"))
        )
        for part_index, (part_lines, suffix) in enumerate(parts):
            target_size = 512 if byte_sizes == "uniform" else 384 + (index % 2) * 256
            text_lines = [
                f"message {index}{suffix} line {line}" for line in range(part_lines)
            ]
            message = {
                "type": "message",
                "data": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "\n".join(text_lines)}],
                    }
                },
            }
            while len(json.dumps(message, separators=(",", ":")).encode()) + 1 < target_size:
                text_lines[-1] += "x"
                message["data"]["message"]["content"][0]["text"] = "\n".join(text_lines)
            if (
                total_size == "above"
                and landing == "mid-row"
                and index == (100 if message_layout == "mixed-pair" else 131)
                and part_index == 0
            ):
                message["padding"] = "x" * 200_000
            encoded = json.dumps(message, separators=(",", ":")).encode() + b"\n"
            rows.append(encoded)
            rendered.extend(f"assistant: {line}" for line in text_lines)

    file_size = sum(len(row) for row in rows)
    prefix_message_count = 0
    if total_size == "above":
        prefix_message_count = (
            100 if message_layout == "mixed-pair" else 131
            if landing == "mid-row"
            else 132
        )
    elif total_size == "at" and header and landing == "boundary":
        prefix_message_count = 0
    elif total_size in {"below", "at"}:
        landing = "start"

    prefix_rows = (1 if header else 0) + prefix_message_count * (
        1 if message_layout == "single" else 2
    )
    prefix_end = sum(len(row) for row in rows[:prefix_rows])
    if total_size == "above" and landing == "mid-row":
        partial_row_end = sum(len(row) for row in rows[: prefix_rows + 1])
        scan_bytes = file_size - partial_row_end + 1
    elif landing == "boundary":
        scan_bytes = file_size - prefix_end
    else:
        scan_bytes = file_size + 1
    monkeypatch.setattr(agent_card, "MAX_AGENT_SCAN_BYTES", scan_bytes)
    child.joinpath("conversation.jsonl").write_bytes(b"".join(rows))

    lines = read_agent_transcript(child, limit=limit)

    if total_size == "above" and landing == "mid-row":
        expected_marker = "[older lines omitted]"
    elif total_size == "above":
        expected_marker = f"[{len(rendered) - limit} older lines omitted]"
    elif total_size == "at" and header and landing == "boundary":
        expected_marker = None
    else:
        expected_marker = None
    expected_tail = rendered[-limit:]
    assert lines == ([expected_marker] if expected_marker else []) + expected_tail


def test_child_transcript_keeps_tail_of_one_oversized_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.agent_lifecycle_path.write_text(
        json.dumps({"description": "Explore", "agent_type": "general", "state": "completed"})
    )
    child.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("\n".join(f"line {index}" for index in range(100_000)))],
        )
    )

    raw_tail_sizes: list[int] = []
    recover_oversized_message = agent_card._oversized_message

    def record_raw_tail(raw_tail: bytes) -> object:
        raw_tail_sizes.append(len(raw_tail))
        return recover_oversized_message(raw_tail)

    monkeypatch.setattr(agent_card, "_oversized_message", record_raw_tail)
    navigation = AgentNavigation(store)
    navigation.selected_index = 1
    navigation.open_selected()

    assert len(raw_tail_sizes) == 1
    assert raw_tail_sizes[0] <= MAX_AGENT_SCAN_BYTES
    assert navigation.transcript_control.lines[-1] == "assistant: line 99999"
    assert navigation.transcript_control.lines[0] == "[older lines omitted]"
    assert "transcript unavailable" not in navigation.transcript_control.lines


def test_transcript_control_scrolls_with_bounded_content(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    _message(child, "assistant", [{"type": "text", "text": "one  \ntwo  \nthree"}])
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


@pytest.mark.asyncio
async def test_child_approval_exits_navigation_in_full_screen(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    child = _child(store, 1, description="Explore")
    child_store = ConversationStore(child.parent, session_id=child.name)
    call = ToolCall("approval-full-screen", "danger", {})
    child_store.append_approval_request(call.id, call)
    instance_id = "root:1"
    policy = ApprovalPolicy(default="ask", store=store)
    policy.register_delegated(
        ApprovalRequest(call.id, call, child_instance_id=instance_id),
        child_store,
        child_instance_id=instance_id,
    )
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
        console=Console(force_terminal=False),
    )
    session = app._make_session()
    assert isinstance(session, FullScreenPromptSession)
    app._active_session = session
    app._install_full_screen_layout(session)
    app._agent_navigation.selected_index = 1
    app._agent_navigation.open_selected()
    assert app._agent_navigation.child_view_active

    app._handle_background_event(
        StreamEvent(
            StreamEventType.TOOL_APPROVAL_START,
            tool_call=call,
            data={"agent_instance_id": instance_id},
        )
    )

    assert app._agent_navigation.current_path == store.session_dir
    assert session.layout.has_focus(session.default_buffer)
    assert "danger" in app._transcript._base_render(80)
    await app._handle_approval_input(f"approve {app.pending_approvals[0].key}")
    assert not app.pending_approvals
    await app.close()
