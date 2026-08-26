import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools import ToolRegistry
from zeta.tui.app import TUIApp
from zeta.tui.todo import TodoWidget
from zeta.types import ToolCall


def _registry(tmp_path: Path) -> tuple[ConversationStore, ToolRegistry]:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    return store, ToolRegistry(tmp_path, session_store=store)


@pytest.mark.asyncio
async def test_todo_writes_and_reads_the_full_list(tmp_path: Path) -> None:
    store, registry = _registry(tmp_path)
    items = [
        {"content": "inspect code", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ]

    written = await registry.execute(ToolCall("write", "todo", {"items": items}))
    read = await registry.execute(ToolCall("read", "todo", {}))
    action_read = await registry.execute(
        ToolCall("action-read", "todo", {"action": "read"})
    )

    expected = {
        "items": items,
        "counts": {"pending": 1, "in_progress": 1, "completed": 0},
    }
    assert written["isError"] is False
    assert written["structuredContent"] == expected
    assert read["structuredContent"] == expected
    assert action_read["structuredContent"] == expected
    assert store.todo_items() == items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ([{"content": " ", "status": "pending"}], "content must be nonempty"),
        ([{"content": "bad status", "status": "paused"}], "status must be one of"),
        (
            [
                {"content": "first", "status": "in_progress"},
                {"content": "second", "status": "in_progress"},
            ],
            "at most one in_progress",
        ),
    ],
)
async def test_todo_rejects_invalid_lists_without_mutating_state(
    tmp_path: Path, items: list[dict[str, str]], reason: str
) -> None:
    store, registry = _registry(tmp_path)
    await registry.execute(
        ToolCall(
            "initial",
            "todo",
            {"items": [{"content": "keep", "status": "pending"}]},
        )
    )
    state_before = store.state_path.read_bytes()

    result = await registry.execute(ToolCall("invalid", "todo", {"items": items}))

    assert result["isError"] is True
    assert reason in result["structuredContent"]["error"]
    assert store.todo_items() == [{"content": "keep", "status": "pending"}]
    assert store.state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_todo_empty_list_clears_state_and_does_not_pollute_transcript(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    await registry.execute(
        ToolCall(
            "write",
            "todo",
            {"items": [{"content": "remove", "status": "completed"}]},
        )
    )

    result = await registry.execute(ToolCall("clear", "todo", {"items": []}))
    reopened = ConversationStore(tmp_path / "sessions", session_id=store.session_id)

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "items": [],
        "counts": {"pending": 0, "in_progress": 0, "completed": 0},
    }
    assert reopened.todo_items() == []
    assert "todo_items" not in json.loads(reopened.state_path.read_text())
    assert reopened.messages() == []


def test_todo_items_persist_across_store_resume(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items([{"content": "resume me", "status": "pending"}])

    resumed = ConversationStore(tmp_path / "sessions", session_id=store.session_id)

    assert resumed.todo_items() == [{"content": "resume me", "status": "pending"}]
    assert json.loads(resumed.state_path.read_text())["todo_items"] == resumed.todo_items()


def test_todo_widget_hides_empty_lists_and_bounds_visible_rows(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    assert widget.create_content(80, 20).line_count == 0

    store.set_todo_items(
        [{"content": f"task {index}", "status": "pending"} for index in range(8)]
    )
    content = widget.create_content(80, 20)
    rendered = [
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    ]

    assert content.line_count == 7
    assert rendered[:2] == ["[ ] task 0", "[ ] task 1"]
    assert rendered[-1] == "+2 more"
    assert all(len(line) <= 80 for line in rendered)


def test_todo_widget_uses_plain_status_glyphs_and_truncates_content(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items(
        [
            {"content": "pending", "status": "pending"},
            {"content": "active", "status": "in_progress"},
            {"content": "done", "status": "completed"},
        ]
    )
    widget = TodoWidget(store)
    content = widget.create_content(10, 10)
    rendered = [
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    ]

    assert rendered == ["[ ] pendi…", "[>] active", "[x] done"]
    assert all("✱" not in line for line in rendered)


def test_status_includes_todo_counts_only_when_nonempty(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    registry = create_slash_registry()

    assert "todo:" not in registry.dispatch(app, "/status")
    store.set_todo_items([{"content": "one", "status": "completed"}])

    output = registry.dispatch(app, "/status")
    assert output is not None
    assert "todo: pending=0, in_progress=0, completed=1" in output


def test_full_screen_layout_places_todo_between_transcript_and_composer(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._install_full_screen_layout(session)

    content = session.layout.container.children[0].children[1]
    bottom = content.children[1]
    todo_panel = bottom.children[0]

    assert todo_panel.__class__.__name__ == "ConditionalContainer"
    assert todo_panel.content.content is app._todo_widget
