import ast
import inspect
import json
import re
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.todo import TODO_STATUSES
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools import ToolRegistry
from zeta.tools import todo as todo_tool
from zeta.tui.app import TUIApp
from zeta.tui.todo import TodoWidget
from zeta.types import ToolCall


def _registry(tmp_path: Path) -> tuple[ConversationStore, ToolRegistry]:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    return store, ToolRegistry(tmp_path, session_store=store)


def _todo_handler_argument_keys() -> set[str]:
    tree = ast.parse(inspect.getsource(todo_tool._todo))
    keys: set[str] = set()

    def string_constant(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "arguments":
                key = string_constant(node.slice)
                if key is not None:
                    keys.add(key)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "arguments"
                and node.func.attr == "get"
                and node.args
            ):
                key = string_constant(node.args[0])
                if key is not None:
                    keys.add(key)
        elif isinstance(node, ast.Compare):
            if len(node.comparators) != 1 or not isinstance(
                node.ops[0], (ast.In, ast.NotIn)
            ):
                continue
            argument_name = node.comparators[0]
            if not (
                isinstance(argument_name, ast.Name) and argument_name.id == "arguments"
            ):
                continue
            key = string_constant(node.left)
            if key is not None:
                keys.add(key)

    return keys


@pytest.mark.asyncio
async def test_todo_writes_and_reads_the_full_list(tmp_path: Path) -> None:
    store, registry = _registry(tmp_path)
    items = [
        {"content": "inspect code", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ]

    written = await registry.execute(ToolCall("write", "todo", {"items": items}))
    read = await registry.execute(ToolCall("read", "todo", {}))
    action_read = await registry.execute(ToolCall("action-read", "todo", {}))

    expected = {
        "items": items,
        "counts": {"pending": 1, "in_progress": 1, "completed": 0},
    }
    assert written["isError"] is False
    assert written["structuredContent"] == expected
    assert read["structuredContent"] == expected
    assert action_read["structuredContent"] == expected
    assert store.todo_items() == items


def test_todo_schema_matches_handler_contract(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    schema = next(schema for schema in registry.schemas if schema["name"] == "todo")
    parameters = schema["parameters"]
    properties = parameters["properties"]
    item_schema = properties["items"]["items"]

    assert "Read the current todo list when items is omitted." in schema["description"]
    assert "Write the full todo list by providing items." in schema["description"]
    assert set(properties) == _todo_handler_argument_keys()
    assert set(item_schema["properties"]) == {"content", "status"}
    assert item_schema["required"] == ["content", "status"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["status"]["enum"] == list(TODO_STATUSES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        *(
            {"items": [{"content": status, "status": status}]}
            for status in TODO_STATUSES
        ),
    ],
)
async def test_every_todo_handler_branch_is_schema_representable(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    _, registry = _registry(tmp_path)

    result = await registry.execute(ToolCall("branch", "todo", arguments))

    assert result["isError"] is False


@pytest.mark.asyncio
async def test_todo_rejects_removed_action_argument(tmp_path: Path) -> None:
    _, registry = _registry(tmp_path)

    result = await registry.execute(ToolCall("action", "todo", {"action": "read"}))

    assert result["isError"] is True
    assert "unexpected properties: action" in result["structuredContent"]["error"]


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
async def test_todo_accepts_fifty_items(tmp_path: Path) -> None:
    store, registry = _registry(tmp_path)
    items = [{"content": f"task {index}", "status": "pending"} for index in range(50)]

    result = await registry.execute(ToolCall("fifty", "todo", {"items": items}))

    assert result["isError"] is False
    assert store.todo_items() == items


@pytest.mark.asyncio
async def test_todo_rejects_more_than_fifty_items_without_mutating_state(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    initial = [{"content": "keep", "status": "pending"}]
    await registry.execute(ToolCall("initial", "todo", {"items": initial}))
    state_before = store.state_path.read_bytes()
    items = [{"content": f"task {index}", "status": "pending"} for index in range(51)]

    result = await registry.execute(ToolCall("fifty-one", "todo", {"items": items}))

    assert result["isError"] is True
    assert "more than 50 items" in result["structuredContent"]["error"]
    assert store.todo_items() == initial
    assert store.state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_todo_rejects_overlong_content_without_mutating_state(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    initial = [{"content": "keep", "status": "pending"}]
    await registry.execute(ToolCall("initial", "todo", {"items": initial}))
    state_before = store.state_path.read_bytes()

    result = await registry.execute(
        ToolCall(
            "overlong",
            "todo",
            {"items": [{"content": "x" * 501, "status": "pending"}]},
        )
    )

    assert result["isError"] is True
    assert "cannot exceed 500 characters" in result["structuredContent"]["error"]
    assert store.todo_items() == initial
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


def test_todo_widget_collapses_completed_list_and_dismisses_at_boundary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items(
        [
            {"content": "first", "status": "completed"},
            {"content": "second", "status": "completed"},
        ]
    )

    content = widget.create_content(80, 20)
    assert content.line_count == 1
    assert "todos done (2)" in "".join(fragment[1] for fragment in content.get_line(0))
    assert widget.visible

    widget.turn_boundary()

    assert not widget.visible
    assert widget.create_content(80, 20).line_count == 0


def test_todo_widget_repins_after_a_new_write(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items([{"content": "done", "status": "completed"}])
    widget.create_content(80, 20)
    widget.turn_boundary()
    assert not widget.visible

    store.set_todo_items([{"content": "new", "status": "pending"}])

    assert widget.visible
    assert widget.create_content(80, 20).line_count == 1


def test_todo_widget_keeps_mixed_lists_pinned(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items(
        [
            {"content": "done", "status": "completed"},
            {"content": "work", "status": "in_progress"},
        ]
    )

    widget.turn_boundary()

    assert widget.visible
    assert widget.create_content(80, 20).line_count == 2


@pytest.mark.asyncio
async def test_todo_widget_keeps_overflow_summary_in_an_80_by_24_terminal(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items(
        [{"content": f"task {index}", "status": "pending"} for index in range(8)]
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._install_full_screen_layout(session)
    output = StringIO()
    terminal_output = Vt100_Output(output, lambda: Size(rows=24, columns=80))
    session.app.output = terminal_output
    session.app.renderer.output = terminal_output

    with set_app(session.app):
        session.app.renderer.render(session.app, session.app.layout)

    rendered = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.getvalue())
    assert rendered.count("[ ] task ") == 6
    assert "+2 more" in rendered


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
    bottom = content.children[1].content
    todo_panel = bottom.children[0]

    assert todo_panel.__class__.__name__ == "ConditionalContainer"
    assert todo_panel.content.content is app._todo_widget
