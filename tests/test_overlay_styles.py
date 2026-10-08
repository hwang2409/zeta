"""Every popup style string must parse as real prompt-toolkit syntax.

The shared overlay frame (``/status``, ``/tasks``, ``/mcp``) and the finder all
express colour with semantic theme tokens run through
:func:`theme.prompt_toolkit_style`. PR #409 crashed the TUI when a Rich ``on
<color>`` background reached prompt-toolkit unconverted, so this guards every
popup across all three built-in palettes: each fragment style parses, no style
leaks the Rich ``on`` spelling, and a selected row really does set ``bg:``.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.styles import Style

from zeta.mcp.management import MCPManagementService
from zeta.tools._shared.process import BackgroundTaskInfo
from zeta.tui import overlay, theme
from zeta.tui.cards.mcp_manager import MCPManager
from zeta.tui.cards.tasks_panel import BackgroundTasksPanel


@pytest.fixture(autouse=True)
def _restore_palette():
    previous = theme.active_palette()
    yield
    theme.set_active_palette(previous)


def _task(task_id: str, **kwargs) -> BackgroundTaskInfo:
    base = {
        "command": "python -m http.server 8000",
        "pid": 4242,
        "owner": "run_background",
        "running": True,
        "exit_code": None,
        "note": None,
        "terminal_phase": None,
        "started_at": 0.0,
        "ended_at": None,
        "output_bytes": 0,
        "output_lines": 0,
    }
    base.update(kwargs)
    return BackgroundTaskInfo(task_id=task_id, **base)


def _tasks_list_lines() -> list[overlay.FragmentLine]:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [
            _task("task-aa", running=True),
            _task("task-bb", running=False, exit_code=0, ended_at=3.0),
            _task("task-cc", running=False, exit_code=1, ended_at=3.0),
        ],
        now=10.0,
    )
    panel.request_kill()  # arm the kill-confirm row too
    return panel.render_lines()


def _tasks_detail_lines() -> list[overlay.FragmentLine]:
    panel = BackgroundTasksPanel()
    panel.set_tasks([_task("task-aa", running=False, exit_code=1, ended_at=3.0)], now=9.0)
    panel.open_details()
    panel.feed_output("line one\nline two\n", total_lines=2)
    return panel.render_lines()


def _mcp_lines(tmp_path) -> list[overlay.FragmentLine]:
    service = MCPManagementService(home=tmp_path / "home", project_dir=tmp_path / "repo")
    service.add("alpha", scope="user", command="python")
    service.add("beta", scope="project", url="https://example.test", enabled=False)
    view = MCPManager(service)
    view.details = True
    view.last_result = "ok"
    return view.render()


def _status_lines() -> list[overlay.FragmentLine]:
    return [
        overlay.title("Status"),
        overlay.rule(),
        *([overlay.value(line)] for line in ("session abc123", "tokens 1,234", "")),
        overlay.rule(),
        overlay.hint("↑/↓ scroll · esc close"),
    ]


def _all_popup_lines(tmp_path) -> dict[str, list[overlay.FragmentLine]]:
    return {
        "status": _status_lines(),
        "tasks-list": _tasks_list_lines(),
        "tasks-detail": _tasks_detail_lines(),
        "mcp": _mcp_lines(tmp_path),
    }


def _assert_parses(lines: list[overlay.FragmentLine]) -> set[str]:
    """Parse every framed fragment style; return the styles that set ``bg:``."""

    parser = Style([])
    backgrounds: set[str] = set()
    for row in overlay.frame(lines, width=80):
        for style, _text in row:
            parser.get_attrs_for_style_str(style)
            assert " on " not in style, f"unconverted Rich background: {style!r}"
            if "bg:" in style:
                backgrounds.add(style)
    return backgrounds


@pytest.mark.parametrize("palette", list(theme.BUILT_IN_PALETTES.values()), ids=lambda p: p.name)
def test_every_popup_style_parses_in_every_palette(palette, tmp_path) -> None:
    theme.set_active_palette(palette)
    for name, lines in _all_popup_lines(tmp_path).items():
        backgrounds = _assert_parses(lines)
        if name in {"tasks-list", "mcp"}:
            # A selected row layers the palette highlight behind its text.
            assert backgrounds, f"{name} never painted a selection background"
