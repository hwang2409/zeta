"""View model and bounded control for the ``/tasks`` background-process panel.

The panel is a deep module with a small surface. :class:`BackgroundTasksPanel`
owns all list/detail/kill state and turns an immutable task snapshot into styled
lines; it never touches the registry, a process, or the clock beyond the
``now`` it is handed. The surrounding mixin feeds it snapshots and output tail
text and drives the (de)activation and refresh cadence. This keeps the view
model pure and testable, and keeps live I/O out of rendering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.utils import get_cwidth

# A long command wraps at this content width so the float settles near, but not
# beyond, the shared status-card cap. Rendering stays deterministic regardless
# of the live terminal size, which also keeps the text screenshots stable.
CONTENT_WIDTH = 68
# The output box keeps only a tail. Reads are incremental (the mixin advances a
# byte cursor), and the kept text is bounded so a chatty task cannot grow the
# panel without limit.
MAX_OUTPUT_CHARS = 16 * 1024
MAX_OUTPUT_LINES = 200
# The first incremental read starts this far back from the live end so an
# already-running task shows recent output immediately without scanning the log.
OUTPUT_TAIL_BYTES = 8 * 1024

Fragment = tuple[str, str]
FragmentLine = list[Fragment]


class TaskSnapshot(Protocol):
    """Read-only shape the panel needs; satisfied by ``BackgroundTaskInfo``.

    Declared here so the TUI does not import the tools layer. ``started_at`` and
    ``ended_at`` are ``time.monotonic`` seconds, or ``None`` when unknown (a task
    recovered from a previous session).
    """

    task_id: str
    command: str
    pid: int
    owner: str
    running: bool
    exit_code: int | None
    note: str | None
    terminal_phase: str | None
    started_at: float | None
    ended_at: float | None
    output_bytes: int
    output_lines: int


def short_id(task_id: str) -> str:
    """Return the compact id shown in the list (the hex after ``task-``)."""

    tail = task_id.rsplit("-", 1)[-1]
    return tail[:8] if tail else task_id


def format_runtime(seconds: float | None) -> str:
    """Render a monotonic duration compactly: ``4s`` / ``1m04s`` / ``1h02m``."""

    if seconds is None:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def _status(info: TaskSnapshot) -> tuple[str, str]:
    """Return the status label and its style suffix (running/ok/failed/killed)."""

    if info.running:
        return "running", "running"
    if info.terminal_phase in {"task_kill", "session_shutdown"}:
        return "killed", "killed"
    if info.exit_code is None:
        return "exited", "exited"
    if info.exit_code == 0:
        return "exited 0", "exited"
    return f"exited {info.exit_code}", "failed"


def _one_line(command: str) -> str:
    return " ".join(command.split())


def _wrap(text: str, width: int) -> list[str]:
    """Wrap by cells, breaking long unbroken runs, never returning empty."""

    width = max(1, width)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        cells = 0
        for word in paragraph.split(" "):
            piece = word if not current else " " + word
            piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            if current and cells + piece_cells > width:
                lines.append(current)
                current, cells = "", 0
                piece = word
                piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            while piece_cells > width:
                head, piece = _split_cells(piece, width)
                lines.append((current + head) if current else head)
                current, cells = "", 0
                piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            current += piece
            cells += piece_cells
        lines.append(current)
    return lines or [""]


def _split_cells(text: str, width: int) -> tuple[str, str]:
    cells = 0
    for index, ch in enumerate(text):
        step = max(0, get_cwidth(ch))
        if cells + step > width:
            return text[:index], text[index:]
        cells += step
    return text, ""


class BackgroundTasksPanel:
    """Interactive state for the background-task list and detail views."""

    def __init__(self) -> None:
        self._tasks: tuple[TaskSnapshot, ...] = ()
        self._now: float = 0.0
        self._selected_id: str | None = None
        self._mode: str = "list"
        self._detail_id: str | None = None
        self._kill_pending: str | None = None
        self._output_text: str = ""
        self._output_total: int = 0

    # -- snapshot ingestion -------------------------------------------------

    def set_tasks(self, tasks: Sequence[TaskSnapshot], now: float) -> None:
        """Replace the snapshot, ordering running first then recent finishes."""

        self._now = now

        def sort_key(item: TaskSnapshot) -> tuple[int, float]:
            if item.running:
                # Oldest running first, so the list is stable as tasks start.
                return (0, item.started_at or 0.0)
            # Most recently finished first; unknown finish time sinks last.
            return (1, -(item.ended_at if item.ended_at is not None else -1e18))

        self._tasks = tuple(sorted(tasks, key=sort_key))
        ids = [task.task_id for task in self._tasks]
        if self._selected_id not in ids:
            self._selected_id = ids[0] if ids else None
        if self._kill_pending not in ids:
            self._kill_pending = None
        if self._detail_id is not None and self._detail_id not in ids:
            self._mode, self._detail_id = "list", None

    # -- read-only state ----------------------------------------------------

    @property
    def tasks(self) -> tuple[TaskSnapshot, ...]:
        return self._tasks

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def detail_id(self) -> str | None:
        return self._detail_id

    @property
    def kill_pending(self) -> str | None:
        return self._kill_pending

    @property
    def selected(self) -> TaskSnapshot | None:
        for task in self._tasks:
            if task.task_id == self._selected_id:
                return task
        return None

    def detail_task(self) -> TaskSnapshot | None:
        for task in self._tasks:
            if task.task_id == self._detail_id:
                return task
        return None

    # -- navigation ---------------------------------------------------------

    def move(self, delta: int) -> None:
        if self._mode != "list" or not self._tasks:
            return
        ids = [task.task_id for task in self._tasks]
        index = ids.index(self._selected_id) if self._selected_id in ids else 0
        index = min(max(0, index + delta), len(ids) - 1)
        self._selected_id = ids[index]
        self._kill_pending = None

    def open_details(self) -> TaskSnapshot | None:
        """Enter the detail view for the selection; returns it, or ``None``."""

        selection = self.selected
        if selection is None:
            return None
        self._mode = "detail"
        self._detail_id = selection.task_id
        self._kill_pending = None
        self._output_text = ""
        self._output_total = selection.output_lines
        return selection

    def back(self) -> bool:
        """Return to the list from detail. ``True`` when it closed a view."""

        if self._mode != "detail":
            return False
        self._mode = "list"
        self._detail_id = None
        self._output_text = ""
        return True

    # -- kill confirmation --------------------------------------------------

    def request_kill(self) -> str | None:
        """Arm or confirm killing the selected running task.

        First press arms a one-key confirm and returns ``None``; the second
        press returns the task id to kill.
        """

        selection = self.selected
        if selection is None or not selection.running:
            self._kill_pending = None
            return None
        if self._kill_pending == selection.task_id:
            self._kill_pending = None
            return selection.task_id
        self._kill_pending = selection.task_id
        return None

    def cancel_kill(self) -> None:
        self._kill_pending = None

    # -- live output tail ---------------------------------------------------

    def feed_output(self, text: str, total_lines: int) -> None:
        """Append new tail text and update the total-line count (``M``)."""

        if text:
            combined = self._output_text + text
            if len(combined) > MAX_OUTPUT_CHARS:
                combined = combined[-MAX_OUTPUT_CHARS:]
            self._output_text = combined
        self._output_total = total_lines

    def _output_display_lines(self) -> list[str]:
        if not self._output_text:
            return []
        lines = self._output_text.splitlines()
        return lines[-MAX_OUTPUT_LINES:]

    # -- rendering ----------------------------------------------------------

    def render_lines(self) -> list[FragmentLine]:
        if self._mode == "detail":
            return self._render_detail()
        return self._render_list()

    def _render_list(self) -> list[FragmentLine]:
        lines: list[FragmentLine] = [
            [("class:tasks-panel.title", "Background tasks")],
            [("class:tasks-panel", "")],
        ]
        if not self._tasks:
            lines.append(
                [("class:tasks-panel.hint", "No background tasks in this session.")]
            )
            lines.append(
                [("class:tasks-panel.hint", "Start one with run_background or a /command macro.")]
            )
            lines.append([("class:tasks-panel", "")])
            lines.append(self._list_hint())
            return lines
        for task in self._tasks:
            lines.append(self._list_row(task))
            if self._kill_pending == task.task_id:
                lines.append(
                    [
                        ("class:tasks-panel", "    "),
                        ("class:tasks-panel.killed", f"kill {short_id(task.task_id)}? press k again · n cancel"),
                    ]
                )
        lines.append([("class:tasks-panel", "")])
        lines.append(self._list_hint())
        return lines

    def _list_row(self, task: TaskSnapshot) -> FragmentLine:
        selected = task.task_id == self._selected_id
        marker = "›" if selected else " "
        label, suffix = _status(task)
        runtime = format_runtime(self._runtime(task))
        base = "class:tasks-panel.selected" if selected else "class:tasks-panel"
        row: FragmentLine = [
            (base, f" {marker} "),
            (base, f"{short_id(task.task_id):<8}  "),
            (f"class:tasks-panel.{suffix}", f"{label:<10}"),
            (base, f" {runtime:>6}  "),
            (base, _one_line(task.command)),
        ]
        return row

    def _list_hint(self) -> FragmentLine:
        return [
            (
                "class:tasks-panel.hint",
                "↑/↓ select · enter details · k kill · esc close",
            )
        ]

    def _render_detail(self) -> list[FragmentLine]:
        task = self.detail_task()
        if task is None:
            return self._render_list()
        label, suffix = _status(task)
        runtime = format_runtime(self._runtime(task))
        lines: list[FragmentLine] = [
            [("class:tasks-panel.title", "Shell details")],
            [("class:tasks-panel", "")],
            self._field("Status", label, f"class:tasks-panel.{suffix}"),
            self._field("Runtime", runtime),
            self._field("PID", str(task.pid)),
        ]
        if task.owner != "run_background":
            lines.append(self._field("Owner", task.owner))
        if not task.running and task.exit_code is not None:
            lines.append(self._field("Exit", str(task.exit_code)))
        if task.note:
            lines.append(self._field("Note", task.note))
        lines.append([("class:tasks-panel", "")])
        lines.append([("class:tasks-panel.label", "Command")])
        for wrapped in _wrap(_one_line(task.command), CONTENT_WIDTH - 2):
            lines.append([("class:tasks-panel", f"  {wrapped}")])
        lines.append([("class:tasks-panel", "")])
        display = self._output_display_lines()
        lines.append(
            [
                ("class:tasks-panel.label", "Output  "),
                (
                    "class:tasks-panel.hint",
                    f"showing {len(display)} of {self._output_total} lines",
                ),
            ]
        )
        if display:
            for raw in display:
                lines.append([("class:tasks-panel.output", f"  {raw}")])
        else:
            lines.append([("class:tasks-panel.hint", "  (no output yet)")])
        lines.append([("class:tasks-panel", "")])
        lines.append(
            [("class:tasks-panel.hint", "↑/↓ scroll · esc back")]
        )
        return lines

    @staticmethod
    def _field(label: str, value: str, value_style: str = "class:tasks-panel") -> FragmentLine:
        return [
            ("class:tasks-panel.label", f"{label:<8}"),
            (value_style, value),
        ]

    def _runtime(self, task: TaskSnapshot) -> float | None:
        if task.started_at is None:
            return None
        end = task.ended_at if task.ended_at is not None else self._now
        return max(0.0, end - task.started_at)


def _fit_fragments(fragments: FragmentLine, width: int) -> FragmentLine:
    """Truncate fragments to ``width`` cells and pad the remainder with blanks."""

    if width <= 0:
        return []
    result: FragmentLine = []
    cells = 0
    for style, text in fragments:
        if cells >= width:
            break
        kept: list[str] = []
        for ch in text:
            step = max(0, get_cwidth(ch))
            if cells + step > width:
                break
            kept.append(ch)
            cells += step
        if kept:
            result.append((style, "".join(kept)))
    if cells < width:
        result.append(("class:tasks-panel", " " * (width - cells)))
    return result


class BackgroundTasksControl(UIControl):
    """A bounded, keyboard-scrollable view of styled panel lines."""

    def __init__(self) -> None:
        self._lines: tuple[FragmentLine, ...] = ()
        self._offset = 0
        self._height = 1

    @property
    def is_focusable(self) -> bool:
        return True

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def set_lines(self, lines: Sequence[FragmentLine], *, keep_offset: bool = False) -> None:
        self._lines = tuple(lines)
        if not keep_offset:
            self._offset = 0
        self._clamp_offset()

    def scroll(self, amount: int) -> None:
        maximum = max(0, len(self._lines) - self._height)
        self._offset = min(max(0, self._offset + amount), maximum)

    def page(self, amount: int) -> None:
        self.scroll(amount * max(1, self._height - 2))

    def top(self) -> None:
        self._offset = 0

    def bottom(self) -> None:
        self._offset = max(0, len(self._lines) - self._height)

    def _clamp_offset(self) -> None:
        maximum = max(0, len(self._lines) - self._height)
        self._offset = min(max(0, self._offset), maximum)

    @staticmethod
    def _line_cells(line: FragmentLine) -> int:
        return sum(
            max(0, get_cwidth(ch)) for _, text in line for ch in text
        )

    def preferred_width(self, max_available_width: int) -> int:
        del max_available_width
        natural = max((self._line_cells(line) for line in self._lines), default=0)
        return natural + 4

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: object | None,
    ) -> int:
        del width, max_available_height, wrap_lines, get_line_prefix
        return max(1, len(self._lines))

    def create_content(self, width: int, height: int | None) -> UIContent:
        self._height = max(1, height or 1)
        self._clamp_offset()
        content_width = max(0, width - 4)
        lines = self._lines

        def get_line(index: int) -> FragmentLine:
            if width < 4 or index >= len(lines):
                return [("class:tasks-panel", " " * width)]
            body = _fit_fragments(lines[index], content_width)
            rendered: FragmentLine = [("class:tasks-panel", "│ "), *body]
            rendered.append(("class:tasks-panel", " │"))
            return rendered

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    def vertical_scroll(self, window: Window) -> int:
        del window
        return self._offset

    def window(self) -> Window:
        return Window(
            self,
            wrap_lines=False,
            get_vertical_scroll=self.vertical_scroll,
            style="class:tasks-panel",
        )


__all__ = [
    "CONTENT_WIDTH",
    "MAX_OUTPUT_LINES",
    "OUTPUT_TAIL_BYTES",
    "BackgroundTasksControl",
    "BackgroundTasksPanel",
    "TaskSnapshot",
    "format_runtime",
    "short_id",
    "tasks_panel_style_rules",
]


def tasks_panel_style_rules() -> dict[str, str]:
    """Return the prompt-toolkit style rules for the panel over its surface.

    Defined next to the control it styles so the composition root does not grow
    a second copy of every class name.
    """

    from .. import theme

    surface = f"bg:{theme.SURFACE}" if theme.SURFACE else ""

    def rule(foreground: str) -> str:
        return " ".join(part for part in (foreground, surface) if part)

    return {
        "tasks-panel": rule(f"fg:{theme.BODY}"),
        "tasks-panel.title": rule(f"fg:{theme.ACCENT} bold"),
        "tasks-panel.label": rule(f"fg:{theme.DIM}"),
        "tasks-panel.hint": rule(f"fg:{theme.CHROME}"),
        "tasks-panel.selected": rule(f"fg:{theme.BODY} bold"),
        "tasks-panel.running": rule(f"fg:{theme.ACCENT} bold"),
        "tasks-panel.exited": rule(f"fg:{theme.SUCCESS}"),
        "tasks-panel.failed": rule(theme.ERROR),
        "tasks-panel.killed": rule(f"fg:{theme.CHROME}"),
        "tasks-panel.output": rule(f"fg:{theme.BODY}"),
    }
