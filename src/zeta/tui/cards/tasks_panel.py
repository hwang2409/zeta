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

from prompt_toolkit.utils import get_cwidth

from .. import overlay, theme

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


def _status_style(suffix: str) -> str:
    """Map a status suffix to a shared theme token, read at paint time.

    Theme constants are rebuilt in place on a palette switch, so the lookup
    stays here instead of a module-level dict that would pin the first palette.
    """

    return {
        "running": f"bold {theme.ACCENT}",
        "exited": theme.SUCCESS,
        "failed": theme.ERROR,
        "killed": theme.DIM,
    }[suffix]


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
            overlay.title("Background tasks"),
            overlay.rule(),
        ]
        if not self._tasks:
            lines.append(overlay.hint("No background tasks in this session."))
            lines.append(
                overlay.hint("Start one with run_background or a /command macro.")
            )
            lines.append(overlay.rule())
            lines.append(self._list_hint())
            return lines
        for task in self._tasks:
            lines.append(self._list_row(task))
            if self._kill_pending == task.task_id:
                lines.append(
                    [
                        overlay.value(
                            f"      kill {short_id(task.task_id)}? "
                            "press k again · n cancel",
                            theme.WARNING,
                        )
                    ]
                )
        lines.append(overlay.rule())
        lines.append(self._list_hint())
        return lines

    def _list_row(self, task: TaskSnapshot) -> FragmentLine:
        selected = task.task_id == self._selected_id
        label, suffix = _status(task)
        runtime = format_runtime(self._runtime(task))
        return overlay.row(
            [
                (theme.BODY, f"{short_id(task.task_id):<8}  "),
                (_status_style(suffix), f"{label:<10}"),
                (theme.DIM, f" {runtime:>6}  "),
                (theme.BODY, _one_line(task.command)),
            ],
            selected=selected,
        )

    def _list_hint(self) -> FragmentLine:
        return overlay.hint("↑/↓ select · enter details · k kill · esc close")

    def _render_detail(self) -> list[FragmentLine]:
        task = self.detail_task()
        if task is None:
            return self._render_list()
        label, suffix = _status(task)
        runtime = format_runtime(self._runtime(task))
        lines: list[FragmentLine] = [
            overlay.title("Shell details"),
            overlay.rule(),
            overlay.field("Status", label, width=8, value_style=_status_style(suffix)),
            overlay.field("Runtime", runtime, width=8),
            overlay.field("PID", str(task.pid), width=8),
        ]
        if task.owner != "run_background":
            lines.append(overlay.field("Owner", task.owner, width=8))
        if not task.running and task.exit_code is not None:
            lines.append(overlay.field("Exit", str(task.exit_code), width=8))
        if task.note:
            lines.append(overlay.field("Note", task.note, width=8))
        lines.append(overlay.blank())
        lines.append([overlay.label("Command")])
        for wrapped in _wrap(_one_line(task.command), CONTENT_WIDTH - 2):
            lines.append([overlay.value(f"  {wrapped}")])
        lines.append(overlay.blank())
        display = self._output_display_lines()
        lines.append(
            [
                overlay.label("Output  "),
                overlay.value(
                    f"showing {len(display)} of {self._output_total} lines", theme.DIM
                ),
            ]
        )
        if display:
            for raw in display:
                lines.append([overlay.value(f"  {raw}")])
        else:
            lines.append([overlay.value("  (no output yet)", theme.DIM)])
        lines.append(overlay.blank())
        lines.append(overlay.hint("↑/↓ scroll · esc back"))
        return lines

    def _runtime(self, task: TaskSnapshot) -> float | None:
        if task.started_at is None:
            return None
        end = task.ended_at if task.ended_at is not None else self._now
        return max(0.0, end - task.started_at)


__all__ = [
    "CONTENT_WIDTH",
    "MAX_OUTPUT_LINES",
    "OUTPUT_TAIL_BYTES",
    "BackgroundTasksPanel",
    "TaskSnapshot",
    "format_runtime",
    "short_id",
]
