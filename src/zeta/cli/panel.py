"""Read-only attention panel CLI and full-screen terminal view."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from ..attention import (
    AttentionRecord,
    PanelSnapshot,
    create_discussion_fork,
    panel_snapshot,
)
from ..core.session import env_home
from ..tui import theme


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "panel", help="show live orchestrators and attention requests"
    )
    parser.add_argument(
        "--list", action="store_true", help="print a plain text summary"
    )


def _age(timestamp: str) -> str:
    try:
        seconds = max(
            0,
            int(
                (datetime.now(UTC) - datetime.fromisoformat(timestamp)).total_seconds()
            ),
        )
    except ValueError:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _lane_text(lane) -> str:
    elapsed = (
        f", {int(lane.elapsed_seconds)}s" if lane.elapsed_seconds is not None else ""
    )
    return f"{lane.kind}: {lane.label} ({lane.status}{elapsed})"


def format_snapshot(snapshot: PanelSnapshot) -> str:
    lines: list[str] = []
    for project in snapshot.projects:
        lines.append(project.name)
        for session in project.sessions:
            lines.append(
                f"  {session.name} [{session.session_id[:8]}] active {_age(session.updated_at)}"
            )
            for lane in (*session.lanes, *session.tasks):
                lines.append(f"    {_lane_text(lane)}")
            for record in session.attention:
                marker = "!" if record.status == "open" else "✓"
                lines.append(
                    f"    {marker} {record.title} ({record.status}, {_age(record.created_at)})"
                )
    return "\n".join(lines) + ("\n" if lines else "No live orchestrators.\n")


class PanelApplication:
    """Keep all storage reads off the prompt_toolkit event loop."""

    def __init__(self, home: Path):
        self.home = home
        self.snapshot = PanelSnapshot(())
        self.selected = 0
        self._attention: list[tuple[str, AttentionRecord]] = []
        self._app: Application[None] | None = None

    async def refresh(self) -> None:
        self.snapshot = await asyncio.to_thread(panel_snapshot, self.home)
        self._attention = [
            (session.session_id, record)
            for project in self.snapshot.projects
            for session in project.sessions
            for record in session.attention
        ]
        self.selected = min(self.selected, max(0, len(self._attention) - 1))
        if self._app is not None:
            self._app.invalidate()

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self.refresh()

    def _render(self) -> FormattedText:
        rows: list[tuple[str, str]] = [("class:title", "Zeta attention panel\n\n")]
        attention_index = 0
        for project in self.snapshot.projects:
            rows.append(("class:project", f"{project.name}\n"))
            for session in project.sessions:
                rows.append(("", f"  {session.name} [{session.session_id[:8]}]\n"))
                for lane in (*session.lanes, *session.tasks):
                    rows.append(
                        (
                            "class:dim",
                            f"    {_lane_text(lane)}\n",
                        )
                    )
                for record in session.attention:
                    selected = attention_index == self.selected
                    style = (
                        "class:selected"
                        if selected
                        else (
                            "class:dim"
                            if record.status == "resolved"
                            else "class:attention"
                        )
                    )
                    marker = "!" if record.status == "open" else "✓"
                    rows.append(
                        (
                            style,
                            f"  {'>' if selected else ' '} {marker} {record.title} ({_age(record.created_at)})\n",
                        )
                    )
                    attention_index += 1
        if not self.snapshot.projects:
            rows.append(("class:dim", "No live orchestrators.\n"))
        rows.append(("class:dim", "\n↑/↓ or j/k move  Enter open  r refresh  q quit"))
        return FormattedText(rows)

    async def run(self) -> int:
        await self.refresh()
        keys = KeyBindings()

        @keys.add("q")
        def _quit(event) -> None:
            event.app.exit()

        @keys.add("down")
        @keys.add("j")
        def _down(event) -> None:
            if self._attention:
                self.selected = min(self.selected + 1, len(self._attention) - 1)

        @keys.add("up")
        @keys.add("k")
        def _up(event) -> None:
            self.selected = max(0, self.selected - 1)

        @keys.add("r")
        def _refresh(event) -> None:
            event.app.create_background_task(self.refresh())

        @keys.add("enter")
        def _open(event) -> None:
            if not self._attention:
                return
            source_id, record = self._attention[self.selected]
            fork_id = create_discussion_fork(self.home, source_id, record.id)
            event.app.exit(result=fork_id)

        style = Style.from_dict(
            {
                "title": f"bold {theme.AGENT_MAIN}",
                "project": f"bold {theme.BODY}",
                "dim": theme.DIM,
                "attention": theme.WARNING,
                "selected": f"reverse {theme.BODY}",
            }
        )
        self._app = Application(
            layout=Layout(Window(FormattedTextControl(self._render))),
            key_bindings=keys,
            style=style,
            full_screen=True,
        )
        fork_id = await self._app.run_async(
            pre_run=lambda: self._app.create_background_task(self._ticker())
        )
        if isinstance(fork_id, str):
            os.execvp("zeta", ["zeta", "--resume", fork_id])
        return 0


def run(args: argparse.Namespace) -> int:
    home = env_home()
    if args.list:
        print(format_snapshot(panel_snapshot(home)), end="")
        return 0
    try:
        return asyncio.run(PanelApplication(home).run())
    except KeyboardInterrupt:
        return 130


__all__ = ["PanelApplication", "add_subcommand", "format_snapshot", "run"]
