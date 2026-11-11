"""tui-only remote view commands."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from ...protocol.types import StreamEvent, StreamEventType
from ...remote.client import RemoteSessionClient
from ...remote.discovery import (
    RemoteSession,
    choose_session,
    configured_hosts,
    split_target,
)
from ...remote.wire import RemoteWireError
from .. import theme
from ..checkpoints import render_replayed_message
from ..transcript import TranscriptPresenter, TranscriptWidget


@dataclass
class RemotePicker:
    rows: tuple[RemoteSession, ...]
    index: int = 0

    def move(self, delta: int) -> None:
        self.index = max(0, min(len(self.rows) - 1, self.index + delta))

    @property
    def selected(self) -> RemoteSession:
        return self.rows[self.index]

    def render(self) -> Panel:
        lines = []
        for index, row in enumerate(self.rows):
            marker = "› " if index == self.index else "  "
            lines.append(Text(f"{marker}{row.host} · {row.name or row.session_id[:8]} · {row.state or 'unknown'}"))
        return Panel(Group(*lines), title="remote", subtitle="↑/↓ move · enter select · esc cancel")


@dataclass
class RemoteViewState:
    host: str
    session_id: str
    session_name: str
    transcript: TranscriptWidget
    presenter: TranscriptPresenter
    header: Window
    banner: Window
    container: HSplit
    disconnected: bool = False

    @classmethod
    def create(cls, app: Any, host: str, session_id: str, name: str) -> RemoteViewState:
        transcript = TranscriptWidget(max_lines=None)
        presenter = TranscriptPresenter(
            transcript, app.console, lambda: True, transcript.append
        )
        header = Window(FormattedTextControl(lambda: [
            ("class:header", f"remote · {host} · {name or session_id}")
        ]), height=1)
        state = cls(host, session_id, name, transcript, presenter, header, Window(
            FormattedTextControl(lambda: state._banner_text()), height=1
        ), HSplit([]))
        state.container.children[:] = [header, transcript.window()]
        return state

    def _banner_text(self) -> list[tuple[str, str]]:
        if not self.disconnected:
            return []
        return [("class:warning", f"disconnected from {self.host} · /detach or /remote to retry")]

    def show_disconnected(self) -> None:
        self.disconnected = True
        self.container.children[:] = [self.header, self.banner, self.transcript.window()]

    def render(self, event: StreamEvent) -> None:
        if event.type is StreamEventType.MESSAGE_START:
            self.presenter.reset_assistant_message()
        elif event.type is StreamEventType.MESSAGE_END and event.message is not None:
            render_replayed_message(
                event.message, presenter=self.presenter,
                print_user=lambda message: self.presenter.print_user(
                    Text(message.content[0].text if message.content else "", style=theme.BODY)
                ), print_unit=self.presenter.print_unit, tool_calls={},
                replay_tool_starts=True, replay_tool_results=True,
            )
        elif event.type is StreamEventType.MESSAGE_UPDATE:
            if event.delta:
                self.presenter.append_assistant(event.delta)
        elif event.type is StreamEventType.MESSAGE_END:
            self.presenter.finish_assistant_message(self.presenter.render_event(event))
        elif event.type in {
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_UPDATE,
            StreamEventType.TOOL_EXECUTION_END,
        }:
            self.presenter.handle_tool_event(event, aborted=False)
        else:
            rendered = self.presenter.render_event(event)
            if rendered is not None:
                self.presenter.print_unit(rendered)


class RemoteViewMixin:
    def _composer_placeholder(self) -> str:
        return "view only · /detach to return" if self._remote_view is not None else "type a message..."

    def _init_remote_view(self) -> None:
        self._remote_view = None
        self._remote_session_client = None
        self._remote_event_task = None
        self._remote_picker = None
        self._remote_picker_unit = None
        self._remote_saved_composer = None

    async def _close_remote_view(self) -> None:
        self.remote_picker_cancel()
        await self.slash_detach("")
        self._remote_session_client = None

    @property
    def remote_picker_active(self) -> bool:
        return self._remote_picker is not None

    def remote_picker_move(self, delta: int) -> None:
        if self._remote_picker is not None:
            self._remote_picker.move(delta)
            if self._remote_picker_unit is not None:
                self._remote_picker_unit.value = self._remote_picker.render()
            self._invalidate_prompt()

    def remote_picker_cancel(self) -> None:
        self._remote_picker = None
        if self._remote_picker_unit is not None:
            self._presenter.transcript.remove(self._remote_picker_unit)
        self._remote_picker_unit = None
        self._invalidate_prompt()

    def remote_picker_select(self) -> None:
        picker = self._remote_picker
        self.remote_picker_cancel()
        if picker is not None:
            asyncio.create_task(self._open_remote(self._remote_client(), picker.selected))

    async def _open_remote(self, client: RemoteSessionClient, chosen: RemoteSession) -> None:
        if self._remote_view is not None:
            # a second attach must stop the old consumer and restore the saved composer first.
            await self.slash_detach("")
        await client.attach(chosen.host, chosen)
        self._save_remote_composer()
        state = RemoteViewState.create(self, chosen.host, chosen.session_id, chosen.name)
        self._remote_view = state
        self._agent_navigation.show_external_view(state.container, state.transcript)
        self._remote_event_task = asyncio.create_task(self._consume_remote_events(state, client))
        self._invalidate_prompt()

    def _remote_client(self) -> RemoteSessionClient:
        if self._remote_session_client is None:
            self._remote_session_client = RemoteSessionClient(
                Path(self._zeta_home or Path.home() / ".zeta")
            )
        return self._remote_session_client

    def _save_remote_composer(self) -> None:
        session = self._active_session
        if session is None:
            return
        buffer = session.default_buffer
        self._remote_saved_composer = (
            Document(buffer.text, buffer.cursor_position),
            tuple(self._pending_attachments),
            self._next_image_token,
        )

    def _restore_remote_composer(self) -> None:
        saved = self._remote_saved_composer
        self._remote_saved_composer = None
        session = self._active_session
        if saved is None or session is None:
            return
        document, attachments, next_image_token = saved
        session.default_buffer.set_document(document, bypass_readonly=True)
        self._pending_attachments[:] = attachments
        self._next_image_token = next_image_token

    async def slash_remote(self, args: str) -> str:
        target = args.strip()
        client = self._remote_client()
        try:
            if target:
                host, prefix = split_target(target)
                configured = configured_hosts(Path(self._zeta_home or Path.home() / ".zeta"))
                configured_target = next(
                    (item for item in configured if item.name == host or item.host == host),
                    None,
                )
                rows = await client.discover((configured_target or host,))
            else:
                rows = await client.discover()
                host, prefix = None, None
            if not rows:
                return "no remote sessions found"
            if prefix is not None:
                chosen = choose_session(rows, prefix)
            elif len(rows) == 1:
                chosen = rows[0]
            else:
                self._remote_picker = RemotePicker(tuple(rows))
                self._remote_picker_unit = self._presenter.print_unit(self._remote_picker.render())
                self._invalidate_prompt()
                return ""
            await self._open_remote(client, chosen)
        except (RemoteWireError, ValueError) as exc:
            return str(exc)
        return ""

    async def _consume_remote_events(self, state: RemoteViewState, client: RemoteSessionClient) -> None:
        while self._remote_view is state:
            event = await client.next_event()
            if isinstance(event, str):
                if event == "disconnected":
                    state.show_disconnected()
                self._invalidate_prompt()
                if event == "disconnected":
                    return
                continue
            state.render(event)
            self._invalidate_prompt()

    async def slash_detach(self, args: str) -> str:
        task = self._remote_event_task
        self._remote_event_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._remote_session_client is not None:
            await self._remote_session_client.detach()
        self._agent_navigation.clear_external_view()
        self._restore_remote_composer()
        self._remote_view = None
        self._invalidate_prompt()
        return ""
