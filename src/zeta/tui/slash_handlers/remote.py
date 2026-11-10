"""TUI-only remote view commands."""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...protocol.types import StreamEvent, StreamEventType
from ...remote.client import RemoteSessionClient
from ...remote.discovery import choose_session, split_target
from ..render import render_event


class RemoteViewMixin:
    def _remote_client(self) -> RemoteSessionClient:
        client = getattr(self, "_remote_session_client", None)
        if client is None:
            client = self._remote_session_client = RemoteSessionClient(
                Path(self._zeta_home or Path.home() / ".zeta")
            )
        return client

    async def slash_remote(self, args: str) -> str:
        target = args.strip()
        client = self._remote_client()
        if not target:
            rows = await client.discover()
            if not rows:
                return "no remote sessions found"
            return "\n".join(f"{row.host}  {row.name or row.session_id[:8]}  {row.state or 'unknown'}" for row in rows)
        host, prefix = split_target(target)
        rows = await client.discover((host,))
        if prefix is None:
            if len(rows) != 1:
                if not rows:
                    return "no remote sessions found"
                return "\n".join(f"{index + 1}. {row.name or row.session_id[:8]}  {row.state or 'unknown'}" for index, row in enumerate(rows))
            chosen = rows[0]
        else:
            try:
                chosen = choose_session(rows, prefix)
            except ValueError as exc:
                return str(exc)
        await client.attach(host, chosen)
        self._remote_view = client.view
        old_task = getattr(self, "_remote_poll_task", None)
        if old_task is not None:
            old_task.cancel()
        self._remote_poll_task = asyncio.create_task(self._poll_remote())
        self._print_system(f"remote · {chosen.host} · {chosen.name or chosen.session_id}")
        return "view only"

    async def _poll_remote(self) -> None:
        client = self._remote_client()
        while client.view is not None:
            for event in client.consume():
                self._render_remote_event(event)
            await asyncio.sleep(0.05)

    def _render_remote_event(self, event: StreamEvent) -> None:
        self._update_usage(event)
        self._prepare_stream_event(event)
        self._handle_tool_event(event)
        if event.type is StreamEventType.MESSAGE_UPDATE:
            self._consume_text(event)
        elif event.type is StreamEventType.MESSAGE_END:
            self._finish_message(event)
        else:
            rendered = render_event(event)
            if rendered is not None:
                self._print(rendered)
        self._invalidate_prompt()

    def slash_detach(self, args: str) -> str:
        client = getattr(self, "_remote_session_client", None)
        task = getattr(self, "_remote_poll_task", None)
        if task is not None:
            task.cancel()
            self._remote_poll_task = None
        if client is not None:
            client.detach()
        self._remote_view = None
        return ""
