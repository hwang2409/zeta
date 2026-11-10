"""TUI-only remote view commands."""
from __future__ import annotations

from pathlib import Path

from ...remote.client import RemoteSessionClient
from ...remote.discovery import choose_session, split_target


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
        self._print_system(f"remote · {chosen.host} · {chosen.name or chosen.session_id}")
        return "view only"

    def slash_detach(self, args: str) -> str:
        client = getattr(self, "_remote_session_client", None)
        if client is not None:
            client.detach()
        self._remote_view = None
        return ""
