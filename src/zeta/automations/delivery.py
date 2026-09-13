"""Deterministic, recipient-bound Slack delivery outside model tool permissions."""

from __future__ import annotations

import json
import re
from typing import Protocol

from ..core.abort import AbortSignal
from ..mcp.mount import MCPMount
from ..types import StructuredToolResult

SLACK_ID = re.compile(r"[CUGD][A-Z0-9]{2,}")
MESSAGE_LIMIT = 4500


class Delivery(Protocol):
    async def resolve(self, target: str) -> str: ...
    async def send(
        self, recipient: str, name: str, session_id: str, text: str
    ) -> str: ...


class SlackDelivery:
    def __init__(self, mount: MCPMount) -> None:
        self.mount = mount

    async def _call(
        self, tool: str, arguments: dict[str, object]
    ) -> StructuredToolResult:
        client = self.mount.client_for("slack")
        if client is None:
            raise ValueError("Slack is not connected")
        result = await client.call_tool(tool, arguments, AbortSignal())
        if result.get("isError"):
            raise ValueError(
                f"Slack {tool} failed: {json.dumps(result, ensure_ascii=False)[:1000]}"
            )
        return result

    async def resolve(self, target: str) -> str:
        value = target.removeprefix("slack:")
        if SLACK_ID.fullmatch(value):
            return value
        if value.startswith("@"):
            tool, query = "slack_search_users", value[1:]
            prefix = "U"
        elif value.startswith("#"):
            tool, query = "slack_search_channels", value[1:]
            prefix = "CG"
        else:
            raise ValueError(
                "Slack recipient must be @name, #channel, or an explicit Slack ID"
            )
        result = await self._call(tool, {"query": query})
        # Official Slack tools may return structured objects or textual search results.
        # Refuse any result with multiple candidate IDs rather than guessing a recipient.
        text = json.dumps(result, ensure_ascii=False)
        candidates = set(re.findall(r"\b[" + prefix + r"][A-Z0-9]{2,}\b", text))
        if len(candidates) != 1:
            raise ValueError(
                "Slack recipient is missing or ambiguous; supply an explicit Slack ID"
            )
        return candidates.pop()

    async def send(self, recipient: str, name: str, session_id: str, text: str) -> str:
        if not SLACK_ID.fullmatch(recipient):
            raise ValueError("delivery requires an approved Slack ID")
        heading = f"{name}\nSession: {session_id}\n\n"
        suffix = "\n[Truncated; full response in the zeta session.]"
        if len(heading) + len(text) > MESSAGE_LIMIT:
            text = text[: MESSAGE_LIMIT - len(heading) - len(suffix)] + suffix
        result = await self._call(
            "slack_send_message", {"channel_id": recipient, "message": heading + text}
        )
        return json.dumps(result, ensure_ascii=False)
