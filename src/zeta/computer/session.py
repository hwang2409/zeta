"""Turn a Zeta session into a computer-use session.

A computer session has exactly one tool surface: the computer MCP server. The
tool policy is the exact list of computer tools, so every host tool (files,
shell, agents, user MCP servers, external tools) is hidden and rejected by the
tool-policy machinery, and command hooks are disabled by the restricted-session
gate. The desktop starts on the first tool call and is removed at session end.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.approval import ApprovalPolicy, ApprovalRule
from ..mcp.config import MCPConfig, MCPServerConfig
from .backend import backend_kind
from .recording import mark_finished
from .settings import ComputerSettings, load_computer_settings
from .spectate import Spectator
from .tools import QUALIFIED_TOOL_NAMES, SERVER_NAME

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop

TOOLS_ARGUMENT = ",".join(QUALIFIED_TOOL_NAMES)
RECORDING_DIRECTORY = "computer"


def prepare_args(args: argparse.Namespace) -> None:
    """Apply the computer tool policy to parsed CLI arguments.

    Sets ``--tools`` to the exact computer tool names and turns on
    ``--require-tools``. Raises ``ValueError`` when the caller also selected a
    different allowlist. ``--disallowed-tools`` still narrows the set.
    """

    if not getattr(args, "computer", False):
        return
    tools = getattr(args, "tools", None)
    if tools is not None and tools != TOOLS_ARGUMENT:
        raise ValueError("--computer sets the tool allowlist; do not combine it with --tools")
    args.tools = TOOLS_ARGUMENT
    args.require_tools = True


def restart_args(args: argparse.Namespace, *, session_id: str, provider: str, model: str) -> None:
    """Rewrite TUI arguments to reopen ``session_id`` as a computer session."""

    args.computer = True
    args.tools = None
    args.continue_session = False
    args.resume = session_id
    args.provider = provider
    args.model = model
    prepare_args(args)


def server_config(
    *, session_id: str, backend: str, recording: Path | None, desktop_minutes: int
) -> MCPServerConfig:
    env = {
        "ZETA_COMPUTER_SESSION": session_id,
        "ZETA_COMPUTER_BACKEND": backend,
        "ZETA_COMPUTER_TTL_SECONDS": str(desktop_minutes * 60),
    }
    if recording is not None:
        env["ZETA_COMPUTER_RECORDING_DIR"] = str(recording)
    return MCPServerConfig(
        name=SERVER_NAME,
        transport="stdio",
        command=sys.executable,
        args=("-m", "zeta.computer.server"),
        env=env,
    )


@dataclass(slots=True)
class ComputerSession:
    """The computer state of one Zeta session, owned by its loop."""

    session_id: str
    backend: str
    recording: Path | None
    spectator: Spectator | None

    @classmethod
    def attach(
        cls,
        loop: AgentLoop,
        approval_policy: ApprovalPolicy,
        *,
        home: Path,
        backend: str | None = None,
    ) -> ComputerSession:
        """Select the computer MCP server for ``loop`` and own its cleanup.

        Call before the loop activates. The approval policy allows the
        computer tools, which act only inside the sandbox; configured deny and
        ask rules still take precedence.
        """

        settings: ComputerSettings = load_computer_settings(home)
        selected = backend or settings.backend
        backend_kind(selected)
        store = loop.store
        recording = store.session_dir / RECORDING_DIRECTORY if settings.recording else None
        config = server_config(
            session_id=store.session_id,
            backend=selected,
            recording=recording,
            desktop_minutes=settings.desktop_minutes,
        )
        loop.select_mcp_config(
            MCPConfig(
                path=store.session_dir,
                servers={SERVER_NAME: config},
                sources={SERVER_NAME: store.session_dir},
            )
        )
        # Computer actions are sandboxed, but they are still subject to the
        # user's explicit deny/ask rules.  Headless mode converts matching ask
        # rules to hard denials before it removes prompts.
        approval_policy.always_allow = {
            *approval_policy.always_allow,
            *(
                ApprovalRule(name)
                for name in QUALIFIED_TOOL_NAMES
                if not any(rule.tool == name for rule in approval_policy.always_ask)
            ),
        }
        spectator = None
        if recording is not None:
            recording.mkdir(parents=True, exist_ok=True)
            spectator = Spectator(recording)
        session = cls(store.session_id, selected, recording, spectator)
        loop.tool_registry.add_cleanup(session.close)
        return session

    @property
    def notices(self) -> tuple[str, ...]:
        lines = [
            f"computer · {self.backend} sandbox desktop starts on first use; host tools are hidden",
        ]
        if self.spectator is not None:
            lines.append(f"computer · spectator {self.spectator.url}")
        else:
            lines.append("computer · recording is off (settings computer.recording)")
        lines.append(f"computer · live view: zeta computer watch --live {self.session_id}")
        return tuple(lines)

    def stop_spectator(self) -> None:
        spectator, self.spectator = self.spectator, None
        if spectator is not None:
            spectator.stop()

    async def close(self) -> None:
        """Remove the session's desktops and close the recording."""

        try:
            await asyncio.to_thread(backend_kind(self.backend).remove_session, self.session_id)
        finally:
            if self.recording is not None:
                mark_finished(self.recording)
            self.stop_spectator()


__all__ = ["TOOLS_ARGUMENT", "ComputerSession", "prepare_args", "restart_args", "server_config"]
