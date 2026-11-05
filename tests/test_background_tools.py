"""Host-environment coverage for both background spawn branches."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def _output(registry: ToolRegistry, task_id: str) -> str:
    await registry.background_tasks.wait(task_id)
    result = await registry.execute(
        ToolCall("host-bg-output", "task_output", {"task_id": task_id})
    )
    return result["structuredContent"]["output"]


@pytest.mark.asyncio
async def test_session_host_commands_reach_run_background_spawns(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        dedicated = await registry.execute(
            ToolCall(
                "host-bg-dedicated",
                "run_background",
                {"command": 'printf %s "$GAMMA_SENTINEL"'},
            ),
            _host_environment={"GAMMA_SENTINEL": "dedicated-pass"},
        )
        assert "dedicated-pass" in await _output(
            registry, dedicated["structuredContent"]["task_id"]
        )

        macro = await registry.execute(
            ToolCall(
                "host-bg-macro",
                "bash",
                {"command": 'printf %s "$GAMMA_SENTINEL"'},
            ),
            _background=True,
            _host_environment={"GAMMA_SENTINEL": "macro-pass"},
        )
        assert "macro-pass" in await _output(
            registry, macro["structuredContent"]["task_id"]
        )

        cwd_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        task_id, _ = await registry.background_tasks.start(
            'printf %s "$GAMMA_SENTINEL"',
            tmp_path,
            cwd_fd=cwd_fd,
            environment={"GAMMA_SENTINEL": "approved-pass"},
        )
        assert "approved-pass" in await _output(registry, task_id)
    finally:
        await registry.close()
