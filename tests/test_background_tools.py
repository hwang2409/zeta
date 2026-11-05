"""Host-environment coverage for both background spawn branches."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zeta.core.approval import ApprovedCwdExecution
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolAbortSignal, ToolRegistry
from zeta.tools.bash import _bash
from zeta.tools.registry import ToolExecutionContext
from zeta.tools.zeta_background import _run_background


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

        identity = os.stat(tmp_path)
        approved = ApprovedCwdExecution(
            str(tmp_path), (identity.st_dev, identity.st_ino)
        )
        approved_context = ToolExecutionContext(
            ToolCall("host-bg-approved", "run_background", {}),
            None,
            None,
            approved,
            {"GAMMA_SENTINEL": "approved-dedicated-pass"},
        )
        approved_dedicated = await _run_background(
            registry,
            {"command": 'printf %s "$GAMMA_SENTINEL"'},
            execution_context=approved_context,
        )
        assert "approved-dedicated-pass" in await _output(
            registry, approved_dedicated["structuredContent"]["task_id"]
        )

        approved_macro_context = ToolExecutionContext(
            ToolCall("host-bg-approved-macro", "bash", {}),
            None,
            None,
            approved,
            {"GAMMA_SENTINEL": "approved-macro-pass"},
        )
        approved_macro = await _bash(
            registry,
            {
                "command": 'printf %s "$GAMMA_SENTINEL"',
                "_background": True,
            },
            ToolAbortSignal(),
            execution_context=approved_macro_context,
        )
        assert "approved-macro-pass" in await _output(
            registry, approved_macro["structuredContent"]["task_id"]
        )
    finally:
        await registry.close()
