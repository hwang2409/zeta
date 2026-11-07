import asyncio
import shutil
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, MessageRole, TextContent, ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def _execute(registry: ToolRegistry, call: ToolCall) -> dict[str, object]:
    return await asyncio.wait_for(registry.execute(call), 10)


async def _collect(loop: AgentLoop) -> list[object]:
    async def collect() -> list[object]:
        return [
            event
            async for event in loop.run_turn("run", origin=MessageOrigin.USER)
        ]

    return await asyncio.wait_for(collect(), 10)


@pytest.mark.asyncio
async def test_bash_explicit_cwd_survives_removed_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "worktree"
    explicit_cwd = tmp_path / "survivor"
    session_cwd.mkdir()
    explicit_cwd.mkdir()
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    try:
        shutil.rmtree(session_cwd)
        result = await _execute(
            registry,
            ToolCall(
                "explicit-bash",
                "bash",
                {"command": "printf expected", "cwd": str(explicit_cwd)},
            ),
        )

        assert result["structuredContent"]["stdout"] == "expected"
        assert result["structuredContent"]["cwd_after"] == str(explicit_cwd)
    finally:
        await asyncio.wait_for(registry.close(), 10)


@pytest.mark.asyncio
async def test_bash_falls_back_once_when_persisted_cwd_was_removed(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "worktree"
    session_cwd.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=session_cwd)
    registry = ToolRegistry(
        session_cwd,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
    )
    calls = [
        ToolCall("fallback-1", "bash", {"command": "pwd"}),
        ToolCall("fallback-2", "bash", {"command": "pwd"}),
    ]
    loop = AgentLoop(
        FakeBackend(
            [
                ScriptedTurn(tool_calls=calls),
                ScriptedTurn(content=[TextContent("done")]),
            ]
        ),
        store,
        registry=registry,
        skill_catalog=SkillCatalog.empty(),
    )

    try:
        shutil.rmtree(session_cwd)
        await _collect(loop)

        results = {
            message.tool_result.tool_call_id: message.tool_result
            for message in store.messages()
            if message.tool_result is not None
        }
        notices = [
            message
            for message in store.messages()
            if message.metadata.get("zeta_event") == "shell_cwd_recovered"
        ]
        assert results["fallback-1"].structured_content["stdout"].strip() == str(
            tmp_path
        )
        assert results["fallback-2"].structured_content["stdout"].strip() == str(
            tmp_path
        )
        assert store.bash_cwd == str(tmp_path)
        assert registry.bash_cwd == str(tmp_path)
        assert len(notices) == 1
        assert notices[0].role is MessageRole.SYSTEM
        assert notices[0].content[0].text == (
            f"[zeta] shell cwd: {session_cwd} was removed; continuing in {tmp_path}"
        )
    finally:
        await asyncio.wait_for(loop.close(), 10)


@pytest.mark.asyncio
async def test_run_background_explicit_cwd_survives_removed_session_cwd(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "worktree"
    explicit_cwd = tmp_path / "survivor"
    session_cwd.mkdir()
    explicit_cwd.mkdir()
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    try:
        shutil.rmtree(session_cwd)
        started = await _execute(
            registry,
            ToolCall(
                "start",
                "run_background",
                {"command": "printf completed", "cwd": str(explicit_cwd)},
            ),
        )
        task_id = started["structuredContent"]["task_id"]
        await asyncio.wait_for(registry.background_tasks.wait(task_id), 10)
        completed = await _execute(
            registry,
            ToolCall("output", "task_output", {"task_id": task_id}),
        )

        assert completed["structuredContent"]["running"] is False
        assert completed["structuredContent"]["exit_code"] == 0
        assert completed["structuredContent"]["output"] == "completed"
    finally:
        await asyncio.wait_for(registry.close(), 10)


@pytest.mark.asyncio
async def test_replaced_session_cwd_still_fails_closed(tmp_path: Path) -> None:
    session_cwd = tmp_path / "worktree"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    try:
        shutil.rmtree(session_cwd)
        session_cwd.mkdir()
        bash_result = await _execute(
            registry, ToolCall("bash", "bash", {"command": "pwd"})
        )
        read_result = await _execute(
            registry, ToolCall("read", "read", {"path": "relative.txt"})
        )

        assert bash_result["isError"] is True
        assert bash_result["structuredContent"]["error"]["message"] == (
            "session cwd was replaced"
        )
        assert read_result["isError"] is True
        assert read_result["structuredContent"]["error"]["message"] == (
            "session cwd was replaced"
        )
    finally:
        await asyncio.wait_for(registry.close(), 10)


@pytest.mark.asyncio
async def test_removed_session_cwd_blocks_only_relative_file_paths(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "worktree"
    absolute_file = tmp_path / "outside.txt"
    session_cwd.mkdir()
    absolute_file.write_text("outside")
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    try:
        shutil.rmtree(session_cwd)
        relative = await _execute(
            registry, ToolCall("relative", "read", {"path": "inside.txt"})
        )
        absolute = await _execute(
            registry, ToolCall("absolute", "read", {"path": str(absolute_file)})
        )

        assert relative["isError"] is True
        assert relative["structuredContent"]["error"]["message"] == (
            f"session cwd {session_cwd} was removed; use an absolute path"
        )
        assert absolute["isError"] is False
        assert absolute["content"][0]["text"] == "outside"
    finally:
        await asyncio.wait_for(registry.close(), 10)
