import asyncio
import json
import shlex
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.agent.tool_results import validated_tool_result
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall
from zeta.server.event_contract import tool_end
from zeta.server.protocol import EventDescriptor, FrameCodec
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy
from zeta.tui.render import render_event


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["edit", "write", "read"])
async def test_relative_file_tools_follow_bash_cd(
    tmp_path: Path, tool_name: str
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    start_file = tmp_path / "data.txt"
    sub_file = sub / "data.txt"
    start_file.write_text("original in start cwd")
    sub_file.write_text("original in sub")
    arguments = {"path": "data.txt"}
    if tool_name == "edit":
        arguments.update(old_string="original", new_string="edited")
    elif tool_name == "write":
        arguments["content"] = "written"

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        changed = await registry.execute(ToolCall("cd", "bash", {"command": "cd sub"}))
        assert changed["structuredContent"]["cwd_after"] == str(sub)
        call = ToolCall("file", tool_name, arguments)
        result = await registry.execute(call)

        if tool_name == "edit":
            assert sub_file.read_text() == "edited in sub"
        elif tool_name == "write":
            assert sub_file.read_text() == "written"
        else:
            assert "original in sub" in result["content"][0]["text"]
        assert start_file.read_text() == "original in start cwd"
        assert result["structuredContent"]["path"] == str(sub_file)

        event = StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=validated_tool_result(result, call.id),
        )
        output = StringIO()
        Console(file=output, width=120, color_system=None).print(render_event(event))
        assert "data.txt" in output.getvalue()
        frame = FrameCodec().notification(
            EventDescriptor.TOOL_END, "session", **tool_end(event, fidelity=True)
        )
        payload = json.loads(frame)["params"]
        assert payload["tool_result"]["structured_content"]["path"] == str(sub_file)
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_with_file", [False, True])
async def test_file_tools_do_not_recover_invalid_bash_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_with_file: bool
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    start_file = tmp_path / "data.txt"
    start_file.write_text("unchanged")
    monkeypatch.setenv("HOME", str(tmp_path))
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        changed = await registry.execute(ToolCall("cd", "bash", {"command": "cd sub"}))
        assert changed["structuredContent"]["cwd_after"] == str(sub)
        sub.rmdir()
        if replace_with_file:
            sub.write_text("not a directory")

        for tool_name, arguments in (
            ("read", {"path": "../data.txt"}),
            ("edit", {"path": "data.txt", "old_string": "unchanged", "new_string": "wrong"}),
            ("write", {"path": "nested/data.txt", "content": "wrong", "create_parents": True}),
        ):
            result = await registry.execute(ToolCall(tool_name, tool_name, arguments))
            assert result["isError"]
            assert f"bash session cwd {str(sub)!r}" in result["content"][0]["text"]
        assert start_file.read_text() == "unchanged"
        assert not (tmp_path / "nested").exists()
        assert sub.exists() == replace_with_file
        assert registry.bash_cwd == str(sub)

        for path in (str(start_file), "~/data.txt"):
            result = await registry.execute(ToolCall("absolute", "read", {"path": path}))
            assert not result["isError"]
            assert result["structuredContent"]["path"] == str(start_file)
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cloned_child", [False, True], ids=["session", "child"])
async def test_relative_cwd_override_uses_persisted_bash_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloned_child: bool,
) -> None:
    process_cwd = tmp_path / "process"
    persisted_cwd = tmp_path / "persisted"
    (process_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    monkeypatch.chdir(process_cwd)

    parent_store = ConversationStore(tmp_path / "sessions", cwd=process_cwd)
    parent = ToolRegistry(
        process_cwd,
        session_store=parent_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry = parent
    if cloned_child:
        child_store = ConversationStore(
            parent_store.session_dir / "agents",
            session_id="1",
            cwd=process_cwd,
        )
        registry = parent.clone_for_session(child_store)

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            )
        )
        result = await registry.execute(
            ToolCall("relative-cwd", "bash", {"command": "pwd", "cwd": "child"})
        )

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert result["structuredContent"]["stdout"].strip() == str(
            persisted_cwd / "child"
        )
    finally:
        if registry is not parent:
            await registry.close()
        await parent.close()


@pytest.mark.asyncio
async def test_run_background_relative_cwd_uses_persisted_bash_cwd(
    tmp_path: Path,
) -> None:
    process_cwd = tmp_path / "process"
    persisted_cwd = tmp_path / "persisted"
    (process_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    registry = ToolRegistry(process_cwd, skill_catalog=SkillCatalog.empty())

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            )
        )
        started = await registry.execute(
            ToolCall(
                "relative-background-cwd",
                "run_background",
                {"command": "pwd", "cwd": "child"},
            )
        )
        task_id = started["structuredContent"]["task_id"]
        await registry.background_tasks.wait(task_id)
        result = await registry.execute(
            ToolCall("background-output", "task_output", {"task_id": task_id})
        )

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert result["structuredContent"]["output"].strip() == str(
            persisted_cwd / "child"
        )
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
async def test_delegated_shell_approval_uses_persisted_bash_cwd(
    tmp_path: Path,
    tool_name: str,
) -> None:
    initial_cwd = tmp_path / "initial"
    persisted_cwd = tmp_path / "persisted"
    (initial_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    parent_store = ConversationStore(tmp_path / "parent", cwd=initial_cwd)
    child_store = ConversationStore(tmp_path / "child", cwd=initial_cwd)
    parent_policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=initial_cwd,
        child_cwd=initial_cwd,
    )
    registry = ToolRegistry(
        initial_cwd,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
        approval_policy=child_policy,
    )

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            ),
            _skip_approval=True,
        )
        call = ToolCall(
            "relative-cwd",
            tool_name,
            {"command": "pwd", "cwd": "child"},
        )
        execution = asyncio.create_task(registry.execute(call))
        async with asyncio.timeout(10):
            while not (await asyncio.wait_for(parent_policy.pending_requests(), 10)):
                await asyncio.sleep(0)

        request = (await asyncio.wait_for(parent_policy.pending_requests(), 10))[0]
        assert parent_policy.approve(request.key)
        result = await asyncio.wait_for(execution, 10)
        if tool_name == "run_background":
            task_id = result["structuredContent"]["task_id"]
            await asyncio.wait_for(registry.background_tasks.wait(task_id), 10)
            result = await registry.execute(
                ToolCall("background-output", "task_output", {"task_id": task_id})
            )
            actual_cwd = result["structuredContent"]["output"].strip()
        else:
            actual_cwd = result["structuredContent"]["stdout"].strip()

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert actual_cwd == str(persisted_cwd / "child")
        assert request.effective_cwd == str(persisted_cwd / "child")
        assert request.audit_display()["effective_cwd"] == str(
            persisted_cwd / "child"
        )
    finally:
        await registry.close()


@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
def test_shell_schema_explains_cwd_resolution(
    tmp_path: Path, tool_name: str
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    description = registry.definitions_by_name[tool_name].parameters["properties"][
        "cwd"
    ].get("description", "")

    assert "persisted session cwd" in description
    assert "Absolute paths are used as given" in description
