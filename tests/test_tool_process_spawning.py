from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SENTINEL = "ATFORK_ESCAPE"


@pytest.mark.parametrize("spawn_path", ["bash", "run_background", "approved_cwd"])
def test_shell_spawn_does_not_leak_atfork_stderr_to_parent_fd2(
    tmp_path: Path, spawn_path: str
) -> None:
    helper = textwrap.dedent(
        f"""
        import asyncio
        import json
        import os
        from pathlib import Path

        from zeta.protocol.types import ToolCall
        from zeta.skills import SkillCatalog
        from zeta.tools import ToolRegistry
        from zeta.tools._shared.background_spawn import create_subprocess_shell_in_fd

        os.register_at_fork(
            after_in_child=lambda: os.write(2, b"{_SENTINEL}\\n")
        )

        async def main():
            root = Path({str(tmp_path)!r})
            if {spawn_path!r} == "approved_cwd":
                cwd_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    process = await create_subprocess_shell_in_fd(
                        "printf child-output",
                        cwd_fd,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=os.environ.copy(),
                        niceness=10,
                    )
                    stdout, stderr = await process.communicate()
                    output = stdout.decode()
                    child_stderr = stderr.decode()
                finally:
                    os.close(cwd_fd)
            else:
                registry = ToolRegistry(
                    root, skill_catalog=SkillCatalog.empty()
                )
                try:
                    result = await registry.execute(
                        ToolCall("spawn", {spawn_path!r}, {{"command": "printf child-output"}})
                    )
                    if {spawn_path!r} == "bash":
                        output = result["structuredContent"]["stdout"]
                    else:
                        task_id = result["structuredContent"]["task_id"]
                        await registry.background_tasks.wait(task_id)
                        result = await registry.execute(
                            ToolCall("output", "task_output", {{"task_id": task_id}})
                        )
                        output = result["structuredContent"]["output"]
                    child_stderr = ""
                finally:
                    await registry.close()
            print(json.dumps({{"output": output, "stderr": child_stderr}}))

        asyncio.run(main())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", helper],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result == {"output": "child-output", "stderr": ""}
