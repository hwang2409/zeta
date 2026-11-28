"""The bash tool honors a `timeout N` inside the command when no tool timeout is set (F066).

The commands define a shell function named `timeout`, so the tests do not need
coreutils (stock macOS has none). The function ignores N on purpose: the only
limit left is the tool timeout, which is what these tests observe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import zeta.tools.bash as bash_tool
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry

_IGNORE_N = 'timeout() { shift; "$@"; }; '


async def _run(tmp_path: Path, arguments: dict[str, Any]) -> tuple[str, dict]:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    try:
        registry = ToolRegistry(
            tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
        )
        try:
            call = asyncio.ensure_future(
                registry.execute(ToolCall("inner", "bash", arguments))
            )
            try:
                done, _ = await asyncio.wait({call}, timeout=10)
                assert done, "bash kept the 30 s default instead of the inner timeout"
                result = call.result()
            finally:
                call.cancel()
                await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), 10)
        finally:
            await registry.close()
    finally:
        store.close()
    return result["content"][0]["text"], result


@pytest.mark.asyncio
async def test_inner_timeout_sets_the_tool_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the tool kills at N plus the 5 s margin, far before the 30 s default
    # test-contract: seam a small default proves the inner timeout raises it
    monkeypatch.setattr(bash_tool, "_DEFAULT_TIMEOUT_SECONDS", 0.2)
    _, result = await _run(tmp_path, {"command": _IGNORE_N + "timeout 0.3 sleep 30"})

    assert result["structuredContent"]["timed_out"] is True
    assert result["structuredContent"]["timeout_seconds"] == 5.3


@pytest.mark.asyncio
async def test_inner_timeout_longer_than_default_is_not_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test-contract: seam the 30 s default cannot be waited out in a unit test
    monkeypatch.setattr(bash_tool, "_DEFAULT_TIMEOUT_SECONDS", 0.2)
    text, result = await _run(
        tmp_path, {"command": _IGNORE_N + "timeout 2 sleep 0.5; echo done"}
    )

    assert result["isError"] is False, text
    assert result["structuredContent"]["stdout"] == "done\n"


@pytest.mark.asyncio
async def test_short_inner_timeout_does_not_lower_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test-contract: seam small values keep the test fast; the ratio matches 2 s vs 30 s
    monkeypatch.setattr(bash_tool, "_DEFAULT_TIMEOUT_SECONDS", 1.0)
    # test-contract: seam the 5 s margin is too slow to wait out here
    monkeypatch.setattr(bash_tool, "_INNER_TIMEOUT_MARGIN_SECONDS", 0.1)
    text, result = await _run(
        tmp_path, {"command": _IGNORE_N + "timeout 0.1 true && sleep 0.5 && echo done"}
    )

    assert result["isError"] is False, text
    assert result["structuredContent"]["stdout"] == "done\n"


@pytest.mark.asyncio
async def test_explicit_tool_timeout_wins(tmp_path: Path) -> None:
    _, result = await _run(
        tmp_path, {"command": _IGNORE_N + "timeout 30 sleep 30", "timeout": 0.2}
    )

    assert result["structuredContent"]["timed_out"] is True
    assert result["structuredContent"]["timeout_seconds"] == 0.2


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("timeout 45 sleep 40", 50.0),
        ("timeout 2m du -sh .", 125.0),
        ("timeout 1.5h x", 3600.0),
        ("timeout 1d x", 3600.0),
        ("timeout 4000 x", 3600.0),
        ("timeout -k 5 -s KILL 40 x", 50.0),
        ("timeout --kill-after=1m --signal=TERM 30s x", 95.0),
        ("timeout -k5 --preserve-status --foreground -v 40 x", 50.0),
        ("cd /tmp && timeout 120 du -xsh .", 125.0),
        ("timeout 10 a; timeout 60 b", 65.0),
        ("timeout 2 true && sleep 60", 30.0),  # never lower the default
        ("FOO=1 timeout 60 x", 30.0),  # not parsed: keep the default
        ("timeout abc x", 30.0),
        ("timeout 0 x", 30.0),  # 0 means no limit in coreutils
        ("timeout -s 10 x", 30.0),
        ("echo timeout 99", 30.0),
        ("grep -n 'timeout 99' f", 30.0),
        ("printf 'unbalanced", 30.0),
        ("sleep 1", 30.0),
    ],
)
def test_effective_timeout_parses_coreutils_syntax(
    command: str, expected: float
) -> None:
    assert bash_tool._effective_timeout({"command": command}, command) == expected
    assert bash_tool._effective_timeout({"timeout": 7.0}, command) == 7.0


@pytest.mark.asyncio
async def test_timeout_names_the_fix_once(tmp_path: Path) -> None:
    text, _ = await _run(tmp_path, {"command": "sleep 5", "timeout": 0.2})

    hints = [line for line in text.splitlines() if line.startswith("hint: ")]
    assert hints == [
        "hint: pass a larger `timeout` (max 3600) or use run_background"
    ], text
    waited, _ = await _run(tmp_path, {"command": "sleep 45", "timeout": 0.2})
    hints = [line for line in waited.splitlines() if line.startswith("hint: ")]
    assert len(hints) == 1 and "blocked your turn" in hints[0], waited
