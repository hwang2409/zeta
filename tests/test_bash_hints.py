"""Call-time hints that the bash tool appends to its result (F060, F065, F080).

Detection rules, kept simple on purpose:
- wait hint: the command text has `sleep N` (N >= 30 seconds, optional s/m/h
  suffix), or `gh pr checks ... --watch`, or `gh run watch`, and the call
  really blocked (it timed out, or ran for 30 seconds or more).
- rg hint: exit 127 and stderr says `rg: command not found`.
- grep hint: exit 1, empty stdout and stderr, and the last pipeline stage is
  grep/rg.
- gh hint: exit 8 and the command runs `gh pr checks`.
A hint is one `hint: ...` line after the normal output; nothing else changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def _run(tmp_path: Path, command: str, timeout: float = 10.0) -> tuple[str, dict]:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    try:
        registry = ToolRegistry(
            tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
        )
        try:
            result = await registry.execute(
                ToolCall("hint", "bash", {"command": command, "timeout": timeout})
            )
        finally:
            await registry.close()
    finally:
        store.close()
    return result["content"][0]["text"], result


def _hint_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("hint: ")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "sleep 45",
        "for i in 1 2; do sleep 95; done",
        "sleep 1m",
    ],
)
async def test_long_foreground_sleep_names_run_background(
    tmp_path: Path, command: str
) -> None:
    # a short tool timeout ends the call fast; the hint depends on the command text
    text, _ = await _run(tmp_path, command, timeout=0.2)

    hints = _hint_lines(text)
    assert len(hints) == 1, text
    assert "run_background" in hints[0]
    assert "blocked your turn for" in hints[0]
    assert "for 0s" not in hints[0], hints[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "gh() { sleep 5; }; gh pr checks 12 --watch --interval 60",
        "gh() { sleep 5; }; gh run watch 99",
    ],
)
async def test_foreground_ci_watch_names_run_background(
    tmp_path: Path, command: str
) -> None:
    # the stub watch waits; a short tool timeout ends the call while it blocks
    text, _ = await _run(tmp_path, command, timeout=0.2)

    hints = _hint_lines(text)
    assert len(hints) == 1, text
    assert "run_background" in hints[0]


@pytest.mark.asyncio
async def test_missing_rg_names_git_grep(tmp_path: Path) -> None:
    text, result = await _run(tmp_path, "PATH=/nonexistent; rg needle")

    assert result["structuredContent"]["exit_code"] == 127
    hints = _hint_lines(text)
    assert len(hints) == 1, text
    assert "git grep -n" in hints[0]
    assert "grep -rn" in hints[0]


@pytest.mark.asyncio
async def test_grep_no_match_says_no_match(tmp_path: Path) -> None:
    text, result = await _run(tmp_path, "printf 'a\\n' | grep zzz")

    assert result["structuredContent"]["exit_code"] == 1
    assert result["isError"] is True  # error semantics unchanged
    assert _hint_lines(text) == ["hint: exit 1 from grep means no match"]


@pytest.mark.asyncio
async def test_gh_pr_checks_exit_8_says_pending(tmp_path: Path) -> None:
    text, result = await _run(tmp_path, "gh() { return 8; }; gh pr checks 12")

    assert result["structuredContent"]["exit_code"] == 8
    assert _hint_lines(text) == [
        "hint: exit 8 from gh pr checks means checks are pending"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "printf hello",
        "sleep 0.1",
        "printf 'a\\n' | grep a",
        "false",
        "grep zzz /nonexistent-file",
        "gh() { return 8; }; gh run list",
        "cd /nonexistent && grep x f",
        "(sleep 45; echo hi) >/dev/null 2>&1 & echo bg",
        "gh() { :; }; gh run watch 1",
    ],
)
async def test_ordinary_results_have_no_hint(tmp_path: Path, command: str) -> None:
    text, _ = await _run(tmp_path, command)

    assert _hint_lines(text) == [], text


@pytest.mark.asyncio
async def test_short_sleep_under_threshold_has_no_hint(tmp_path: Path) -> None:
    text, _ = await _run(tmp_path, "sleep 29", timeout=0.2)

    # no wait hint; the plain timeout hint (F066) is the only line
    assert _hint_lines(text) == [
        "hint: pass a larger `timeout` (max 3600) or use run_background"
    ], text


@pytest.mark.asyncio
async def test_hint_leaves_output_bytes_unchanged(tmp_path: Path) -> None:
    text, _ = await _run(tmp_path, "printf 'x\\n'; printf 'a\\n' | grep zzz")

    assert text == "stdout:\nx\n\nstderr:\n"  # stdout not empty: no hint
    plain, _ = await _run(tmp_path, "printf hello")
    assert plain == "stdout:\nhello\nstderr:\n"
    hinted, hinted_result = await _run(tmp_path, "printf '' | grep zzz")
    assert hinted == "stdout:\n\nstderr:\n\nhint: exit 1 from grep means no match"
    assert hinted_result["structuredContent"]["stdout"] == ""
