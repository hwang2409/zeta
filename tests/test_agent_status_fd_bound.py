import os
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

_CHILDREN = 300
# Each open store pins a few descriptors; holding every child at once costs
# hundreds. A bounded scan holds the caller chain plus one child at a time.
_BOUND = 40


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def _seed_children(root: ConversationStore) -> None:
    for number in range(1, _CHILDREN + 1):
        handle = f"root:{number}"
        call = ToolCall(f"call-{number}", "agent", {"prompt": handle})
        child = ConversationStore(root.session_dir / "agents", session_id=str(number))
        try:
            root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
            root.register_agent_child(
                call,
                child_session_path=str(child.session_dir),
                description=handle,
                background=True,
                child_instance_id=handle,
            )
            root.append_child_run_started(
                handle, None, "general", handle, "2026-11-08T00:00:00+00:00"
            )
            child.mark_agent_parent(call.id)
            child.start_agent_lifecycle(
                handle=handle,
                started_at="2026-11-08T00:00:00+00:00",
                depth=1,
                agent_type="general",
                description=handle,
            )
        finally:
            child.close()


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("agent_status", {"handle": "root:7"}),
        ("agent_status", {}),
        ("agent_output", {"handle": "root:7"}),
    ],
)
@pytest.mark.asyncio
async def test_child_scan_holds_bounded_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, object],
) -> None:
    from zeta.tools.agent import subtree

    root = ConversationStore(tmp_path, session_id="root")
    loop = None
    try:
        _seed_children(root)
        loop = AgentLoop(
            FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
        )
        baseline = _open_fds()
        peak = baseline
        opened = 0
        original = subtree._open_snapshot

        def measured_open(path: Path) -> ConversationStore:
            nonlocal peak, opened
            snapshot = original(path)
            opened += 1
            peak = max(peak, _open_fds())
            return snapshot

        # test-contract: seam observe descriptors at each child snapshot open
        monkeypatch.setattr(subtree, "_open_snapshot", measured_open)
        result = await loop.tool_registry.execute(ToolCall("scan", tool, arguments))

        assert result.get("isError") is not True, result
        assert peak - baseline <= _BOUND, (
            f"{tool} held {peak - baseline} extra descriptors over {opened} child opens"
        )
        if "handle" in arguments:
            # the requested child, plus the caller's own fresh notification view
            assert opened <= 2, f"{tool} for one handle opened {opened} stores"
        assert _open_fds() <= baseline
    finally:
        if loop is not None:
            await loop.close()
        root.close()
