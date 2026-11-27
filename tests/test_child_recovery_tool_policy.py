from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.agent import runner
from zeta.agent.presets import GENERAL_PRESET
from zeta.agent.recovery_lease import activate_agent_children
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spawn_tools", "required_tools", "removed_tool", "block_reason"),
    [
        pytest.param((), None, None, None, id="same-allowlist"),
        pytest.param(("ask_parent",), None, "ask_parent", None, id="optional-ask-parent-removed"),
        pytest.param(("ask_parent",), frozenset({"read"}), None, None, id="profile-with-optional-runtime-tool"),
        pytest.param(
            (), frozenset({"read"}), "read",
            "current tool policy cannot satisfy the saved profile's required tools",
            id="required-profile-tool-removed",
        ),
        pytest.param(
            ("finish", "agent_handoff"), None, "finish",
            "current policy blocks required explicit-finish tools: finish",
            id="required-finish-removed",
        ),
    ],
)
async def test_child_recovery_respects_spawn_tool_policy(
    tmp_path, monkeypatch, spawn_tools, required_tools, removed_tool, block_reason,
):
    allowed = ("agent", "read", *spawn_tools)
    explicit_finish = "finish" in spawn_tools
    preset = replace(GENERAL_PRESET, tool_names=required_tools, explicit_finish=explicit_finish)
    catalog = AgentCatalog((preset,))
    running = asyncio.Event()
    release = asyncio.Event()
    seen_ask_parent = []

    def build_loop(store, tools):
        backend = FakeBackend([])
        backend.model = "gpt-5.6-sol"
        backend.provider = "codex"
        skills = SkillCatalog.empty()
        registry = ToolRegistry(cwd=tmp_path, skill_catalog=skills, tool_allow=tools, agent_catalog=catalog)
        loop = AgentLoop(backend, store, registry=registry, skill_catalog=skills)
        loop.explicit_finish = explicit_finish
        return loop

    async def consume(child, prompt, **kwargs):
        seen_ask_parent.append(child.tool_registry.tool_is_allowed("ask_parent"))
        running.set()
        await await_or_fail(release.wait(), what="child held open for restart")

    # test-contract: use an offline backend and hold the running child at dispatch.
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda loop, model, **kw: (loop.backend, None))
    monkeypatch.setattr(runner, "consume_child", consume)
    monkeypatch.setattr(runner, "consume_run", consume)
    root = tmp_path / "sessions"
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        loop = build_loop(store, allowed)
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn child")
            await await_or_fail(running.wait(), what="original child ran")
            assert seen_ask_parent == ["ask_parent" in allowed]
            marker = store.agent_children()[run_id]
            assert marker["resume"]["explicit_finish"] is explicit_finish
            path = Path(marker["child_session_path"])
        finally:
            await await_or_fail(loop.close(), what="suspend child")
    running.clear()

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = build_loop(store, tuple(name for name in allowed if name != removed_tool))
        try:
            await await_or_fail(activate_agent_children(loop), what="recover suspended child")
            lifecycle, _ = read_child_lifecycle(store, path)
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            if block_reason is not None:
                assert lifecycle.value["state"] == "suspended"
                assert lifecycle.value["current_step"] == f"recovery blocked: {block_reason}"
                assert not running.is_set()
            else:
                assert lifecycle.value["state"] == "running", lifecycle.value.get("recovery_blocked")
                await await_or_fail(running.wait(), what="recovered child ran")
                assert seen_ask_parent == ["ask_parent" in allowed, "ask_parent" in allowed and removed_tool != "ask_parent"]
        finally:
            await await_or_fail(loop.close(), what="close resumed runtime")
            release.set()
