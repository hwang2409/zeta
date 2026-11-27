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


ALLOWLIST = ("agent", "agent_cancel", "agent_status", "read", "bash", "ask_parent", "agent_handoff", "finish")
FINISH_BLOCK = "current policy blocks required explicit-finish tools: finish"
PROFILE_BLOCK = "current tool policy cannot satisfy the saved profile's required tools"


def without(*names):
    return tuple(name for name in ALLOWLIST if name not in names)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("explicit_finish", "spawn_tools", "resume_tools", "required_tools", "block_reason"),
    [
        pytest.param(
            explicit_finish, spawn_tools, resume_tools, None,
            FINISH_BLOCK if explicit_finish and case == "removed-finish" else None,
            id=f"{case}-{'explicit' if explicit_finish else 'ordinary'}",
        )
        for explicit_finish in (True, False)
        for case, spawn_tools, resume_tools in (
            ("unrestricted", None, None),
            ("no-ask-parent", without("ask_parent"), without("ask_parent")),
            ("no-handoff", without("agent_handoff"), without("agent_handoff")),
            ("no-ask-parent-or-handoff", without("ask_parent", "agent_handoff"), without("ask_parent", "agent_handoff")),
            ("removed-finish", ALLOWLIST, without("finish")),
            ("removed-bash", ALLOWLIST, without("bash")),
        )
    ] + [
        pytest.param(False, ALLOWLIST, ALLOWLIST, frozenset({"read"}), None, id="profile-with-optional-runtime-tool"),
        pytest.param(False, ALLOWLIST, without("read"), frozenset({"read"}), PROFILE_BLOCK, id="required-profile-tool-removed"),
    ],
)
async def test_child_recovery_respects_spawn_tool_policy(
    tmp_path, monkeypatch, explicit_finish, spawn_tools, resume_tools, required_tools, block_reason,
):
    preset = replace(GENERAL_PRESET, tool_names=required_tools, explicit_finish=explicit_finish)
    catalog = AgentCatalog((preset,))
    running = asyncio.Event()
    release = asyncio.Event()
    seen_tools = []

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
        seen_tools.append(child.tool_registry.registered_names)
        running.set()
        await await_or_fail(release.wait(), what="child held open for restart")

    async def spawn(loop):
        running.clear()
        run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn child")
        await await_or_fail(running.wait(), what="spawned child ran")
        return run_id

    # test-contract: use an offline backend and hold the running child at dispatch.
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda loop, model, **kw: (loop.backend, None))
    monkeypatch.setattr(runner, "consume_child", consume)
    monkeypatch.setattr(runner, "consume_run", consume)
    root = tmp_path / "sessions"
    try:
        with ConversationStore(root, cwd=tmp_path) as store:
            session_id = store.session_id
            loop = build_loop(store, spawn_tools)
            try:
                run_id = await spawn(loop)
                assert ("finish" in seen_tools[-1]) is explicit_finish
                marker = store.agent_children()[run_id]
                assert marker["resume"]["explicit_finish"] is explicit_finish
                path = Path(marker["child_session_path"])
            finally:
                await await_or_fail(loop.close(), what="suspend child")
            lifecycle, _ = read_child_lifecycle(store, path)
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            assert lifecycle.value["state"] == "suspended"

        if block_reason != PROFILE_BLOCK:
            with ConversationStore(root, cwd=tmp_path) as store:
                loop = build_loop(store, resume_tools)
                try:
                    await spawn(loop)
                    fresh_tools = seen_tools[-1]
                    if resume_tools is not None:
                        assert fresh_tools <= set(resume_tools)
                finally:
                    await await_or_fail(loop.close(), what="close fresh runtime")

        running.clear()
        with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
            loop = build_loop(store, resume_tools)
            try:
                await await_or_fail(activate_agent_children(loop), what="recover suspended child")
                lifecycle, _ = read_child_lifecycle(store, path)
                assert isinstance(lifecycle, LoadedAgentLifecycle)
                if block_reason is not None:
                    assert lifecycle.value["state"] == "suspended"
                    assert lifecycle.value["current_step"] == f"recovery blocked: {block_reason}"
                    assert not running.is_set()
                else:
                    assert lifecycle.value["state"] == "running", lifecycle.value["current_step"]
                    await await_or_fail(running.wait(), what="recovered child ran")
                    assert seen_tools[-1] == fresh_tools
            finally:
                await await_or_fail(loop.close(), what="close resumed runtime")
    finally:
        release.set()
