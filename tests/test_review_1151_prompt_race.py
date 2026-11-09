import asyncio
import threading
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEventType,
    TextContent,
)
from zeta.runtime.loop.agent import AgentLoop
from zeta.skills import SkillCatalog


def _update_prompt(assembler: ContextAssembler, prompt: Message) -> None:
    update = getattr(assembler, "update_system_prompt", None)
    if update is None:
        assembler.system_prompt = prompt
    else:
        update(prompt)


def _write_skill(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: live prompt\n---\nlive body\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_system_prompt_change_at_compaction_start_is_used(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        system_prompt="old prompt",
        skill_catalog=SkillCatalog.empty(),
    )
    assembler = loop.context_assembler
    real_prepare = assembler.prepare_context

    async def prepare_then_expose_compaction_window(**kwargs):
        await real_prepare(**kwargs)
        return True

    assembler.prepare_context = prepare_then_expose_compaction_window
    run = loop.run_turn("hello", origin=MessageOrigin.USER)
    try:
        while True:
            event = await asyncio.wait_for(anext(run), 10)
            if event.type is StreamEventType.COMPACTION_START:
                break
        _update_prompt(assembler, Message(MessageRole.SYSTEM, [TextContent("new prompt")]))
        async for _ in run:
            pass
        assert backend.calls[0][0][0].content[0].text == "new prompt"
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_system_prompt_change_during_preparation_is_retried(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        system_prompt="old prompt",
        skill_catalog=SkillCatalog.empty(),
    )
    assembler = loop.context_assembler
    started = threading.Event()
    release = threading.Event()
    calls = 0
    real_prepare = assembler._preparation_owner._prepare

    def blocked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(2)
        return real_prepare(*args, **kwargs)

    assembler._preparation_owner._prepare = blocked_prepare
    preparation = asyncio.create_task(assembler.prepare_context())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        _update_prompt(assembler, Message(MessageRole.SYSTEM, [TextContent("new prompt")]))
        release.set()
        await asyncio.wait_for(preparation, 10)
        messages = await asyncio.wait_for(assembler.assemble(), 10)
        assert messages[0].content[0].text == "new prompt"
        assert calls == 2
    finally:
        if not preparation.done():
            preparation.cancel()
            await asyncio.wait_for(preparation, 10)
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_live_reload_between_turns_updates_next_provider_prompt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    store = ConversationStore(tmp_path / "session", cwd=project)
    old_skill = home / "skills" / "old-live-prompt.md"
    _write_skill(old_skill, "old-live-prompt")
    backend = FakeBackend(
        [ScriptedTurn([TextContent("first")]), ScriptedTurn([TextContent("second")])]
    )
    loop = AgentLoop(
        backend,
        store,
        skill_catalog=SkillCatalog.empty(),
        live_config_home=home,
        live_config_project_dir=project,
    )
    try:
        await loop.refresh_live_config(force=True)
        async for _ in loop.run_turn("first", origin=MessageOrigin.USER):
            pass
        old_skill.unlink()
        _write_skill(home / "skills" / "new-live-prompt.md", "new-live-prompt")
        await loop.refresh_live_config()
        async for _ in loop.run_turn("second", origin=MessageOrigin.USER):
            pass
        prompt = "\n".join(
            block.text
            for message in backend.calls[1][0]
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextContent)
        )
        assert "new-live-prompt" in prompt
    finally:
        await loop.close()
        store.close()
