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
