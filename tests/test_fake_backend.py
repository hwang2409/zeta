from __future__ import annotations

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall


async def _complete(backend: FakeBackend, prompt: str) -> None:
    messages = [Message(MessageRole.USER, [TextContent(prompt)])]
    _ = [event async for event in backend.complete(messages, [])]


@pytest.mark.asyncio
async def test_fake_backend_rejects_shared_background_actor_script() -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "child",
                        "agent",
                        {
                            "prompt": "child work",
                            "description": "child",
                            "background": True,
                        },
                    )
                ]
            ),
            ScriptedTurn([TextContent("child done")]),
        ]
    )

    await _complete(backend, "parent work")
    with pytest.raises(
        AssertionError,
        match="FakeBackend cannot serve multiple actors; use PerActorFakeBackend",
    ):
        await _complete(backend, "child work")
