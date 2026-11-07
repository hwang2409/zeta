from __future__ import annotations

import pytest

from tests.support.fake_backend import FakeBackend, PerActorFakeBackend, ScriptedTurn
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
)


async def _complete(backend: CompletionBackend, *prompts: str):
    messages = [Message(MessageRole.USER, [TextContent(prompt)]) for prompt in prompts]
    return [event async for event in backend.complete(messages, [])]


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
        await _complete(backend, "parent work", "child work")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actors", "error"),
    [
        (("parent work", "child work"), None),
        (("child work", "parent work"), None),
        (("unknown work",), "no scripted backend for actor prompt.*unknown work"),
    ],
)
async def test_per_actor_fake_backend_routes_by_actor_in_any_order(
    actors: tuple[str, ...], error: str | None
) -> None:
    parent_call = ToolCall(
        "child",
        "agent",
        {
            "prompt": "child work",
            "description": "child",
            "background": True,
        },
    )
    delegated_call = ToolCall("delegated", "read", {"path": "input"})
    backend = PerActorFakeBackend(
        [ScriptedTurn(tool_calls=[parent_call])],
        {"child work": [ScriptedTurn(tool_calls=[delegated_call])]},
        parent_prompt="parent work",
        gated_child_call=0,
    )

    if error is not None:
        with pytest.raises(AssertionError, match=error):
            await _complete(backend, *actors)
        return

    events = [event for actor in actors for event in await _complete(backend, actor)]
    delegated = [
        block.tool_call
        for event in events
        if isinstance(block := event.content, ToolUseContent)
        and block.tool_call.name == "read"
    ]
    assert delegated == [delegated_call]
