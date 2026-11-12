import asyncio

from zeta.accounts.registry import AccountId
from zeta.providers.factory import build_backend
from zeta.protocol.types import Message, MessageRole, TextContent
from tests.support.completion import completion_control

backend, model = build_backend(
    "claude",
    "claude-opus-4-8",
    account=AccountId("anthropic", "default"),
    require_credentials=True,
)


async def main() -> None:
    print("model:", model, "max_tokens:", backend.max_tokens,
          "thinking_budget:", backend.thinking_budget)
    events = []
    async for event in backend.complete(
        [Message(MessageRole.USER, [TextContent("Reply with the single word: ok")])],
        [],
        control=completion_control(),
    ):
        events.append(event)
    await backend.aclose()
    print("event count:", len(events))
    for event in events[-3:]:
        print("event:", event.type, repr(getattr(event, "delta", None))[:80])


asyncio.run(main())
