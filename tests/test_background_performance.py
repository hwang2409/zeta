"""Structural performance checks for background agent ownership."""

from __future__ import annotations

import asyncio

import pytest

from tests.support.owned import await_or_fail
from zeta.agent.background_lifecycle import _recover_nested_children
from zeta.agent.background_owner import BackgroundAgentOwner, CancellationRequest
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall, ToolResult
from zeta.storage_format import WRITER_EPOCH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent_ids", "expected_order"),
    [
        (
            {"root": None, "one": None, "two": None, "three": None},
            ["root", "one", "two", "three"],
        ),
        (
            {"root": None, "one": "root", "two": "one", "three": "two"},
            ["root", "one", "two", "three"],
        ),
    ],
    ids=("flat", "chain"),
)
async def test_cancel_all_visits_each_child_once_without_subtree_walks(
    parent_ids: dict[str, str | None], expected_order: list[str]
) -> None:
    owner = BackgroundAgentOwner.__new__(BackgroundAgentOwner)
    owner._canceling = False
    owner._cancel_requested = set()
    owner._cancel_confirmations = {}
    owner._parent_ids = parent_ids
    events: list[tuple[str, str]] = []
    confirmations: dict[str, asyncio.Task[bool]] = {}

    def cancel(instance_id: str) -> CancellationRequest:
        events.append(("cancel_requested", instance_id))
        confirmation = asyncio.create_task(asyncio.sleep(0, result=True))
        confirmations[instance_id] = confirmation
        return CancellationRequest(confirmation, confirmation)

    owner._cancellers = {
        instance_id: (lambda instance_id=instance_id: cancel(instance_id))
        for instance_id in parent_ids
    }
    subtree_walks: list[str] = []
    original_subtree_ids = owner._subtree_ids

    def count_subtree_walk(instance_id: str) -> set[str]:
        subtree_walks.append(instance_id)
        return original_subtree_ids(instance_id)

    owner._subtree_ids = count_subtree_walk
    try:
        owner.cancel_all()
        await await_or_fail(
            asyncio.gather(*confirmations.values()),
            what="background cancellation confirmations",
        )

        assert subtree_walks == []
        assert owner._cancel_requested == set(expected_order)
        assert list(confirmations) == expected_order
        assert events == [
            ("cancel_requested", instance_id) for instance_id in expected_order
        ]
        assert owner._cancel_confirmations == confirmations

        owner.cancel_all()
        assert events == [
            ("cancel_requested", instance_id) for instance_id in expected_order
        ]
    finally:
        for confirmation in confirmations.values():
            confirmation.cancel()
        await await_or_fail(
            asyncio.gather(*confirmations.values(), return_exceptions=True),
            what="background cancellation confirmation cleanup",
        )


def test_nested_recovery_uses_indexed_tool_result_lookup() -> None:
    tool_calls = [ToolCall(f"call-{index}", "child", {}) for index in range(3)]

    class Store:
        session_id = "parent"

        def __init__(self) -> None:
            self.messages_calls = 0
            self.tool_result_calls = 0
            self.appended: list[Message] = []
            self.finished: list[str] = []
            self.results = {
                call.id: ToolResult(call.id, f"done-{index}")
                for index, call in enumerate(tool_calls)
            }

        def agent_children(self) -> dict[str, dict[str, object]]:
            return {
                f"marker-{index}": {
                    "tool_call": call.to_dict(),
                    "child_session_path": f"child-{index}",
                    "description": f"child {index}",
                }
                for index, call in enumerate(tool_calls)
            }

        def messages(self) -> list[Message]:
            self.messages_calls += 1
            return [
                Message(
                    MessageRole.TOOL_RESULT,
                    [TextContent(result.content)],
                    tool_result=result,
                )
                for result in self.results.values()
            ]

        def tool_result(self, tool_call_id: str) -> ToolResult | None:
            self.tool_result_calls += 1
            return self.results.get(tool_call_id)

        def append_message(self, message: Message) -> None:
            self.appended.append(message)

        def finish_agent_child(self, marker_key: str) -> None:
            self.finished.append(marker_key)

    store = Store()
    _recover_nested_children(store, store, store, {}, WRITER_EPOCH)  # type: ignore[arg-type]

    assert store.messages_calls == 0
    assert store.tool_result_calls == len(tool_calls)
    assert store.appended == []
    assert store.finished == [f"marker-{index}" for index in range(3)]
