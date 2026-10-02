import json
from pathlib import Path

import pytest

from zeta.context_strategies import context_strategies, recall_history
from zeta.context_strategies.archive import archive_context, restore_context
from zeta.context_strategies.edit import replace_context
from zeta.context_strategies.evict import evict_messages
from zeta.context_strategies.fold import fold_messages
from zeta.core.context import CompactionPolicy, ContextAssembler
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def tool_pair(name: str, call_id: str, output: str, *, error: bool = False):
    call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall(call_id, name, {"path": "src/foo.py"}))],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult(call_id, output, is_error=error),
    )
    return call, result


def assert_payload_pairing(messages: list[Message]) -> None:
    anthropic = build_messages_payload(
        messages, [], model="claude-test", max_tokens=2048, thinking_budget=1024
    )
    anthropic_calls = {
        block["id"]
        for message in anthropic["messages"]
        for block in message["content"]
        if block["type"] == "tool_use"
    }
    anthropic_results = {
        block["tool_use_id"]
        for message in anthropic["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    }
    assert anthropic_calls == anthropic_results

    codex = build_responses_payload(messages, [], model="gpt-test")["input"]
    codex_calls = {
        item["call_id"] for item in codex if item.get("type") == "function_call"
    }
    codex_results = {
        item["call_id"] for item in codex if item.get("type") == "function_call_output"
    }
    assert codex_calls == codex_results

    ollama = build_ollama_messages(messages)
    ollama_calls = {
        call["function"]["name"]
        for message in ollama
        for call in message.get("tool_calls", [])
    }
    ollama_results = {
        message["tool_name"] for message in ollama if message["role"] == "tool"
    }
    assert ollama_calls == ollama_results


def test_strategy_flags_are_composable() -> None:
    assert context_strategies("fold,evict,recall,budget,unknown") == {
        "fold",
        "evict",
        "recall",
        "budget",
    }


def test_fold_is_deterministic_and_protects_non_rederivable_and_failures() -> None:
    read_call, read_result = tool_pair("read", "read-1", "x" * 4000)
    edit_call, edit_result = tool_pair("edit", "edit-1", "changed" * 500)
    bash_call, bash_failure = tool_pair("bash", "bash-1", "fatal" * 500, error=True)
    messages = [
        (10, text(MessageRole.USER, "keep user")),
        (11, read_call),
        (12, read_result),
        (13, text(MessageRole.ASSISTANT, "keep assistant")),
        (14, edit_call),
        (15, edit_result),
        (16, bash_call),
        (17, bash_failure),
    ]

    first = fold_messages(messages)
    second = fold_messages(messages)

    assert [message.to_dict() for message in first.messages] == [
        message.to_dict() for message in second.messages
    ]
    assert first.items_folded == 1
    stub = first.messages[2].tool_result
    assert stub is not None
    assert "[folded read src/foo.py" in stub.content
    assert "seq 12" in stub.content
    assert "recall_history seq_start=12" in stub.content
    assert first.messages[0] == messages[0][1]
    assert first.messages[3] == messages[3][1]
    assert first.messages[4:] == [message for _, message in messages[4:]]
    assert_payload_pairing(first.messages)


def test_evict_is_deterministic_and_protects_user_mutations_and_failures() -> None:
    read_call, read_result = tool_pair("read", "read-1", "x" * 8000)
    edit_call, edit_result = tool_pair("edit", "edit-1", "changed")
    bash_call, bash_failure = tool_pair("bash", "bash-1", "fatal" * 500, error=True)
    records = [
        (20, text(MessageRole.USER, "never remove this")),
        (21, read_call),
        (22, read_result),
        (23, edit_call),
        (24, edit_result),
        (25, bash_call),
        (26, bash_failure),
    ]

    first = evict_messages(records, fixed_tokens=0, target_tokens=900)
    second = evict_messages(records, fixed_tokens=0, target_tokens=900)

    assert first.reached_target
    assert [message.to_dict() for message in first.messages] == [
        message.to_dict() for message in second.messages
    ]
    assert first.messages[0] == records[0][1]
    assert first.messages[3:] == [message for _, message in records[3:]]
    assert_payload_pairing(first.messages)


def test_evict_keeps_parallel_tool_batches_valid_for_all_payloads() -> None:
    calls = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(ToolCall("read-a", "read", {"path": "a"})),
            ToolUseContent(ToolCall("read-b", "read", {"path": "b"})),
        ],
        metadata={
            "codex_output_items": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "read-a",
                    "name": "read",
                    "arguments": '{"path":"a"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "read-b",
                    "name": "read",
                    "arguments": '{"path":"b"}',
                    "status": "completed",
                },
            ]
        },
    )
    records = [
        (1, calls),
        (
            2,
            Message(
                MessageRole.TOOL_RESULT, tool_result=ToolResult("read-a", "a" * 4000)
            ),
        ),
        (
            3,
            Message(
                MessageRole.TOOL_RESULT, tool_result=ToolResult("read-b", "b" * 4000)
            ),
        ),
    ]

    evicted = evict_messages(records, fixed_tokens=0, target_tokens=170)

    assert evicted.reached_target
    assert_payload_pairing(evicted.messages)


class RecordingPolicy(CompactionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[list[Message]] = []

    async def summarize_chunked(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.inputs.append(list(messages))
        return "summary"


@pytest.mark.asyncio
async def test_fold_changes_only_summary_input_and_emits_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "fold,recall,budget")
    monkeypatch.setenv("ZETA_CONTEXT_TELEMETRY", str(telemetry))
    store = ConversationStore(tmp_path / "session")
    store.append_message(text(MessageRole.USER, "inspect"))
    call, result = tool_pair("read", "read-1", "result " * 1000)
    call_entry = store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(text(MessageRole.USER, "tail"))
    policy = RecordingPolicy()

    await ContextAssembler(
        store,
        token_budget=300,
        retained_tail=1,
        compaction_policy=policy,
    ).assemble()

    summary_input = policy.inputs[0]
    folded = next(
        message.tool_result for message in summary_input if message.tool_result
    )
    assert "[folded read src/foo.py" in folded.content
    assert f"seq {result_entry.seq}" in folded.content
    assert Message.from_dict(call_entry.data["message"]) == call
    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    event = next(
        row
        for row in events
        if row["event"] == "context_strategy" and row["kind"] == "fold"
    )
    assert event["range"] == [1, result_entry.seq]
    assert event["items_folded"] == 1
    assert event["tokens_after"] < event["tokens_before"]


@pytest.mark.asyncio
async def test_evict_persists_replays_recalls_and_keeps_payload_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict,recall,budget")
    monkeypatch.setenv("ZETA_CONTEXT_TELEMETRY", str(telemetry))
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="session")
    old_user = store.append_message(text(MessageRole.USER, "protected early fact"))
    call, result = tool_pair("read", "read-1", "large output " * 1000)
    store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [
                ThinkingContent("private reasoning " * 300),
                TextContent("old answer " * 300),
            ],
        )
    )
    store.append_message(text(MessageRole.USER, "retained tail"))
    policy = RecordingPolicy()
    assembler = ContextAssembler(
        store,
        token_budget=600,
        retained_tail=1,
        compaction_policy=policy,
    )

    first = await assembler.assemble()

    assert policy.inputs == []
    assert any(
        "seq 3" in block.text
        for message in first
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert any(
        "protected early fact" in block.text
        for message in first
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert_payload_pairing(first[:-1])  # budget readout is deliberately synthetic
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["kind"] == "eviction"

    reopened = ConversationStore(sessions, session_id="session")
    replayed = await ContextAssembler(
        reopened,
        token_budget=600,
        retained_tail=1,
        compaction_policy=RecordingPolicy(),
    ).assemble()
    assert [message.to_dict() for message in replayed] == [
        message.to_dict() for message in first
    ]

    first_marker = next(
        entry for entry in reopened.replay() if entry.type == "compaction"
    )
    reopened.append_message(text(MessageRole.ASSISTANT, "later answer " * 1000))
    reopened.append_message(text(MessageRole.USER, "new retained tail"))
    second_policy = RecordingPolicy()
    await ContextAssembler(
        reopened,
        token_budget=600,
        retained_tail=1,
        compaction_policy=second_policy,
    ).assemble()
    assert second_policy.inputs == []
    latest_marker = [
        entry for entry in reopened.replay() if entry.type == "compaction"
    ][-1]
    assert first_marker.id in latest_marker.data["replaces"]
    assert latest_marker.data["source_seq_start"] == old_user.seq

    registry = ToolRegistry(
        tmp_path, session_store=reopened, skill_catalog=SkillCatalog.empty()
    )
    recalled = await registry.execute(
        ToolCall(
            "recall-1",
            "recall_history",
            {"seq_start": result_entry.seq, "seq_end": result_entry.seq},
        ),
        _skip_approval=True,
    )
    assert recalled["content"][0]["text"].find("large output") >= 0

    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    event = next(row for row in events if row.get("kind") == "evict")
    assert event["range"] == [old_user.seq, 5]
    assert event["items_evicted"] > 0
    assert event["tokens_after"] < event["tokens_before"]


@pytest.mark.asyncio
async def test_evict_falls_back_to_normal_summarization_when_nothing_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "protected " * 1000))
    store.append_message(text(MessageRole.USER, "tail"))
    policy = RecordingPolicy()

    assembled = await ContextAssembler(
        store,
        token_budget=200,
        retained_tail=1,
        compaction_policy=policy,
    ).assemble()

    assert len(policy.inputs) == 1
    assert policy.inputs[0][0].role is MessageRole.USER
    assert "protected" in policy.inputs[0][0].content[0].text
    assert any(message.role is MessageRole.COMPACTION for message in assembled)
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data.get("kind", "summary") == "summary"


@pytest.mark.asyncio
async def test_all_seven_strategies_compose_and_replay_for_every_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "ZETA_CONTEXT_STRATEGY",
        "recall,budget,archive,edit,nudge,fold,evict",
    )
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="all-seven")
    archived = store.append_message(text(MessageRole.USER, "archive this old turn"))
    edited = store.append_message(text(MessageRole.ASSISTANT, "replace this old answer"))
    call, result = tool_pair("read", "all-seven-read", "large output " * 1200)
    store.append_message(call)
    store.append_message(result)
    store.append_message(text(MessageRole.ASSISTANT, "old prose " * 500))
    store.append_message(text(MessageRole.USER, "retained tail"))

    archive_context(
        store,
        seq_start=archived.seq,
        seq_end=archived.seq,
        retained_tail=1,
    )
    replace_context(
        store,
        seq_start=edited.seq,
        seq_end=edited.seq,
        replacement="bounded replacement",
    )

    first = await ContextAssembler(
        store,
        token_budget=500,
        retained_tail=1,
        compaction_policy=RecordingPolicy(),
    ).assemble()

    assert_payload_pairing(first)
    assert sum(bool(message.metadata.get("context_budget_readout")) for message in first) == 1
    tail = next(message for message in first if message.metadata.get("context_budget_readout"))
    assert tail.metadata.get("context_nudge")
    assert "[context budget]" in tail.content[0].text
    assert "[context nudge]" in tail.content[0].text
    assert any(message.metadata.get("context_archive") for message in first)
    assert any(message.metadata.get("context_model_note") for message in first)
    assert any(
        entry.type == "compaction" and entry.data.get("kind") == "eviction"
        for entry in store.replay()
    )

    reopened = ConversationStore(sessions, session_id="all-seven")
    replayed = await ContextAssembler(
        reopened,
        token_budget=500,
        retained_tail=1,
        compaction_policy=RecordingPolicy(),
    ).assemble()
    assert [message.to_dict() for message in replayed] == [
        message.to_dict() for message in first
    ]
    assert_payload_pairing(replayed)


async def _persist_eviction(
    store: ConversationStore,
    *,
    token_budget: int = 500,
) -> tuple[int, int]:
    call, result = tool_pair("read", "overlap-read", "exact overlap output " * 1000)
    call_entry = store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(text(MessageRole.ASSISTANT, "disposable " * 500))
    store.append_message(text(MessageRole.USER, "tail"))
    await ContextAssembler(
        store,
        token_budget=token_budget,
        retained_tail=1,
        compaction_policy=RecordingPolicy(),
    ).assemble()
    return call_entry.seq, result_entry.seq


@pytest.mark.asyncio
async def test_later_archive_overlapping_eviction_wins_and_restore_only_unarchives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict,archive,recall")
    store = ConversationStore(tmp_path)
    call_seq, result_seq = await _persist_eviction(store)

    archived = archive_context(
        store,
        seq_start=result_seq,
        seq_end=result_seq,
        retained_tail=1,
    )
    assert (archived.seq_start, archived.seq_end) == (call_seq, result_seq)
    hidden = await ContextAssembler(store, retained_tail=1).assemble()
    assert_payload_pairing(hidden)
    assert any(message.metadata.get("context_archive") for message in hidden)
    assert not any(message.tool_result for message in hidden)

    restore_context(store, archive_id=archived.archive_id)
    restored = await ContextAssembler(store, retained_tail=1).assemble()
    assert_payload_pairing(restored)
    assert not any(message.metadata.get("context_archive") for message in restored)
    assert any("evicted" in message.content[0].text for message in restored if message.content)


@pytest.mark.asyncio
async def test_later_edit_overlapping_eviction_replaces_the_whole_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict,edit,recall")
    store = ConversationStore(tmp_path)
    call_seq, result_seq = await _persist_eviction(store)

    replaced = replace_context(
        store,
        seq_start=result_seq,
        seq_end=result_seq,
        replacement="authoritative compact note",
    )
    assert (replaced.seq_start, replaced.seq_end) == (call_seq, result_seq)
    assembled = await ContextAssembler(store, retained_tail=1).assemble()
    assert_payload_pairing(assembled)
    notes = [message for message in assembled if message.metadata.get("context_model_note")]
    assert len(notes) == 1
    assert "authoritative compact note" in notes[0].content[0].text
    assert not any(message.tool_result for message in assembled)


@pytest.mark.asyncio
async def test_fold_stub_points_to_exact_recall_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "fold,recall")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "inspect"))
    call, result = tool_pair("read", "fold-recall", "exact folded source " * 800)
    store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(text(MessageRole.USER, "tail"))
    policy = RecordingPolicy()

    await ContextAssembler(
        store,
        token_budget=300,
        retained_tail=1,
        compaction_policy=policy,
    ).assemble()
    folded_result = next(
        message.tool_result for message in policy.inputs[0] if message.tool_result
    )
    assert f"recall_history seq_start={result_entry.seq}" in folded_result.content


    recalled, mode = recall_history(
        store,
        seq_start=result_entry.seq,
        seq_end=result_entry.seq,
    )
    assert mode == "range"
    assert "exact folded source" in recalled


@pytest.mark.asyncio
async def test_evict_can_reduce_oversized_paired_retained_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "inspect all logs"))
    calls = [ToolCall(f"read-{index}", "read", {"path": f"log-{index}"}) for index in range(5)]
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(call) for call in calls],
        )
    )
    for call in calls:
        store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult(call.id, "large log output " * 1000),
            )
        )

    assembled = await ContextAssembler(
        store,
        token_budget=1_000,
        retained_tail=8,
        compaction_policy=RecordingPolicy(),
    ).assemble()

    assert_payload_pairing(assembled)
    assert any("inspect all logs" in message.content[0].text for message in assembled)
    assert any(
        entry.type == "compaction" and entry.data.get("kind") == "eviction"
        for entry in store.replay()
    )
