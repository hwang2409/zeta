import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from zeta.context_eviction import EvictionResult, evict_messages, recall_history
from zeta.core.context import CompactionPolicy, ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def tool_pair(
    name: str,
    call_id: str,
    output: str,
    *,
    arguments: dict[str, object] | None = None,
    error: bool = False,
) -> tuple[Message, Message]:
    call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall(call_id, name, arguments or {"path": "RULES.md"}))],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult(call_id, output, is_error=error),
    )
    return call, result


def rendered_text(messages: list[Message]) -> str:
    values: list[str] = []
    for message in messages:
        values.extend(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        if message.tool_result is not None:
            values.append(message.tool_result.content)
    return "\n".join(values)


def assert_payload_pairing(messages: list[Message]) -> None:
    anthropic = build_messages_payload(
        messages, [], model="claude-test", max_tokens=2048, thinking_budget=1024
    )["messages"]
    assert {
        block["id"]
        for message in anthropic
        for block in message["content"]
        if block["type"] == "tool_use"
    } == {
        block["tool_use_id"]
        for message in anthropic
        for block in message["content"]
        if block["type"] == "tool_result"
    }

    codex = build_responses_payload(messages, [], model="gpt-test")["input"]
    assert {
        item["call_id"] for item in codex if item.get("type") == "function_call"
    } == {
        item["call_id"]
        for item in codex
        if item.get("type") == "function_call_output"
    }

    ollama = build_ollama_messages(messages)
    assert {
        call["function"]["name"]
        for message in ollama
        for call in message.get("tool_calls", [])
    } == {
        message["tool_name"] for message in ollama if message["role"] == "tool"
    }


def test_digest_is_deterministic_bounded_and_retains_load_bearing_lines() -> None:
    output = "\n".join(
        [
            "# Maintainers",
            "first useful line",
            *(f"unimportant filler {index}" for index in range(80)),
            "MUST run the complete validation suite",
            "Do not force-push this branch",
            "last useful line",
        ]
    )
    call, result = tool_pair("read", "read-1", output)
    records = [(10, call), (11, result)]

    first = evict_messages(records, fixed_tokens=0, target_tokens=1)
    second = evict_messages(records, fixed_tokens=0, target_tokens=1)

    assert [message.to_dict() for message in first.messages] == [
        message.to_dict() for message in second.messages
    ]
    digest = first.messages[1].tool_result
    assert digest is not None
    assert "RULES.md" in digest.content
    assert "85 lines" in digest.content
    assert "# Maintainers" in digest.content
    assert "MUST run the complete validation suite" in digest.content
    assert "Do not force-push this branch" in digest.content
    assert "last useful line" in digest.content
    assert "recall_history seq_start=11" in digest.content
    assert len(digest.content) <= 440


def test_repeated_reads_dedupe_oldest_first_and_preserve_pairing() -> None:
    first_call, first_result = tool_pair("read", "read-1", "same output\n" * 300)
    second_call, second_result = tool_pair("read", "read-2", "same output\n" * 300)
    result = evict_messages(
        [(1, first_call), (2, first_result), (3, second_call), (4, second_result)],
        fixed_tokens=0,
        target_tokens=250,
    )

    output = rendered_text(result.messages)
    assert output.count("RULES.md") == 1
    assert "read 2 times" in output
    assert "older duplicate read collapsed into seq 4" in output
    assert "duplicate result collapsed into seq 4" in output
    assert_payload_pairing(result.messages)


def test_repeated_reads_with_changed_content_keep_distinct_digests() -> None:
    old_call, old_result = tool_pair(
        "read", "read-old", "OLD UNIQUE FACT\n" * 300
    )
    new_call, new_result = tool_pair(
        "read", "read-new", "NEW DIFFERENT FACT\n" * 300
    )

    result = evict_messages(
        [(1, old_call), (2, old_result), (3, new_call), (4, new_result)],
        fixed_tokens=0,
        target_tokens=1,
    )

    output = rendered_text(result.messages)
    assert "collapsed into" not in output
    assert "semantic read digest · seq 2" in output
    assert "recall_history seq_start=2, seq_end=2" in output
    assert "semantic read digest · seq 4" in output
    assert "recall_history seq_start=4, seq_end=4" in output


def test_every_eviction_stub_points_to_matching_original_content() -> None:
    first_call, first_result = tool_pair("read", "read-1", "SAME FACT\n" * 300)
    second_call, second_result = tool_pair("read", "read-2", "SAME FACT\n" * 300)
    changed_call, changed_result = tool_pair(
        "read", "read-3", "CHANGED FACT\n" * 300
    )
    records = [
        (1, first_call),
        (2, first_result),
        (3, second_call),
        (4, second_result),
        (5, changed_call),
        (6, changed_result),
    ]

    result = evict_messages(records, fixed_tokens=0, target_tokens=1)
    original_results = {
        seq: message.tool_result.content
        for seq, message in records
        if message.tool_result is not None
    }

    for (source_seq, original), stub in zip(records, result.messages, strict=True):
        rendered = rendered_text([stub])
        pointer = re.search(r"(?:seq_start=|collapsed into seq )(\d+)", rendered)
        if pointer is None:
            continue
        target_seq = int(pointer.group(1))
        if original.tool_result is not None:
            assert original_results[target_seq] == original.tool_result.content, source_seq


def test_parallel_results_preserve_provider_pairing() -> None:
    calls = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(ToolCall("read-a", "read", {"path": "a.md"})),
            ToolUseContent(ToolCall("read-b", "read", {"path": "b.md"})),
        ],
    )
    result = evict_messages(
        [
            (1, calls),
            (2, Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("read-a", "a" * 4000))),
            (3, Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("read-b", "b" * 4000))),
        ],
        fixed_tokens=0,
        target_tokens=100,
    )
    assert_payload_pairing(result.messages)


@pytest.mark.asyncio
async def test_explicit_summary_keeps_previous_default_request_bytes(
    tmp_path: Path,
) -> None:
    default_store = ConversationStore(tmp_path / "default")
    explicit_store = ConversationStore(tmp_path / "explicit")
    for store in (default_store, explicit_store):
        store.append_message(text(MessageRole.USER, "hello"))
        store.append_message(text(MessageRole.ASSISTANT, "world"))

    default = await ContextAssembler(default_store, system_prompt="system").assemble_context()
    explicit = await ContextAssembler(
        explicit_store, system_prompt="system", compaction="summary"
    ).assemble_context()

    assert default.digest == explicit.digest
    assert [message.to_dict() for message in default.messages] == [
        message.to_dict() for message in explicit.messages
    ]

    default_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    summary_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    default_loop = AgentLoop(
        default_backend,
        ConversationStore(tmp_path / "default-request"),
        skill_catalog=SkillCatalog.empty(),
        max_turns=1,
    )
    summary_store = ConversationStore(tmp_path / "summary-request")
    summary_registry = ToolRegistry(
        tmp_path,
        session_store=summary_store,
        skill_catalog=SkillCatalog.empty(),
        compaction="summary",
    )
    summary_loop = AgentLoop(
        summary_backend,
        summary_store,
        registry=summary_registry,
        skill_catalog=SkillCatalog.empty(),
        compaction="summary",
        max_turns=1,
    )

    _ = [event async for event in default_loop.run_turn("same request")]
    _ = [event async for event in summary_loop.run_turn("same request")]

    assert default_backend.request_bytes == summary_backend.request_bytes
    assert "recall_history" not in summary_registry.registered_names


@pytest.mark.asyncio
async def test_eviction_pins_latest_user_inside_retained_tail(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "old request"))
    old_call, old_result = tool_pair("read", "read-old", "old output\n" * 3000)
    store.append_message(old_call)
    store.append_message(old_result)
    store.append_message(text(MessageRole.USER, "latest request verbatim"))
    new_call, new_result = tool_pair("read", "read-new", "new output\n" * 100)
    store.append_message(new_call)
    store.append_message(new_result)

    context = await ContextAssembler(
        store,
        token_budget=1000,
        retained_tail=8,
        compaction="evict",
    ).assemble_context()

    assert "latest request verbatim" in rendered_text(context.messages)
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["pinned_message"] == text(
        MessageRole.USER, "latest request verbatim"
    ).to_dict()


@pytest.mark.asyncio
async def test_hysteresis_replay_identity_pinned_user_and_reopen(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="evict")
    store.append_message(text(MessageRole.USER, "early requirement"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    store.append_message(text(MessageRole.ASSISTANT, "old reasoning " * 20))
    store.append_message(text(MessageRole.USER, "latest request verbatim"))
    assembler = ContextAssembler(
        store, token_budget=700, retained_tail=1, compaction="evict"
    )

    assembled = [await assembler.assemble_context() for _ in range(4)]

    markers = [entry for entry in store.replay() if entry.type == "compaction"]
    assert len(markers) == 1
    assert markers[0].data["kind"] == "evict"
    assert {context.digest for context in assembled} == {assembled[0].digest}
    assert "latest request verbatim" in rendered_text(assembled[0].messages)
    assert_payload_pairing(assembled[0].messages)

    reopened = ConversationStore(sessions, session_id="evict")
    replayed = await ContextAssembler(
        reopened, token_budget=700, retained_tail=1, compaction="evict"
    ).assemble_context()
    assert replayed.digest == assembled[0].digest
    assert [message.to_dict() for message in replayed.messages] == [
        message.to_dict() for message in assembled[0].messages
    ]
    assert len([entry for entry in reopened.replay() if entry.type == "compaction"]) == 1


class EmptyEvictionPolicy(CompactionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def summarize_chunked(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return "fallback summary"


@pytest.mark.asyncio
async def test_forced_retry_reuses_existing_eviction_inside_hysteresis(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "request"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    policy = EmptyEvictionPolicy()
    assembler = ContextAssembler(
        store,
        token_budget=700,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    )
    first = await assembler.assemble_context()
    marker_count = store.compaction_marker_count()

    with patch("zeta.core.context.evict_messages", wraps=evict_messages) as eviction:
        retried = await assembler.assemble_context(force=True)

    eviction.assert_not_called()
    assert policy.calls == 0
    assert store.compaction_marker_count() == marker_count
    assert [message.to_dict() for message in retried.messages] == [
        message.to_dict() for message in first.messages
    ]


@pytest.mark.asyncio
async def test_manual_eviction_bypasses_hysteresis_without_summary_fallback(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "request"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    policy = EmptyEvictionPolicy()
    assembler = ContextAssembler(
        store,
        token_budget=700,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    )
    first = await assembler.assemble_context()

    refreshed = await assembler.assemble_context(
        force=True,
        bypass_eviction_hysteresis=True,
    )

    assert policy.calls == 0
    assert [message.to_dict() for message in refreshed.messages] == [
        message.to_dict() for message in first.messages
    ]


@pytest.mark.asyncio
async def test_eviction_can_replace_a_prior_summary(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    source = store.append_message(text(MessageRole.USER, "old request"))
    store.append_compaction_marker("large summary " * 2000, source.seq, source.seq)
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    policy = EmptyEvictionPolicy()

    await ContextAssembler(
        store,
        token_budget=1000,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    ).assemble_context()

    assert policy.calls == 0
    markers = [entry for entry in store.replay() if entry.type == "compaction"]
    assert markers[-1].data["kind"] == "evict"
    assert markers[-1].data["replaces"] == [markers[0].id]


@pytest.mark.asyncio
async def test_forced_eviction_uses_evict_mode(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "request"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    policy = EmptyEvictionPolicy()

    context = await ContextAssembler(
        store,
        token_budget=700,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    ).assemble_context(force=True)

    assert context.compacted is True
    assert policy.calls == 0
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["kind"] == "evict"


@pytest.mark.asyncio
async def test_eviction_that_fits_budget_does_not_require_target_or_summary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "request"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    tail = text(MessageRole.ASSISTANT, "small retained tail")
    store.append_message(tail)
    policy = EmptyEvictionPolicy()
    assembler = ContextAssembler(
        store,
        token_budget=1000,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    )
    fitted_result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("read-1", "deterministic digest"),
    )
    fitted = EvictionResult(
        messages=[call, fitted_result, tail],
        items_evicted=1,
        tokens_before=1200,
        tokens_after=581,
        reached_target=False,
    )

    with patch("zeta.core.context.evict_messages", return_value=fitted):
        context = await assembler.assemble_context(force=True)

    assert context.token_count <= 1000
    assert policy.calls == 0
    assert [
        entry.data.get("kind")
        for entry in store.replay()
        if entry.type == "compaction"
    ] == ["evict"]


@pytest.mark.asyncio
async def test_eviction_validates_replay_view_before_persisting(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="metadata-budget")
    for index in range(12):
        store.append_message(text(MessageRole.USER, f"u{index}"))
    call, result = tool_pair("read", "read-1", "x" * 700)
    store.append_message(call)
    store.append_message(result)
    store.append_message(text(MessageRole.USER, "latest pinned user"))
    assembler = ContextAssembler(
        store,
        token_budget=400,
        retained_tail=1,
        compaction="evict",
    )

    first = await assembler.assemble_context()

    assert first.token_count <= 400
    assert store.compaction_marker_count() == 1
    assert assembler.last_compaction_telemetry["tokens_after"] == first.token_count
    assert all(
        type(message.metadata.get("source_seq")) is int
        for message in first.messages
        if message.role is not MessageRole.USER
    )
    reopened = ConversationStore(sessions, session_id="metadata-budget")
    replayed = await ContextAssembler(
        reopened,
        token_budget=400,
        retained_tail=1,
        compaction="evict",
    ).assemble_context()
    assert [message.to_dict() for message in first.messages] == [
        message.to_dict() for message in replayed.messages
    ]


@pytest.mark.asyncio
async def test_rejected_eviction_view_leaves_store_unchanged(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.USER, "latest"))
    before = store.path.read_bytes()
    oversized = EvictionResult(
        messages=[text(MessageRole.USER, "x" * 10_000)],
        items_evicted=1,
        tokens_before=10_000,
        tokens_after=1,
        reached_target=True,
    )
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        compaction="evict",
    )

    with (
        patch("zeta.core.context.evict_messages", return_value=oversized),
        pytest.raises(RuntimeError, match="completion backend"),
    ):
        await assembler.assemble_context(force=True)

    assert store.path.read_bytes() == before
    assert store.compaction_marker_count() == 0


@pytest.mark.asyncio
async def test_eviction_falls_back_to_existing_summary(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "old user facts " * 1000))
    store.append_message(text(MessageRole.USER, "latest request"))
    policy = EmptyEvictionPolicy()

    context = await ContextAssembler(
        store,
        token_budget=300,
        retained_tail=1,
        compaction="evict",
        compaction_policy=policy,
    ).assemble_context()

    assert policy.calls == 1
    assert "fallback summary" in rendered_text(context.messages)
    assert [entry.data.get("kind", "summary") for entry in store.replay() if entry.type == "compaction"] == ["summary"]


def test_recall_tool_only_changes_the_evict_tool_surface_and_children_inherit(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    summary = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )
    evict = ToolRegistry(
        tmp_path,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        compaction="evict",
    )
    child_store = ConversationStore(tmp_path / "sessions")
    child = evict.clone_for_session(child_store)

    assert "recall_history" not in summary.registered_names
    assert "recall_history" in evict.registered_names
    assert "recall_history" in child.registered_names


def test_oversized_recall_range_pages_one_entry_without_losing_content(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    call, result = tool_pair(
        "bash",
        "call-1",
        "oversized result " * 2500,
        arguments={"command": "generate-report", "timeout": 30},
    )
    call_entry = store.append_message(call)
    entry = store.append_message(result)
    store.append_compaction_marker("summary", call_entry.seq, entry.seq)
    expected = "\n".join(
        (
            f"seq {call_entry.seq}: "
            + json.dumps(call.to_dict(), sort_keys=True, separators=(",", ":")),
            f"seq {entry.seq}: "
            + json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")),
        )
    )
    offset = 0
    chunks: list[str] = []
    seen_offsets: list[int] = []

    while True:
        page = recall_history(
            store,
            seq_start=call_entry.seq,
            seq_end=entry.seq,
            offset=offset,
            max_chars=1000,
        )
        content, marker = page.rsplit("\n[", 1)
        assert content
        chunks.append(content)
        if marker == "end of range]":
            break
        match = re.fullmatch(
            rf"truncated; continue with seq_start={call_entry.seq}, "
            rf"seq_end={entry.seq}, offset=(\d+)]",
            marker,
        )
        assert match is not None
        next_offset = int(match.group(1))
        assert next_offset > offset
        seen_offsets.append(next_offset)
        offset = next_offset

    assert seen_offsets == sorted(set(seen_offsets))
    assert "".join(chunks) == expected
    assert '"role":"tool_result"' in expected
    assert '"name":"bash"' in expected
    assert '"arguments":{"command":"generate-report","timeout":30}' in expected
    assert '"content":"oversized result ' in expected


def test_recall_query_mode_is_unchanged_by_range_pagination(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    entry = store.append_message(text(MessageRole.USER, "searchable pagination needle"))
    store.append_compaction_marker("summary", entry.seq, entry.seq)

    assert recall_history(store, query="pagination needle") == (
        f"seq {entry.seq}: "
        + json.dumps(
            text(MessageRole.USER, "searchable pagination needle").to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def test_recall_range_search_branch_isolation_and_no_mutation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    root = store.append_message(text(MessageRole.USER, "root"))
    store.append_message(text(MessageRole.ASSISTANT, "inactive forbidden secret"))
    store.append_message_fork(root.id)
    active = store.append_message(text(MessageRole.USER, "active searchable needle"))
    store.append_compaction_marker("summary", active.seq, active.seq)
    before = store.path.read_bytes()

    exact = recall_history(store, seq_start=active.seq, seq_end=active.seq)
    found = recall_history(store, query="searchable needle")
    absent = recall_history(store, query="forbidden secret")

    assert f"seq {active.seq}" in exact
    assert json.dumps(text(MessageRole.USER, "active searchable needle").to_dict(), sort_keys=True, separators=(",", ":")) in exact
    assert "active searchable needle" in found
    assert "inactive forbidden secret" not in absent
    assert store.path.read_bytes() == before


@pytest.mark.parametrize("needle", ["漢字", "🙂", "café"])
def test_recall_query_matches_non_ascii_text(tmp_path: Path, needle: str) -> None:
    store = ConversationStore(tmp_path)
    call, result = tool_pair("read", "read-1", f"before {needle} after\n" * 3)
    store.append_message(call)
    entry = store.append_message(result)
    store.append_compaction_marker("summary", entry.seq, entry.seq)

    found = recall_history(store, query=needle)

    assert "No matching compacted messages" not in found
    assert f"seq {entry.seq}:" in found


def test_evict_loop_builds_matching_registry_and_rejects_mismatch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "evict-loop")
    loop = AgentLoop(
        FakeBackend([]),
        store,
        skill_catalog=SkillCatalog.empty(),
        compaction="evict",
        max_turns=1,
    )
    assert "recall_history" in loop.tool_registry.registered_names

    summary_store = ConversationStore(tmp_path / "mismatch")
    summary_registry = ToolRegistry(
        tmp_path,
        session_store=summary_store,
        skill_catalog=SkillCatalog.empty(),
        compaction="summary",
    )
    with pytest.raises(ValueError, match="compaction mode must match"):
        AgentLoop(
            FakeBackend([]),
            summary_store,
            registry=summary_registry,
            skill_catalog=SkillCatalog.empty(),
            compaction="evict",
            max_turns=1,
        )
