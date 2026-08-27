import asyncio
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import (
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICES,
    CompactionSummary,
    SlashStatus,
    UNPRICED_MODEL_IDS,
    UsageTracker,
    UsageSnapshot,
    compaction_history,
    context_fill_percent,
    create_slash_registry,
    render_context_gauge,
)
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.providers import PROVIDER_MODELS
from zeta.providers.usage import normalize_usage
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings
from zeta.types import (
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
)


@dataclass(frozen=True, slots=True)
class FakeSlashSession:
    status: SlashStatus

    def slash_status(self) -> SlashStatus:
        return self.status


def session() -> FakeSlashSession:
    return FakeSlashSession(
        SlashStatus(
            session_id="session-1",
            provider="fake",
            model="offline",
            retained_tail=8,
            tokens_used_this_session=123,
            tokens_in_current_context=45,
            compaction_marker_count=2,
            pending_approvals=("approval-1 (write)",),
            cache_read_input_tokens=50,
            cache_creation_input_tokens=25,
            uncached_input_tokens=25,
            output_tokens_this_session=4,
        )
    )


@pytest.mark.asyncio
async def test_status_returns_live_required_fields(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="test-xyz-123")
    store.append_message(Message(MessageRole.USER, [TextContent("old")]))
    store.append_compaction_marker("summary", 1, 1)
    store.append_compaction_marker("summary 2", 1, 1)
    backend = FakeBackend([])
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=policy,
        retained_tail=17,
    )
    await loop.context_assembler.assemble()
    loop.context_assembler.record_usage(
        {
            "total_tokens": 321,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
            "input_tokens": 25,
            "output_tokens": 4,
        }
    )
    calls = [
        ToolCall("approval-live-1", "write", {}),
        ToolCall("approval-live-2", "write", {}),
    ]
    store.append_message_with_approval_requests(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(call) for call in calls],
        ),
        [(call.id, call) for call in calls],
    )
    app = TUIApp(
        loop,
        provider="provider-live",
        model="model-live",
        approval_policy=policy,
    )
    output = create_slash_registry().dispatch(app, "/status")

    assert output is not None
    assert f"session_id: {store.session_id}" in output
    assert f"provider: {app.provider}" in output
    assert f"model: {app.model}" in output
    assert "vim_mode: on" in output
    assert f"retained_tail: {loop.context_assembler.retained_tail}" in output
    assert (
        "tokens_used_this_session: "
        f"{loop.context_assembler.tokens_used_this_session}"
    ) in output
    assert (
        "tokens_in_current_context: "
        f"{loop.context_assembler.token_count}"
    ) in output
    assert f"compaction_marker_count: {store.compaction_marker_count()}" in output
    assert "compaction_marker_count: 2" in output
    assert "live_pending_approvals: 2" in output
    assert "prompt_cache_read: 50" in output
    assert "prompt_cache_write: 25" in output
    assert "prompt_cache_uncached_input: 25" in output
    assert "prompt_cache_hit_rate: 50.0%" in output
    assert "output_tokens_this_session: 4" in output


@pytest.mark.asyncio
async def test_status_counts_compaction_usage(tmp_path: Path) -> None:
    def token_count(message: Message) -> int:
        if message.role is MessageRole.COMPACTION or message.metadata.get(
            "compaction_summary"
        ):
            return 1
        return 40

    backend = FakeBackend(
        [
            ScriptedTurn(
                [TextContent("first")],
                usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            ScriptedTurn(
                [TextContent("summary")],
                usage={"input_tokens": 30, "output_tokens": 3, "total_tokens": 33},
            ),
            ScriptedTurn(
                [TextContent("second")],
                usage={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            ),
        ]
    )
    store = ConversationStore(tmp_path / "sessions")
    assembler = ContextAssembler(
        store,
        backend=backend,
        token_budget=80,
        retained_tail=1,
        token_counter=token_count,
    )
    loop = AgentLoop(backend, store, context_assembler=assembler)
    app = TUIApp(loop, provider="claude", model="claude-sonnet-4-6")

    await app._consume_turn("first")
    await app._consume_turn("second")

    output = create_slash_registry().dispatch(app, "/status")

    assert output is not None
    assert "tokens_used_this_session: 70" in output
    assert [
        (snapshot.turn, snapshot.input_tokens, snapshot.output_tokens)
        for snapshot in app.slash_status().usage_history
    ] == [(1, 10, 2), (2, 20, 5)]
    assert "estimated_cost_usd: $0.035357" in output


def test_usage_cost_keeps_all_turns_outside_bounded_trend() -> None:
    class Counters:
        uncached_input_tokens_this_session = 0
        output_tokens_this_session = 0
        cache_read_input_tokens_this_session = 0
        cache_creation_input_tokens_this_session = 0

    counters = Counters()
    tracker = UsageTracker(counters)
    for _ in range(9):
        counters.uncached_input_tokens_this_session += 1_000_000
        tracker.record(StreamEventType.TURN_END, "claude-sonnet-4-6")

    status = replace(
        session().status,
        provider="claude",
        model="claude-sonnet-4-6",
        usage_history=tracker.history,
        usage_cost_by_model=tracker.cost_by_model,
    )
    output = create_slash_registry().dispatch(FakeSlashSession(status), "/status")

    assert output is not None
    assert len(tracker.history) == 8
    assert "estimated_cost_usd: $27.000000" in output


def test_status_renders_cache_hit_rate_as_na_without_usage() -> None:
    empty_session = FakeSlashSession(
        replace(
            session().status,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            uncached_input_tokens=0,
        )
    )
    output = create_slash_registry().dispatch(empty_session, "/status")

    assert output is not None
    assert "prompt_cache_hit_rate: n/a" in output


def test_status_renders_usage_trend_cost_and_context_gauge() -> None:
    output = create_slash_registry().dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="claude",
                model="claude-sonnet-4-6",
                uncached_input_tokens=100,
                output_tokens_this_session=50,
                cache_read_input_tokens=50,
                cache_creation_input_tokens=25,
                tokens_in_current_context=50,
                model_window=100,
                usage_history=(
                    UsageSnapshot(
                        1,
                        input_tokens=100,
                        cache_creation_input_tokens=25,
                        model="claude-sonnet-4-6",
                    ),
                    UsageSnapshot(
                        2,
                        input_tokens=100,
                        cache_read_input_tokens=50,
                        model="claude-sonnet-4-6",
                    ),
                ),
            )
        ),
        "/status",
    )

    assert output is not None
    assert "cache_hit_trend: 1:0% 2:33%" in output
    assert "estimated_cost_usd: $0.000709" in output
    assert "window: 50 / 100" in output
    assert "fill: [##########----------] 50%" in output


def test_status_marks_unknown_model_cost_and_window() -> None:
    output = create_slash_registry().dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="claude",
                model="claude-future",
                tokens_in_current_context=50,
            )
        ),
        "/status",
    )

    assert output is not None
    assert "estimated_cost_usd: unavailable (unknown model: claude-future)" in output
    assert "fill: [????????????????????] unknown" in output


@pytest.mark.parametrize(
    ("tokens", "window", "expected"),
    [(0, 100, 0), (100, 100, 100), (200, 100, 100), (-1, 100, 0), (50, 0, None)],
)
def test_context_fill_percent_boundaries(
    tokens: int, window: int, expected: int | None
) -> None:
    assert context_fill_percent(tokens, window) == expected


def test_context_gauge_unknown_window_is_bounded() -> None:
    assert render_context_gauge(10, None, width=8) == "[????????] unknown"


def test_status_renders_compaction_history_and_empty_state() -> None:
    empty = create_slash_registry().dispatch(session(), "/status")
    assert empty is not None
    assert "compaction_history:\n  none" in empty

    status = replace(
        session().status,
        compaction_history=(CompactionSummary(3, 4, 120),),
    )
    output = create_slash_registry().dispatch(
        FakeSlashSession(status), "/status"
    )
    assert output is not None
    assert "compaction_history:\n  turn 3: 4 entries, 120 tokens saved" in output


def test_price_table_covers_current_provider_models() -> None:
    for provider, models in PROVIDER_MODELS.items():
        assert models == MODEL_PRICES[provider].keys()
        assert models == MODEL_CONTEXT_WINDOWS[provider].keys()
        for model in models:
            pricing = MODEL_PRICES[provider][model]
            window = MODEL_CONTEXT_WINDOWS[provider][model]
            if model in UNPRICED_MODEL_IDS[provider]:
                assert pricing is None
                assert window is None
            else:
                assert pricing is not None
                assert window is not None


def test_codex_cache_writes_use_existing_usage_categories() -> None:
    assert normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 4,
            "input_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 70,
            },
        }
    ) == {
        "input_tokens": 10,
        "output_tokens": 4,
        "input_tokens_details": {
            "cached_tokens": 20,
            "cache_write_tokens": 70,
        },
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 70,
    }


def test_cost_uses_the_model_for_each_turn() -> None:
    output = create_slash_registry().dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="claude",
                model="claude-opus-4-6",
                usage_history=(
                    UsageSnapshot(
                        1,
                        input_tokens=1_000_000,
                        model="claude-sonnet-4-6",
                    ),
                    UsageSnapshot(
                        2,
                        input_tokens=1_000_000,
                        model="claude-opus-4-6",
                    ),
                ),
            )
        ),
        "/status",
    )

    assert output is not None
    assert "estimated_cost_usd: $8.000000" in output


def test_gpt_5_5_has_published_cost() -> None:
    output = create_slash_registry().dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="codex",
                model="gpt-5.5",
                usage_history=(
                    UsageSnapshot(
                        1,
                        input_tokens=1_000_000,
                        output_tokens=1_000_000,
                        model="gpt-5.5",
                    ),
                ),
            )
        ),
        "/status",
    )

    assert output is not None
    assert "estimated_cost_usd: $35.000000" in output


def test_compaction_history_counts_folded_messages_only(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    for index in range(2):
        store.append_message(
            Message(MessageRole.USER, [TextContent(f"message {index}")])
        )
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"reply {index}")])
        )
    store.append_checkpoint("before warning")
    store._append_row("warning", {"message": "ignored"})
    store.append_compaction_marker("summary", 1, 6)

    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
    )

    history = app.slash_status().compaction_history
    assert len(history) == 1
    assert history[0].entries_folded == 4


def test_compaction_history_subtracts_both_replacements(tmp_path: Path) -> None:
    def token_count(message: Message) -> int:
        if message.role is MessageRole.COMPACTION:
            return 3
        if message.metadata.get("compaction_summary"):
            return 1
        return 9

    store = ConversationStore(tmp_path / "sessions")
    source = [
        store.append_message(Message(MessageRole.USER, [TextContent("source")])),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")])),
        store.append_message(Message(MessageRole.USER, [TextContent("source 2")])),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply 2")])),
    ]
    store.append_compaction_marker(
        "summary",
        source[0].seq,
        source[-1].seq,
        replaces=[entry.id for entry in source],
    )

    history = compaction_history(store.replay(), token_count)

    assert history == (CompactionSummary(2, 4, 32),)


def test_repeated_compaction_counts_only_new_source_entries(tmp_path: Path) -> None:
    def token_count(message: Message) -> int:
        if message.role is MessageRole.COMPACTION:
            return 3
        if message.metadata.get("compaction_summary"):
            return 1
        return 9

    store = ConversationStore(tmp_path / "sessions")
    first_source = [
        store.append_message(Message(MessageRole.USER, [TextContent("source")])),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")])),
    ]
    first_marker = store.append_compaction_marker(
        "first summary",
        first_source[0].seq,
        first_source[-1].seq,
        replaces=[entry.id for entry in first_source],
    )
    second_source = [
        store.append_message(Message(MessageRole.USER, [TextContent("new source")])),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("new reply")])),
        store.append_message(Message(MessageRole.USER, [TextContent("new source 2")])),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("new reply 2")])),
    ]
    store.append_compaction_marker(
        "second summary",
        first_source[0].seq,
        second_source[-1].seq,
        replaces=[first_marker.id, *(entry.id for entry in second_source)],
    )

    history = compaction_history(store.replay(), token_count)

    assert history[0] == CompactionSummary(1, 2, 14)
    assert history[1] == CompactionSummary(3, 4, 36)


def test_compaction_history_uses_the_active_fork_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("original")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")]))
    checkpoint = store.append_checkpoint("saved")
    store.append_message(Message(MessageRole.USER, [TextContent("abandoned")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("later")]))
    store.append_compaction_marker("abandoned summary", 1, 5)
    store.append_fork(str(checkpoint.seq))
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
    )

    assert app.slash_status().compaction_history == ()


def test_unknown_command_passes_through_unchanged() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "/unknown arg") is None
    assert registry.input_for_model("/unknown arg") == "/unknown arg"


def test_double_slash_escapes_registered_command() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "//status") is None
    assert registry.input_for_model("//status") == "/status"


def test_multiline_known_command_consumes_the_whole_message() -> None:
    output = create_slash_registry().dispatch(
        session(), "/status\nmodel must not see this"
    )

    assert output is not None
    assert "model must not see this" not in output


def test_empty_message_does_nothing() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "") is None
    assert registry.input_for_model("") == ""


def test_model_command_shows_and_changes_the_model() -> None:
    class ModelSession:
        def __init__(self) -> None:
            self.model = "offline"

        def slash_status(self) -> SlashStatus:
            return session().status

        def slash_model(self, args: str) -> str:
            if args:
                self.model = args
            return f"model: {self.model}"

        async def slash_compact(self) -> str:
            return "compact: nothing to compact"

    model_session = ModelSession()
    registry = create_slash_registry()

    assert registry.dispatch(model_session, "/model") == "model: offline"
    assert registry.dispatch(model_session, "/model faster") == "model: faster"


@pytest.mark.asyncio
async def test_compact_command_uses_async_dispatch() -> None:
    class CompactSession:
        def slash_status(self) -> SlashStatus:
            return session().status

        def slash_model(self, args: str) -> str:
            del args
            return "model: offline"

        async def slash_compact(self) -> str:
            return "compacted entries 1–2; tokens after: 3"

    output = await create_slash_registry().dispatch_async(
        CompactSession(), "/compact"
    )

    assert output == "compacted entries 1–2; tokens after: 3"


@pytest.mark.asyncio
async def test_tui_renders_status_without_calling_the_model(tmp_path: Path) -> None:
    backend = FakeBackend([])
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    with create_pipe_input() as pipe:
        prompt = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=app.abort_active,
                on_exit=app.request_exit,
            ),
            multiline=True,
        )
        run_task = asyncio.create_task(app.run(prompt))
        pipe.send_text("/status\r")
        for _ in range(100):
            if "session_id:" in output.getvalue():
                break
            await asyncio.sleep(0.01)
        pipe.send_text("\x04")
        await run_task

    assert "session_id:" in output.getvalue()
    assert "provider: fake" in output.getvalue()
    assert backend.calls == []
