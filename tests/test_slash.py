from zeta.skills import SkillCatalog

import asyncio
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.model_input import ModelInputEnvelope
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
    budget_for_model,
    resolve_session_budget,
)
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.providers import PROVIDER_MODELS
from zeta.providers.usage import normalize_usage
from zeta.skills import discover_session_skills
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings
from zeta.tui.composer import ComposerCompleter, DollarSkillCompleter, SlashCompleter
from zeta.tui.user import displayed_user_text
from zeta.protocol.types import (
    with_message_origin,
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
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
            provider="codex",
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


def _write_skill(path: Path, name: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_ollama_context_budgets_are_known_and_capped() -> None:
    assert budget_for_model("ollama", "qwen3:4b") == 40_960
    assert budget_for_model("ollama", "qwen3:4b-instruct") == 32_768
    assert resolve_session_budget(0, False, "ollama", "qwen3:4b", None) == (
        40_960,
        False,
    )
    assert resolve_session_budget(0, False, "ollama", "qwen3:4b", 8_192) == (
        8_192,
        True,
    )
    assert resolve_session_budget(0, False, "ollama", "qwen3:4b", 100_000) == (
        40_960,
        True,
    )
    assert resolve_session_budget(100_000, True, "ollama", "qwen3:4b", None) == (
        40_960,
        True,
    )
    assert resolve_session_budget(100_000, True, "ollama", "qwen3:4b-instruct", None) == (
        32_768,
        True,
    )
    assert resolve_session_budget(16_000, True, "ollama", "qwen3:4b-instruct", None) == (
        16_000,
        True,
    )
    assert resolve_session_budget(
        100_000, True, "ollama", "locally-created-model", None
    ) == (8_192, True)


def test_skill_slash_commands_follow_collision_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_skill(home / "skills" / "review.md", "review", "home review")
    _write_skill(home / "skills" / "custom.md", "custom", "home custom")
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "project review")
    _write_skill(project / ".zeta" / "skills" / "status.md", "status", "shadowed")
    _write_skill(project / ".zeta" / "skills" / "unique.md", "unique", "unique body")
    command_dir = home / "commands"
    command_dir.mkdir(parents=True)
    (command_dir / "custom.md").write_text("custom command", encoding="utf-8")

    registry = create_slash_registry(
        zeta_home=home,
        project_dir=project,
        skill_catalog=discover_session_skills(home=home, project_dir=project),
    )

    result = registry.dispatch(session(), "/unique")
    assert isinstance(result, ModelInputEnvelope)
    assert result.text == "unique body"
    assert registry.dispatch(session(), "/status") is not None
    assert registry.input_for_model("/custom").text == "custom command"
    assert "ignored skill" in "\n".join(registry.notices)
    assert any("shadows built-in /status" in notice for notice in registry.notices)
    assert any(
        "shadows custom command /custom" in notice for notice in registry.notices
    )


def test_skill_commands_appear_in_completer_with_source_badge(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "unique.md", "unique", "unique body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/uni"), CompleteEvent(completion_requested=True)
        )
    )

    assert len(completions) == 1
    assert completions[0].text == "unique"
    assert "[project] unique description" in str(completions[0].display_meta)


def test_dollar_skill_at_start_matches_slash_skill(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    dollar = registry.dispatch(session(), "$review this branch")
    slash = registry.dispatch(session(), "/review this branch")

    assert dollar.text == slash.text
    assert dollar.origin is slash.origin
    assert dollar.display_text == "$review this branch"
    assert slash.display_text == "/review this branch"
    assert dollar.text == "review body\n\nUser request:\nthis branch"
    assert isinstance(dollar, ModelInputEnvelope)
    assert dollar.display_text == "$review this branch"
    assert isinstance(slash, ModelInputEnvelope)
    assert slash.display_text == "/review this branch"


def test_skill_invocation_keeps_trailing_request_text(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    for value in ("/review PR 123\nfocus on tests", "$review PR 123\nfocus on tests"):
        result = registry.dispatch(session(), value)
        assert isinstance(result, ModelInputEnvelope)
        assert result.text == "review body\n\nUser request:\nPR 123\nfocus on tests"

    for value in ("/review", "$review", "/review   ", "$review\n"):
        result = registry.dispatch(session(), value)
        assert isinstance(result, ModelInputEnvelope)
        assert result.text == "review body"


def test_inline_dollar_skills_load_in_mention_order_and_dedupe(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    _write_skill(project / ".zeta" / "skills" / "test.md", "test", "test body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    result = registry.dispatch(
        session(), "please $review then $test and $review this branch"
    )

    assert isinstance(result, ModelInputEnvelope)
    assert result.text == (
        "review body\n\ntest body\n\n"
        "User request:\nplease $review then $test and $review this branch"
    )


@pytest.mark.parametrize(
    "value",
    [
        "$5",
        "$HOME",
        "$(cmd)",
        "${VAR}",
        "a$review",
        "$unknown-skill",
        "please `$review` this",
        "please ```\n$review\n``` this",
        "! echo $review",
        "!! echo $review",
    ],
)
def test_non_skill_dollar_tokens_stay_literal(tmp_path: Path, value: str) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    assert registry.dispatch(session(), value) is None
    assert registry.input_for_model(value).text == value


def test_dollar_completion_lists_only_skills_and_filters_mid_message(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    _write_skill(project / ".zeta" / "skills" / "research.md", "research", "research body")
    command_dir = project / ".zeta" / "commands"
    command_dir.mkdir(parents=True)
    (command_dir / "report.md").write_text("report command", encoding="utf-8")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )
    registry.set_mcp_prompts(
        [("remote", "remote prompt", object())]  # type: ignore[list-item]
    )

    completions = list(
        ComposerCompleter(registry).get_completions(
            Document("please $rev"), CompleteEvent(completion_requested=True)
        )
    )

    assert [
        (item.text, item.display[0][1], item.display_meta[0][1])
        for item in completions
    ] == [("review ", "$review", "[project] review description")]
    all_names = {
        item.text.strip()
        for item in DollarSkillCompleter(registry).get_completions(
            Document("$"), CompleteEvent(completion_requested=True)
        )
    }
    assert all_names == {"review", "research"}
    assert {"status", "report", "remote"}.isdisjoint(all_names)


def test_dollar_completion_acceptance_adds_space_and_shell_mode_is_excluded(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )
    completer = ComposerCompleter(registry)
    buffer = Buffer(completer=completer, document=Document("please $rev"))
    completion = next(
        completer.get_completions(
            buffer.document, CompleteEvent(completion_requested=True)
        )
    )

    buffer.apply_completion(completion)

    assert buffer.text == "please $review "
    assert list(
        completer.get_completions(
            Document("! echo $rev"), CompleteEvent(completion_requested=True)
        )
    ) == []


def test_slash_completion_is_unchanged_when_dollar_skills_are_enabled(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    completions = list(
        ComposerCompleter(registry).get_completions(
            Document("/sta"), CompleteEvent(completion_requested=True)
        )
    )

    assert [(item.text, item.display[0][1]) for item in completions] == [
        ("status", "/status")
    ]


@pytest.mark.asyncio
async def test_inline_dollar_skill_request_is_sent_and_persisted(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    catalog = discover_session_skills(project_dir=project)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=project)
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=catalog),
        provider="codex",
        model="offline",
        zeta_home=tmp_path / "home",
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("please $review this branch")
    assert app._active_task is not None
    await app._active_task

    expected = "review body\n\nUser request:\nplease $review this branch"
    sent = next(
        message for message in backend.calls[0][0] if message.role is MessageRole.USER
    )
    persisted = next(
        message for message in store.messages() if message.role is MessageRole.USER
    )
    assert sent.content[0].text == expected
    assert persisted.content[0].text == expected
    assert displayed_user_text(persisted) == "please $review this branch"
    assert (
        persisted.metadata[MESSAGE_ORIGIN_METADATA]
        == MessageOrigin.SKILL_EXPANSION
    )
    await app.close()


def test_help_documents_both_skill_invocation_forms(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_skill(project / ".zeta" / "skills" / "review.md", "review", "review body")
    registry = create_slash_registry(
        project_dir=project,
        skill_catalog=discover_session_skills(project_dir=project),
    )

    assert "skills (use /name or $name; $name also works inline):" in registry.help_text()
    assert "/review" in registry.help_text()


def test_directory_skill_slash_load_reports_resource_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "bundle"
    _write_skill(skill_dir / "SKILL.md", "bundle", "bundle body")
    catalog = discover_session_skills(home=tmp_path)
    registry = create_slash_registry(skill_catalog=catalog)

    result = registry.dispatch(session(), "/bundle")

    assert isinstance(result, ModelInputEnvelope)
    assert result.text == (
        "bundle body\n\nSkill resources directory: "
        f"{skill_dir.resolve()}"
    )


@pytest.mark.asyncio
async def test_status_returns_live_required_fields(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="test-xyz-123")
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("old")]), MessageOrigin.USER))
    store.append_compaction_marker("summary", 1, 1)
    store.append_compaction_marker("summary 2", 1, 1)
    backend = FakeBackend([])
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=policy,
        retained_tail=17,
skill_catalog=SkillCatalog.empty(),
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
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(app, "/status")

    assert output is not None
    assert f"session_id: {store.session_id}" in output
    assert f"provider: {app.provider}" in output
    assert f"model: {app.model}" in output
    assert "vim_mode: on" in output
    assert f"retained_tail: {loop.context_assembler.retained_tail}" in output
    assert (
        f"tokens_used_this_session: {loop.context_assembler.tokens_used_this_session}"
    ) in output
    assert (
        f"tokens_in_current_context: {loop.context_assembler.token_count}"
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
    loop = AgentLoop(backend, store, context_assembler=assembler, skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="claude", model="claude-sonnet-4-6")

    await app._consume_turn("first")
    await app._consume_turn("second")

    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(app, "/status")

    assert output is not None
    assert "tokens_used_this_session: 70" in output
    assert [
        (snapshot.turn, snapshot.input_tokens, snapshot.output_tokens)
        for snapshot in app.slash_status().usage_history
    ] == [(1, 10, 2), (2, 20, 5)]
    pricing = MODEL_PRICES["claude"]["claude-sonnet-4-6"]
    assert pricing is not None
    expected_cost = (
        sum(
            snapshot.input_tokens * pricing.input
            + snapshot.cache_read_input_tokens * pricing.cache_read
            + snapshot.cache_creation_input_tokens * (pricing.cache_write or 0)
            + snapshot.output_tokens * pricing.output
            for snapshot in app.slash_status().usage_cost_by_model
        )
        / 1_000_000
    )
    assert f"estimated_cost_usd: ${expected_cost:.6f}" in output


@pytest.mark.asyncio
async def test_manual_compact_captures_summarization_cost(tmp_path: Path) -> None:
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
                usage={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            ),
            ScriptedTurn(
                [TextContent("summary")],
                usage={"input_tokens": 40, "output_tokens": 2, "total_tokens": 42},
            ),
        ]
    )
    store = ConversationStore(tmp_path / "sessions")
    assembler = ContextAssembler(
        store,
        backend=backend,
        token_budget=1_000_000,
        retained_tail=1,
        token_counter=token_count,
    )
    loop = AgentLoop(backend, store, context_assembler=assembler, skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="claude", model="claude-sonnet-4-6")

    await app._consume_turn("first")
    baseline_tokens = assembler.uncached_input_tokens_this_session
    assert await app.slash_compact() != "compact: nothing to compact"

    per_model = {
        snapshot.model: snapshot for snapshot in app.slash_status().usage_cost_by_model
    }
    snapshot = per_model["claude-sonnet-4-6"]
    assert snapshot.input_tokens >= baseline_tokens + 40
    assert snapshot.output_tokens >= 3


@pytest.mark.asyncio
async def test_errored_turn_usage_does_not_leak_to_next_model(tmp_path: Path) -> None:
    class _ErrorBackend(FakeBackend):
        async def complete(self, messages, tool_schemas):
            self.calls.append((list(messages), list(tool_schemas)))
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(
                    role=MessageRole.ASSISTANT, content=[TextContent("boom")]
                ),
                data={
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "total_tokens": 105,
                    }
                },
            )
            raise RuntimeError("backend exploded after usage")

    store = ConversationStore(tmp_path / "sessions")
    error_backend = _ErrorBackend([])
    loop = AgentLoop(error_backend, store, skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="claude", model="claude-sonnet-4-6")

    await app._consume_turn("first")

    app.model = "claude-opus-4-6"
    per_model = {
        snapshot.model: snapshot for snapshot in app.slash_status().usage_cost_by_model
    }
    assert "claude-sonnet-4-6" in per_model
    assert "claude-opus-4-6" not in per_model
    assert per_model["claude-sonnet-4-6"].input_tokens == 100


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
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(FakeSlashSession(status), "/status")

    assert output is not None
    assert len(tracker.history) == 8
    assert "estimated_cost_usd: $27.000000" in output


def test_status_shows_last_automatic_memory_failure() -> None:
    failed_session = FakeSlashSession(
        replace(
            session().status,
            automatic_memory_failure=(
                "2026-10-07T17:00:00+00:00 source range is not in the transcript "
                "(seq 1281-1281; skipped)"
            ),
        )
    )

    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        failed_session, "/status"
    )

    assert output is not None
    assert "automatic_memory_failure: 2026-10-07T17:00:00+00:00" in output
    assert "source range is not in the transcript (seq 1281-1281; skipped)" in output


def test_status_renders_cache_hit_rate_as_na_without_usage() -> None:
    empty_session = FakeSlashSession(
        replace(
            session().status,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            uncached_input_tokens=0,
        )
    )
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(empty_session, "/status")

    assert output is not None
    assert "prompt_cache_hit_rate: n/a" in output


def test_status_includes_child_usage_without_misattributing_cost() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        FakeSlashSession(
            replace(
                session().status,
                child_cache_read_input_tokens=70,
                child_cache_creation_input_tokens=10,
                child_uncached_input_tokens=30,
                child_output_tokens_this_session=3,
            )
        ),
        "/status",
    )

    assert output is not None
    assert "prompt_cache_read: 120" in output
    assert "prompt_cache_write: 35" in output
    assert "prompt_cache_uncached_input: 55" in output
    assert "prompt_cache_hit_rate: 57.1%" in output
    assert "tokens_used_this_session: 236" in output
    assert "output_tokens_this_session: 7" in output
    assert "child_output_tokens: 3" in output
    assert "child_cache_hit_rate: 63.6%" in output
    assert "cache_hit_trend: none (parent turns only)" in output
    assert "estimated_cost_usd: unavailable (unknown model: offline) (parent only)" in output


def test_status_renders_usage_trend_cost_and_context_gauge() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
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
    assert "estimated_cost_usd: $0.000765" in output
    assert "window: 50 / 100" in output
    assert "fill: [##########----------] 50%" in output


def test_status_marks_unknown_model_cost_and_window() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
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
    empty = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(session(), "/status")
    assert empty is not None
    assert "compaction_history:\n  none" in empty

    status = replace(
        session().status,
        compaction_history=(CompactionSummary(3, 4, 120),),
    )
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(FakeSlashSession(status), "/status")
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
                if (provider, model) not in {
                    ("ollama", "qwen3:4b"),
                    ("ollama", "qwen3:4b-instruct"),
                }:
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
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
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


def test_gpt_5_5_long_context_applies_published_multiplier() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
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
    assert "estimated_cost_usd: $55.000000" in output


def test_gpt_5_5_below_long_context_threshold_uses_base_rates() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="codex",
                model="gpt-5.5",
                usage_history=(
                    UsageSnapshot(
                        1,
                        input_tokens=100_000,
                        output_tokens=100_000,
                        model="gpt-5.5",
                    ),
                ),
            )
        ),
        "/status",
    )

    assert output is not None
    assert "estimated_cost_usd: $3.500000" in output


def test_gpt_5_5_window_matches_published_value() -> None:
    assert MODEL_CONTEXT_WINDOWS["codex"]["gpt-5.5"] == 1_050_000


def test_codex_cache_write_is_free() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        FakeSlashSession(
            replace(
                session().status,
                provider="codex",
                model="gpt-5.6-sol",
                usage_history=(
                    UsageSnapshot(
                        1,
                        input_tokens=100,
                        cache_creation_input_tokens=1_000_000,
                        model="gpt-5.6-sol",
                    ),
                ),
            )
        ),
        "/status",
    )

    assert output is not None
    assert "estimated_cost_usd: $0.000800" in output


def test_compaction_history_counts_folded_messages_only(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    for index in range(2):
        store.append_message(
            with_message_origin(Message(MessageRole.USER, [TextContent(f"message {index}")]), MessageOrigin.USER)
        )
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"reply {index}")])
        )
    store.append_checkpoint("before warning")
    store._append_row("warning", {"message": "ignored"})
    store.append_compaction_marker("summary", 1, 6)

    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
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
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("source")]), MessageOrigin.USER)),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")])),
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("source 2")]), MessageOrigin.USER)),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply 2")])),
    ]
    store.append_compaction_marker(
        "summary",
        source[0].seq,
        source[-1].seq,
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
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("source")]), MessageOrigin.USER)),
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")])),
    ]
    first_marker = store.append_compaction_marker(
        "first summary",
        first_source[0].seq,
        first_source[-1].seq,
    )
    second_source = [
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("new source")]), MessageOrigin.USER)),
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("new reply")])
        ),
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("new source 2")]), MessageOrigin.USER)),
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("new reply 2")])
        ),
    ]
    store.append_compaction_marker(
        "second summary",
        first_source[0].seq,
        second_source[-1].seq,
        replaces=[first_marker.id],
    )

    history = compaction_history(store.replay(), token_count)

    assert history[0] == CompactionSummary(1, 2, 14)
    assert history[1] == CompactionSummary(3, 4, 36)


def test_compaction_history_uses_the_active_fork_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("original")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")]))
    checkpoint = store.append_checkpoint("saved")
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("abandoned")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("later")]))
    store.append_compaction_marker("abandoned summary", 1, 5)
    store.append_fork(str(checkpoint.seq))
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
    )

    assert app.slash_status().compaction_history == ()


def test_unknown_command_passes_through_unchanged() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert registry.dispatch(session(), "/unknown arg") is None
    envelope = registry.input_for_model("/unknown arg")
    assert envelope.text == "/unknown arg"
    assert envelope.display_text == "/unknown arg"
    assert envelope.origin is MessageOrigin.USER


def test_double_slash_escapes_registered_command() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert registry.dispatch(session(), "//status") is None
    envelope = registry.input_for_model("//status")
    assert envelope.text == "/status"
    assert envelope.display_text == "//status"
    assert envelope.origin is MessageOrigin.USER


def test_multiline_known_command_consumes_the_whole_message() -> None:
    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        session(), "/status\nmodel must not see this"
    )

    assert output is not None
    assert "model must not see this" not in output


def test_empty_message_does_nothing() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert registry.dispatch(session(), "") is None
    assert registry.input_for_model("").text == ""
    assert registry.exec_command_for("/") is None


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
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

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

    output = await create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch_async(CompactSession(), "/compact")

    assert output == "compacted entries 1–2; tokens after: 3"


@pytest.mark.asyncio
async def test_mcp_command_dispatches_status_and_reconnect() -> None:
    class MCPTestSession:
        async def slash_mcp(self, args: str) -> str:
            return f"mcp args: {args}"

    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())
    session_value = MCPTestSession()

    assert await registry.dispatch_async(session_value, "/mcp") == "mcp args: "
    assert (
        await registry.dispatch_async(session_value, "/mcp reconnect server")
        == "mcp args: reconnect server"
    )


@pytest.mark.asyncio
async def test_tui_renders_status_without_calling_the_model(tmp_path: Path) -> None:
    backend = FakeBackend([])
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="codex",
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
    assert "provider: codex" in output.getvalue()
    assert backend.calls == []


@pytest.mark.asyncio
async def test_slash_webhook_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime
    from zeta.automations import commands
    from zeta.automations.models import parse_job
    from zeta.automations.store import SQLiteStore

    def job(name: str, webhook: bool = True):
        trigger = {"kind": "webhook", "verify": "github"} if webhook else {
            "kind": "schedule", "cron": "0 9 * * *", "timezone": "UTC"
        }
        return parse_job(name, {
            "prompt": "test", "trigger": trigger, "servers": [], "allow": [],
            "deliver": "slack:U123", "provider": "codex", "model": "fake",
        "cwd": str(tmp_path),
        })

    with SQLiteStore(tmp_path) as store:
        for name, webhook in (("hook", True), ("plain", False), ("off", True)):
            state = store.draft(job(name, webhook))
            store.approve(name, state.revision, "U123", datetime.now(UTC))
        store.disable("off")
        before = store.webhook_credentials("hook")

    class FakeMount:
        async def close(self) -> None:
            return None

    class FakeSlackDelivery:
        def __init__(self, _mount: object) -> None:
            pass

        async def resolve(self, _target: str) -> str:
            return "U123"

    async def fake_mount(*_args: object, **_kwargs: object) -> FakeMount:
        return FakeMount()

    monkeypatch.setattr(commands, "mount_services", fake_mount)
    monkeypatch.setattr(commands, "SlackDelivery", FakeSlackDelivery)
    with SQLiteStore(tmp_path) as store:
        review = await commands.review_job(store, "hook", tmp_path)
    assert before.secret.hex() not in review.text and before.token not in review.text

    url = await commands.slash("webhook url hook", home=tmp_path, cwd=str(tmp_path))
    assert before.token in url
    assert await commands.slash(
        "webhook show-secret hook", home=tmp_path, cwd=str(tmp_path)
    ) == before.secret.hex()
    await commands.slash("webhook rotate-secret hook", home=tmp_path, cwd=str(tmp_path))
    with SQLiteStore(tmp_path) as store:
        after_secret = store.webhook_credentials("hook")
    assert after_secret.secret != before.secret and after_secret.token == before.token
    await commands.slash("webhook rotate-url hook", home=tmp_path, cwd=str(tmp_path))
    with SQLiteStore(tmp_path) as store:
        after_url = store.webhook_credentials("hook")
    assert after_url.secret == after_secret.secret and after_url.token != after_secret.token
    listing = await commands.slash("", home=tmp_path, cwd=str(tmp_path))
    shown = await commands.slash("show hook", home=tmp_path, cwd=str(tmp_path))
    for private in (after_url.secret.hex(), after_url.token):
        assert private not in listing and private not in shown
    for name in ("plain", "off"):
        for operation in ("url", "show-secret", "rotate-secret", "rotate-url"):
            with pytest.raises(ValueError):
                await commands.slash(
                    f"webhook {operation} {name}", home=tmp_path, cwd=str(tmp_path)
                )
