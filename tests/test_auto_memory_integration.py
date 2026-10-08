from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.config.settings import load_settings, resolve
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.project_context import ProjectContext
from zeta.core.session import OpenedSession, SessionManager
from zeta.core.slash import create_slash_registry
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolSchema,
    with_message_origin,
)
from zeta.runtime.composition import RuntimeComposition, compose_runtime
from zeta.server.server import _Client
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.transcript_search.index import TranscriptIndex


class _MemoryScriptBackend(CompletionBackend):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del tools
        prompt = "".join(
            block.text
            for message in messages
            for block in message.content
            if isinstance(block, TextContent)
        )
        self.prompts.append(prompt)
        rows = json.loads(prompt.split("Completed transcript rows:", 1)[1].strip())
        session_id = prompt.split("from session\n", 1)[1].split(".", 1)[0].strip()
        start, end = rows[0]["seq"], rows[-1]["seq"]
        proposal = json.dumps(
            {
                "changes": [
                    {
                        "file": "decisions.md",
                        "content": f"# Decisions\n\ncomposed range {start}-{end}\n",
                        "sources": [
                            {
                                "session_id": session_id,
                                "seq_start": start,
                                "seq_end": end,
                            }
                        ],
                    }
                ]
            }
        )
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(proposal)]),
            data={"usage": {"input_tokens": 10, "output_tokens": 5}},
        )


class _UnusedBackend(CompletionBackend):
    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools
        if False:
            yield StreamEvent(StreamEventType.MESSAGE_END)


def _resolve(home: Path, *, cli_auto_memory: bool | None = None):
    settings = load_settings(home=home).settings
    return resolve(
        settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
        cli_auto_memory=cli_auto_memory,
    )


def test_memory_settings_defaults_and_overrides(tmp_path: Path) -> None:
    defaults = _resolve(tmp_path)
    assert defaults.memory_auto is True
    assert defaults.memory_model == "gpt-5.6-luna"
    assert defaults.memory_token_threshold == 50_000
    assert defaults.memory_idle_minutes == 10

    (tmp_path / "settings.toml").write_text(
        """[memory]
auto = false
model = "gpt-5.6-terra"
token_threshold = 1234
idle_minutes = 2
""",
        encoding="utf-8",
    )
    configured = _resolve(tmp_path)
    assert configured.memory_auto is False
    assert configured.memory_model == "gpt-5.6-terra"
    assert configured.memory_token_threshold == 1234
    assert configured.memory_idle_minutes == 2
    assert _resolve(tmp_path, cli_auto_memory=True).memory_auto is True


def test_memory_slash_command_dispatches_log_and_undo() -> None:
    calls: list[str] = []
    session = SimpleNamespace(
        slash_memory=lambda args: calls.append(args) or f"memory:{args}"
    )
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert registry.dispatch(session, "/memory log") == "memory:log"
    assert registry.dispatch(session, "/memory undo") == "memory:undo"
    assert calls == ["log", "undo"]


@pytest.mark.asyncio
async def test_composed_runtime_reconciles_real_persisted_trigger_and_notices(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home.mkdir()
    (home / "settings.toml").write_text(
        '[memory]\nmodel = "gpt-5.6-luna"\ntoken_threshold = 1\n',
        encoding="utf-8",
    )
    config = _resolve(home)
    manager = SessionManager(home)
    project = manager.project_registry.create_project("demo", "scope", workspace)
    manager.project_registry.initialize_memory(project.project_id)
    memory_backend = _MemoryScriptBackend()
    notices: list[str] = []

    def backend_builder(provider: str, model: str | None, **kwargs: object):
        del provider, kwargs
        if model == config.memory_model:
            return memory_backend, model
        return _UnusedBackend(), model or "unused"

    composition = compose_runtime(
        home=home,
        cwd=workspace,
        manager=manager,
        config=config,
        provider="fake",
        model="unused-main",
        project_context=ProjectContext("system", ()),
        backend_builder=backend_builder,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        auto_project=False,
        project_id=project.project_id,
        memory_notice=notices.append,
    )
    reconciler = composition.memory_reconciler
    assert reconciler is not None
    try:
        first = composition.opened.store.append_message(
            with_message_origin(Message(MessageRole.USER, [TextContent("durable decision one")]), MessageOrigin.USER)
        )
        await reconciler.drain()

        second = composition.opened.store.append_message(
            with_message_origin(Message(MessageRole.USER, [TextContent("durable decision two")]), MessageOrigin.USER)
        )
        assert composition.loop.context_assembler.on_before_eviction is not None
        composition.loop.context_assembler.on_before_eviction(second.seq, second.seq)
        await reconciler.drain()

        project_id = composition.opened.metadata.project_id
        assert project_id is not None
        snapshot = manager.project_registry.memory_snapshot(project_id)
        assert snapshot.contents["decisions.md"].endswith(
            f"composed range {second.seq}-{second.seq}\n"
        )
        records = manager.project_registry.memory_log(project_id)
        assert [record["provenance"]["seq_end"] for record in records] == [
            first.seq,
            second.seq,
        ]
        assert len(memory_backend.prompts) == 2
        assert notices == [
            "memory updated: decisions.md (+1)",
            "memory updated: decisions.md (+1)",
        ]
    finally:
        await composition.loop.close()


async def test_serve_memory_event_requires_negotiated_feature() -> None:
    client = object.__new__(_Client)
    client._closed = False
    client.features = frozenset({"memory_updated"})
    events: list[tuple[str, str | None, dict[str, object]]] = []

    async def notify(
        event: str, session_id: str | None = None, **fields: object
    ) -> None:
        events.append((event, session_id, fields))

    client._notify = notify
    client._publish_memory_notice("session", "memory updated: decisions.md (+1)")
    await asyncio.sleep(0)
    assert events == [
        (
            "memory_updated",
            "session",
            {"message": "memory updated: decisions.md (+1)"},
        )
    ]

    client.features = frozenset()
    client._publish_memory_notice("session", "hidden")
    await asyncio.sleep(0)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_attention_fork_composition_cannot_write_project_memory(
    tmp_path: Path,
) -> None:
    from zeta.attention_forks import create_discussion_fork
    from zeta.attention_records import AttentionStore

    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    config = _resolve(home)
    manager = SessionManager(home)
    project = manager.project_registry.create_project("demo", "scope", workspace)
    manager.project_registry.initialize_memory(project.project_id)
    original = manager.create(
        provider="fake",
        model="fake",
        cwd=workspace,
        project_id=project.project_id,
        auto_project=False,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    anchor = original.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Choice",
        why="Choose a direction.",
    )
    fork_id = create_discussion_fork(home, original.store.session_id, record.id)
    fork = manager.open(fork_id)

    def backend_builder(provider: str, model: str | None, **kwargs: object):
        del provider, kwargs
        return _UnusedBackend(), model or "fake"

    before = {
        path.relative_to(home / "projects"): path.read_bytes()
        for path in (home / "projects").rglob("*")
        if path.is_file()
    }
    composition = compose_runtime(
        home=home,
        cwd=workspace,
        manager=manager,
        config=config,
        provider="fake",
        model="fake",
        project_context=ProjectContext("system", ()),
        backend_builder=backend_builder,
        opened=fork,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    try:
        assert composition.memory_reconciler is None
        assert composition.opened.store.on_persisted_activity is None
        assert composition.loop.context_assembler.on_before_eviction is None
        assert "inbox" not in composition.loop.tool_registry.registered_names
        composition.opened.store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("fork-only discussion")]),
                MessageOrigin.USER,
            )
        )
        await asyncio.sleep(0)
        after = {
            path.relative_to(home / "projects"): path.read_bytes()
            for path in (home / "projects").rglob("*")
            if path.is_file()
        }
        assert after == before
    finally:
        await composition.loop.close()
        fork.store.close()
        original.store.close()


@pytest.mark.asyncio
async def test_attention_fork_runtime_does_not_enter_project_transcript_index(
    tmp_path: Path,
) -> None:
    from zeta.attention_forks import create_discussion_fork
    from zeta.attention_records import AttentionStore

    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    config = _resolve(home)
    manager = SessionManager(home)
    project = manager.project_registry.create_project("demo", "scope", workspace)
    original = manager.create(
        provider="fake",
        model="fake",
        cwd=workspace,
        project_id=project.project_id,
        auto_project=False,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    anchor = original.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Choice",
        why="Choose a direction.",
    )
    fork_id = create_discussion_fork(home, original.store.session_id, record.id)

    def composition_for(opened: OpenedSession, canary: str) -> RuntimeComposition:
        backend = FakeBackend([ScriptedTurn([TextContent(f"answered {canary}")])])

        def backend_builder(provider: str, model: str | None, **kwargs: object):
            del provider, kwargs
            return backend, model or "fake"

        return compose_runtime(
            home=home,
            cwd=workspace,
            manager=manager,
            config=config,
            provider="fake",
            model="fake",
            project_context=ProjectContext("system", ()),
            backend_builder=backend_builder,
            opened=opened,
            skill_catalog=SkillCatalog.empty(),
            agent_catalog=AgentCatalog.empty(),
        )

    normal = composition_for(original, "normalindexcanary")
    try:
        async for _ in normal.loop.run_turn(
            "normalindexcanary", origin=MessageOrigin.USER
        ):
            pass
        await asyncio.gather(*tuple(normal.loop._tracked_tasks))
    finally:
        await normal.loop.close()

    fork = manager.open(fork_id)
    discussion = composition_for(fork, "forkonlyindexcanary")
    try:
        async for _ in discussion.loop.run_turn(
            "forkonlyindexcanary", origin=MessageOrigin.USER
        ):
            pass
        await asyncio.gather(*tuple(discussion.loop._tracked_tasks))
    finally:
        await discussion.loop.close()

    index = TranscriptIndex(
        manager.project_registry.root / project.project_id, project.project_id
    )
    assert index.search("normalindexcanary")
    assert index.search("forkonlyindexcanary") == ()
