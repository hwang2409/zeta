from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

import zeta
from zeta.attention_forks import (
    ATTENTION_FORK_POLICY,
    create_discussion_fork,
    read_attention_fork,
    release_discussion_fork,
    validate_attention_fork,
)
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies, ReplyConflict
from zeta.config.tool_policy import ToolPolicy
from zeta.core.project_context import ProjectContext, load_project_context
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.runtime.project_registry_state import ProjectRegistryState
from zeta.server.runtime import ServerRuntime
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools.registry import ToolDefinition, ToolRegistry

load_project_context = partial(
    load_project_context, registry=ProjectRegistryState(None)
)


def _session(home: Path, *, project_id: str | None = None):
    manager = SessionManager(home)
    return manager.create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )


def test_request_attention_writes_atomic_record(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    entry = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    registry = ToolRegistry(
        tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
    )
    definition = registry.definitions_by_name["request_attention"]
    assert isinstance(definition, ToolDefinition)
    assert definition.description == (
        "Use once per topic to put important user attention in the inbox, then continue work. "
        "Set kind to decision only when the user must choose; include options and a recommendation when useful. "
        "Set kind to answer for a direct answer to an earlier user question, decided for an important decision you made for the user, or deferred for an important issue you left for the user. "
        "Write why as self-contained context, assuming the user has read nothing since their last message. "
        "Do not use this for routine progress, and do not repeat an open item."
    )
    identity = Path(zeta.__file__).parent / "prompts" / "identity.md"
    assert "request_attention" not in identity.read_text(encoding="utf-8")

    result = asyncio.run(
        definition.handler(
            {
                "title": "Choose database",
                "why": "The implementation needs one durable database.",
                "options": ["SQLite", "Postgres"],
                "recommendation": "SQLite",
            }
        )
    )

    records = list((opened.store.session_dir / "attention").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert result["content"][0]["text"].startswith("Attention requested:")
    assert record["entry_id"] == entry.id
    assert record["entry_seq"] == entry.seq
    assert record["status"] == "open"
    assert not list(records[0].parent.glob("*.tmp"))
    asyncio.run(registry.close())
    opened.store.close()


def test_request_attention_is_not_available_to_children(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    registry = ToolRegistry(
        tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
    )
    child = registry.clone_for_session(
        opened.store, exclude_names={"request_attention"}
    )
    assert "request_attention" in registry.registered_names
    assert "request_attention" not in child.registered_names
    asyncio.run(child.close())
    asyncio.run(registry.close())
    opened.store.close()


def test_attention_fork_policy_is_action_aware_and_read_only() -> None:
    assert isinstance(ATTENTION_FORK_POLICY, ToolPolicy)
    assert ATTENTION_FORK_POLICY.allows("read")
    assert ATTENTION_FORK_POLICY.allows("resolve_attention")
    assert not ATTENTION_FORK_POLICY.allows("edit")
    assert not ATTENTION_FORK_POLICY.allows("bash")
    assert not ATTENTION_FORK_POLICY.allows("agent")
    assert not ATTENTION_FORK_POLICY.allows("run_background")


def _tool_result(call: ToolCall, content: str) -> Message:
    return Message(
        MessageRole.TOOL_RESULT,
        [],
        tool_result=ToolResult(call.id, content),
    )


def _anthropic_payload(messages: list[Message]) -> dict:
    return build_messages_payload(
        messages,
        [],
        model="claude-sonnet-4-5",
        max_tokens=10_000,
    )


def _anthropic_tool_ids(messages: list[Message]) -> tuple[set[str], set[str]]:
    payload = _anthropic_payload(messages)
    uses = {
        block["id"]
        for message in payload["messages"]
        for block in message["content"]
        if block["type"] == "tool_use"
    }
    results = {
        block["tool_use_id"]
        for message in payload["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    }
    return uses, results


def test_parallel_attention_fork_copies_the_complete_tool_batch(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    calls = [
        ToolCall(f"attention-{index}", "request_attention", {"title": str(index)})
        for index in range(4)
    ]
    anchor = opened.store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(call) for call in calls],
        )
    )
    results = [
        opened.store.append_message(_tool_result(call, str(index)))
        for index, call in enumerate(calls)
    ]
    attention_store = AttentionStore(opened.store.session_dir, writer_epoch=5)
    attention = [
        attention_store.request(
            session_id=opened.store.session_id,
            project_id=None,
            entry_id=entry.id,
            entry_seq=entry.seq,
            title=f"Choice {index}",
            why=f"Created by parallel request {index}.",
            options=("keep it", "change it"),
            recommendation="keep it",
        )
        for index, entry in enumerate([anchor, *results[:-1]])
    ][1]

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    try:
        branch = fork.store.replay()
        assert [entry.id for entry in branch[:5]] == [
            anchor.id,
            *(result.id for result in results),
        ]
        assert (
            Message.from_dict(branch[5].data["message"]).metadata["kind"]
            == "attention_fork"
        )
        messages = fork.store.messages()
        assert _anthropic_tool_ids(messages) == (
            {call.id for call in calls},
            {call.id for call in calls},
        )
        payload_text = json.dumps(_anthropic_payload(messages))
        assert "Created by parallel request 1." in payload_text
        assert "keep it" in payload_text
        assert "change it" in payload_text
    finally:
        fork.store.close()
        opened.store.close()


def test_attention_fork_cuts_before_an_incomplete_tool_batch(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    before = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("before")])
    )
    calls = [
        ToolCall(f"attention-{index}", "request_attention", {"title": str(index)})
        for index in range(2)
    ]
    opened.store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls])
    )
    first_result = opened.store.append_message(_tool_result(calls[0], "created"))
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=first_result.id,
        entry_seq=first_result.seq,
        title="First choice",
        why="The sibling result is not durable yet.",
        options=("wait", "continue"),
        recommendation="wait",
    )

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    try:
        assert [entry.id for entry in fork.store.replay()[:1]] == [before.id]
        messages = fork.store.messages()
        assert _anthropic_tool_ids(messages) == (set(), set())
        payload_text = json.dumps(_anthropic_payload(messages))
        assert "The sibling result is not durable yet." in payload_text
        assert "wait" in payload_text
        assert "continue" in payload_text
    finally:
        fork.store.close()
        opened.store.close()


def test_single_attention_call_fork_cut_is_unchanged(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    call = ToolCall("attention-one", "request_attention", {"title": "one"})
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )
    result = opened.store.append_message(_tool_result(call, "created"))
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=result.id,
        entry_seq=result.seq,
        title="One choice",
        why="One request.",
    )

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    try:
        assert [entry.id for entry in fork.store.replay()[:2]] == [anchor.id, result.id]
        assert _anthropic_tool_ids(fork.store.messages()) == ({call.id}, {call.id})
    finally:
        fork.store.close()
        opened.store.close()


def test_reused_fork_with_dangling_parallel_calls_is_safe_on_reopen(
    tmp_path: Path,
) -> None:
    opened = _session(tmp_path)
    calls = [
        ToolCall(f"attention-{index}", "request_attention", {"title": str(index)})
        for index in range(4)
    ]
    opened.store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls])
    )
    opened.store.append_message(_tool_result(calls[0], "created"))
    opened.store.append_message(
        Message(
            MessageRole.SYSTEM,
            [TextContent("discussion fork")],
            metadata={"origin": "harness", "kind": "attention_fork"},
        )
    )
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("help me")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    fork_id = opened.store.session_id
    opened.store.close()

    source = _session(tmp_path)
    anchor = source.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("choose")])
    )
    attention_store = AttentionStore(source.store.session_dir, writer_epoch=5)
    attention = attention_store.request(
        session_id=source.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Existing choice",
        why="Reuse the existing discussion.",
    )
    attention_store.replace(replace(attention, fork_session_id=fork_id))
    assert (
        create_discussion_fork(tmp_path, source.store.session_id, attention.id)
        == fork_id
    )
    source.store.close()

    reopened = SessionManager(tmp_path).open(fork_id, _read_only=True)
    try:
        messages = reopened.store.messages()
        assert _anthropic_tool_ids(messages) == (set(), set())
        payload = build_messages_payload(
            messages,
            [],
            model="claude-sonnet-4-5",
            max_tokens=10_000,
        )
        assert any(
            block.get("text") == "help me"
            for message in payload["messages"]
            for block in message["content"]
        )
    finally:
        reopened.store.close()


def test_fork_copies_active_branch_through_anchor_without_modifying_source(
    tmp_path: Path,
) -> None:
    opened = _session(tmp_path)
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("one")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("two")])
    )
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("three")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    original = {
        p.name: p.read_bytes()
        for p in opened.store.session_dir.iterdir()
        if p.is_file()
    }

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)

    assert {
        p.name: p.read_bytes()
        for p in opened.store.session_dir.iterdir()
        if p.is_file()
    } == original
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    assert [
        entry.data["message"]["content"][0]["text"] for entry in fork.store.replay()[:2]
    ] == ["one", "two"]
    assert len(fork.store.replay()) == 3  # harness note follows copied branch
    fork_metadata = read_attention_fork(fork.store.session_dir)
    assert fork_metadata is not None
    assert fork_metadata.forked_from_session == opened.store.session_id
    assert fork_metadata.forked_at_entry == anchor.id
    assert fork_metadata.attention_id == attention.id
    assert fork.metadata.tool_allow == (
        "read",
        "fetch",
        "websearch",
        "recall_history",
        "project",
        "mcp_discover",
        "resolve_attention",
    )
    fork.store.close()
    opened.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe", ["default", "custom"])
async def test_attention_fork_preserves_prompt_composition_on_resume(
    tmp_path: Path, recipe: str
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home.mkdir()
    (home / "AGENTS.md").write_text("identity before", encoding="utf-8")
    skills = SkillCatalog.empty()
    agents = AgentCatalog.empty()
    if recipe == "default":
        context = load_project_context(
            cwd=workspace,
            repo_root=workspace,
            zeta_home=home,
            catalog=skills,
        )
    else:
        prompt = "CUSTOM-FORK-PROMPT"
        context = ProjectContext(
            prompt,
            (),
            has_override=True,
            prompt_recipe="custom",
            prompt_components={
                "custom": {
                    "offset": 0,
                    "length": len(prompt),
                    "digest": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            },
        )
    manager = SessionManager(home)
    source = await asyncio.to_thread(lambda: manager.create(
        provider="codex",
        model="offline",
        cwd=workspace,
        system_prompt=context.system_prompt,
        context_files=context.files,
        skill_catalog=skills,
        agent_catalog=agents,
        project_memory_offset=context.memory_offset,
        project_memory_length=context.memory_length,
        project_memory_digest=context.memory_digest,
        prompt_recipe=context.prompt_recipe,
        prompt_components=context.prompt_components,
        auto_project=False,
    ))
    anchor = source.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("choose")])
    )
    attention = AttentionStore(source.store.session_dir, writer_epoch=5).request(
        session_id=source.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    fork_id = await asyncio.to_thread(lambda: create_discussion_fork(home, source.store.session_id, attention.id))
    source.store.close()
    fork = manager.open(fork_id, _read_only=True)
    try:
        assert fork.metadata.prompt_recipe == recipe
        assert fork.metadata.prompt_components == context.prompt_components
    finally:
        fork.store.close()

    (home / "AGENTS.md").write_text("identity after", encoding="utf-8")
    runtime = ServerRuntime(home, cwd=tmp_path, provider="codex")
    await runtime.resume_session(fork_id)
    resumed = runtime.loop.context_assembler.system_prompt.content[0].text
    await runtime.close()

    if recipe == "default":
        assert "identity after" in resumed
        assert "identity before" not in resumed
    else:
        assert resumed == "CUSTOM-FORK-PROMPT"


def test_concurrent_fork_creators_reuse_one_usable_session(
    tmp_path: Path, monkeypatch
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    opened = _session(tmp_path, project_id=project.project_id)
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    sessions_before = {path.name for path in manager.sessions_dir.iterdir()}
    original_create = SessionManager.create

    def delayed_create(self, *args, **kwargs):
        time.sleep(0.1)
        return original_create(self, *args, **kwargs)

    monkeypatch.setattr(SessionManager, "create", delayed_create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fork_ids = tuple(
            pool.map(
                lambda _: create_discussion_fork(
                    tmp_path, opened.store.session_id, attention.id
                ),
                range(2),
            )
        )

    assert fork_ids[0] == fork_ids[1]
    assert {path.name for path in manager.sessions_dir.iterdir()} == sessions_before | {
        fork_ids[0]
    }
    assert {
        path.name for path in (opened.store.session_dir / "attention").glob("*.lock")
    } == {".fork-allocation.lock"}
    fork = manager.open(fork_ids[0], _read_only=True)
    validated = validate_attention_fork(
        home=tmp_path,
        current_session_id=fork_ids[0],
        current_project_id=project.project_id,
        directory_fd=fork.store.directory_fd,
    )
    assert validated.record.fork_session_id == fork_ids[0]
    fork.store.close()
    opened.store.close()


def test_failed_fork_binding_removes_created_session(
    tmp_path: Path, monkeypatch
) -> None:
    opened = _session(tmp_path)
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    manager = SessionManager(tmp_path)
    sessions_before = {path.name for path in manager.sessions_dir.iterdir()}

    def fail_replace(self, record, *, create_directory=False):
        raise OSError("binding publication failed")

    monkeypatch.setattr(AttentionStore, "replace", fail_replace)
    with pytest.raises(OSError, match="binding publication failed"):
        create_discussion_fork(tmp_path, opened.store.session_id, attention.id)

    assert {path.name for path in manager.sessions_dir.iterdir()} == sessions_before
    opened.store.close()


def _create_gated_discussion_fork(
    home: str,
    source_session_id: str,
    attention_id: str,
    publication_started: object,
    publication_release: object,
    result: object,
) -> None:
    original = SessionManager._reconcile_project_link

    def gated_publication(manager: SessionManager, session_id: str) -> None:
        publication_started.set()  # type: ignore[attr-defined]
        if not publication_release.wait(10):  # type: ignore[attr-defined]
            raise AssertionError("fork project publication was not released")
        original(manager, session_id)

    SessionManager._reconcile_project_link = gated_publication
    try:
        fork_id = create_discussion_fork(
            Path(home), source_session_id, attention_id
        )
    except BaseException as exc:  # noqa: BLE001 - reported to parent process
        result.put(("rejected", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
    else:
        result.put(("created", fork_id))  # type: ignore[attr-defined]


def _mutate_fork_source(
    home: str, source_session_id: str, mutation: str, result: object
) -> None:
    try:
        manager = SessionManager(Path(home))
        if mutation == "delete":
            manager.delete(source_session_id)
        else:
            source = manager.open(source_session_id)
            try:
                source.store.append_message(
                    Message(MessageRole.ASSISTANT, [TextContent("Keep going")])
                )
            finally:
                source.store.close()
    except BaseException as exc:  # noqa: BLE001 - reported to parent process
        result.put(("failed", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
    else:
        result.put(("mutated",))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("mutation", "refusal"),
    [
        ("delete", "not found"),
        # Only the exact post-publication revalidation can reject this case:
        # the attention record is still present and the bind would succeed.
        ("append", "source changed during fork allocation"),
    ],
)
def test_p06_8_discussion_fork_allocation_process_progress(
    tmp_path: Path, mutation: str, refusal: str
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    opened = _session(tmp_path, project_id=project.project_id)
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    attention = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    source_session_id = opened.store.session_id
    opened.store.close()

    context = multiprocessing.get_context("spawn")
    publication_started = context.Event()
    publication_release = context.Event()
    fork_result = context.Queue()
    mutation_result = context.Queue()
    creator = context.Process(
        target=_create_gated_discussion_fork,
        args=(
            str(tmp_path),
            source_session_id,
            attention.id,
            publication_started,
            publication_release,
            fork_result,
        ),
    )
    mutator = context.Process(
        target=_mutate_fork_source,
        args=(str(tmp_path), source_session_id, mutation, mutation_result),
    )
    creator.start()
    try:
        assert publication_started.wait(10), "fork project publication did not start"
        mutator.start()
        assert mutation_result.get(timeout=10) == ("mutated",), (
            "public opposing mutation did not complete before fork project publication"
        )
        publication_release.set()
        creator.join(10)
        mutator.join(10)
        assert not creator.is_alive(), "fork creator did not finish"
        assert not mutator.is_alive(), "source mutator did not finish"
        assert creator.exitcode == 0
        assert mutator.exitcode == 0

        outcome = fork_result.get(timeout=10)
        assert outcome[0] == "rejected", "stale discussion fork was bound"
        assert refusal in outcome[2]
        remaining = {path.name for path in manager.sessions_dir.iterdir()}
        assert remaining == (set() if mutation == "delete" else {source_session_id}), (
            "created fork was not cleaned"
        )
    finally:
        publication_release.set()
        for process in (creator, mutator):
            if process.pid is not None and process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(10)
                process.close()
        for result in (fork_result, mutation_result):
            result.close()
            result.join_thread()


def _open_project_attention(home: Path, project_id: str) -> tuple[str, str]:
    opened = _session(home, project_id=project_id)
    try:
        anchor = opened.store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("Choose")])
        )
        record = AttentionStore(opened.store.session_dir, writer_epoch=5).request(
            session_id=opened.store.session_id,
            project_id=project_id,
            entry_id=anchor.id,
            entry_seq=anchor.seq,
            title="Question",
            why="Need a choice.",
        )
        return opened.store.session_id, record.id
    finally:
        opened.store.close()


def _inbox_ids(manager: SessionManager, project_id: str) -> set[str]:
    state = ProjectInbox(
        manager.project_registry, sessions_root=manager.sessions_dir
    ).read(project_id)
    return {item["id"] for status in ("new", "claimed", "done") for item in state[status]}


def _resolution_lock_held(attention_dir: Path) -> bool:
    fd = os.open(attention_dir / ".resolution.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


@pytest.mark.parametrize("opponent", ["competing_resolver", "fork_release"])
def test_p06_6_attention_resolution_commits_message_and_record_or_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opponent: str
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    session_id, attention_id = _open_project_attention(tmp_path, project.project_id)
    fork_id = create_discussion_fork(tmp_path, session_id, attention_id)
    store = AttentionStore(manager.sessions_dir / session_id, writer_epoch=5)
    send_started = threading.Event()
    send_go = threading.Event()
    original_send = ProjectInbox.send

    def gated_send(self, *args, **kwargs):
        if threading.current_thread().name.startswith("deliver"):
            send_started.set()
            assert send_go.wait(10), "deliverer was not released"
        return original_send(self, *args, **kwargs)

    def compete() -> object:
        replies = AttentionReplies(tmp_path, session_id)
        if opponent == "competing_resolver":
            try:
                return replies.resolve_local(attention_id, "chat answer")
            except ReplyConflict as exc:
                return exc
        return release_discussion_fork(tmp_path, fork_id)

    # test-contract: seam hold the public reply inside its inbox send so the
    # opponent runs while the message is being published.
    monkeypatch.setattr(ProjectInbox, "send", gated_send)
    with (
        ThreadPoolExecutor(1, thread_name_prefix="deliver") as deliverer,
        ThreadPoolExecutor(1, thread_name_prefix="opponent") as competitor,
    ):
        try:
            delivered = deliverer.submit(
                AttentionReplies(tmp_path, session_id).reply,
                attention_id,
                "popup answer",
                fork_id,
            )
            assert send_started.wait(10), "deliverer did not reach its inbox send"
            competed = competitor.submit(compete)
            assert _resolution_lock_held(store.directory)
        finally:
            send_go.set()
        outcome = delivered.result(10)
        competed.result(10)

    final = store.get(attention_id)
    delivered_message = outcome.message_id in _inbox_ids(manager, project.project_id)
    assert delivered_message == (
        final.status == "resolved" and final.decision == "popup answer"
    ), "attention resolution committed only one of message and record"
    assert [item.status for item in outcome.items] == ["applied"]


def _reply_in_process(
    home: str,
    session_id: str,
    attention_id: str,
    reply: str,
    send_started: object | None,
    send_release: object | None,
    result: object,
) -> None:
    try:
        if send_started is not None:
            original_publish = ProjectInbox._publish_outgoing

            def gated_publish(self, *args, **kwargs):
                send_started.set()  # type: ignore[attr-defined]
                if not send_release.wait(10):  # type: ignore[attr-defined]
                    raise TimeoutError("holder was not released")
                return original_publish(self, *args, **kwargs)

            ProjectInbox._publish_outgoing = gated_publish  # type: ignore[method-assign]
        outcome = AttentionReplies(Path(home), session_id).reply(
            attention_id, reply, session_id
        )
    except BaseException as exc:  # noqa: BLE001 - reported to parent process
        # AttentionReplies wraps operation errors; report the original type.
        cause = exc.__cause__ or exc
        result.put(("refused", type(cause).__name__, str(exc)))  # type: ignore[attr-defined]
    else:
        result.put(("delivered", outcome.items[0].status))  # type: ignore[attr-defined]


def test_p06_8_attention_resolution_process_progress(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    held_session, held_attention = _open_project_attention(tmp_path, project.project_id)
    other_session, other_attention = _open_project_attention(
        tmp_path, project.project_id
    )

    context = multiprocessing.get_context("spawn")
    send_started = context.Event()
    send_release = context.Event()
    holder_result = context.Queue()
    opponent_result = context.Queue()
    holder = context.Process(
        target=_reply_in_process,
        args=(
            str(tmp_path),
            held_session,
            held_attention,
            "holder answer",
            send_started,
            send_release,
            holder_result,
        ),
    )
    opponent = context.Process(
        target=_reply_in_process,
        args=(
            str(tmp_path),
            other_session,
            other_attention,
            "opponent answer",
            None,
            None,
            opponent_result,
        ),
    )
    holder.start()
    try:
        assert send_started.wait(10), "holder did not reach its inbox publication"
        opponent.start()
        outcome = opponent_result.get(timeout=10)
        assert not send_release.is_set()
        assert outcome[0] in {"refused", "delivered"}, (
            "opposing attention resolution made no progress while the holder held"
        )
        if outcome[0] == "refused":
            assert outcome[1] in {"InboxBusyError", "ProjectRegistryBusyError"}, outcome
            assert AttentionStore(manager.sessions_dir / other_session, writer_epoch=5).get(
                other_attention
            ).status == "open"
        send_release.set()
        assert holder_result.get(timeout=10) == ("delivered", "applied")
        holder.join(10)
        opponent.join(10)
        assert holder.exitcode == 0
        assert opponent.exitcode == 0

        published = _inbox_ids(manager, project.project_id)
        retried = AttentionReplies(tmp_path, other_session).reply(
            other_attention, "opponent answer", other_session
        )
        assert (retried.message_id in published) is (outcome[0] == "delivered"), (
            "a refused attention resolution left its message behind"
        )
        # A refused send leaves its sealed journal; the retry's recovery
        # finishes it first, so the item is already resolved either way.
        assert [item.status for item in retried.items] == ["already-resolved"]
        assert retried.message_id in _inbox_ids(manager, project.project_id)
        assert AttentionStore(manager.sessions_dir / other_session, writer_epoch=5).get(
            other_attention
        ).decision == "opponent answer"
    finally:
        send_release.set()
        for process in (holder, opponent):
            if process.pid is not None and process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(10)
                process.close()
        for queue in (holder_result, opponent_result):
            queue.close()
            queue.join_thread()
