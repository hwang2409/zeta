"""Creation and validation of isolated attention discussion forks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .attention_records import (
    AttentionRecord,
    AttentionStore,
    read_bounded_session_json,
)
from .config.tool_policy import ToolPolicy
from .core.checkpoints import ConversationEntry
from .core.session import OpenedSession, SessionManager
from .core.session_files import (
    child_directory,
    open_session_file,
    session_directory,
    write_session_file,
)
from .protocol.types import Message, MessageRole, TextContent
from .skills import SkillCatalog
from .skills.agent_catalog import AgentCatalog

ATTENTION_FORK_POLICY = ToolPolicy.create(
    (
        "read",
        "fetch",
        "websearch",
        "recall_history",
        "project",
        "mcp_discover",
        "resolve_attention",
    )
)


@dataclass(frozen=True, slots=True)
class AttentionFork:
    forked_from_session: str
    forked_at_entry: str
    attention_id: str


@dataclass(frozen=True, slots=True)
class ValidatedAttentionFork:
    metadata: AttentionFork
    record: AttentionRecord


def read_attention_fork(
    session_dir: Path, *, directory_fd: int | None = None
) -> AttentionFork | None:
    try:
        if directory_fd is None:
            fd = os.open(session_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                value = read_bounded_session_json(fd, "attention_fork.json")
            finally:
                os.close(fd)
        else:
            value = read_bounded_session_json(directory_fd, "attention_fork.json")
    except FileNotFoundError:
        return None
    if not isinstance(value, dict) or set(value) != {
        "forked_from_session",
        "forked_at_entry",
        "attention_id",
    }:
        raise ValueError("invalid attention fork metadata")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError("invalid attention fork metadata")
    return AttentionFork(**value)


def validate_attention_fork(
    *,
    home: Path,
    current_session_id: str,
    current_project_id: str | None,
    directory_fd: int,
) -> ValidatedAttentionFork:
    """Bind fork metadata to its source record, anchor, project, and current session."""
    manager = SessionManager(home)
    current_dir = manager.sessions_dir / current_session_id
    fork = read_attention_fork(current_dir, directory_fd=directory_fd)
    if fork is None or current_project_id is None:
        raise ValueError("resolve_attention is available only in an attention fork")
    source_metadata = manager.read_metadata(fork.forked_from_session)
    record = AttentionStore(manager.sessions_dir / fork.forked_from_session).get(
        fork.attention_id
    )
    if (
        record.session_id != fork.forked_from_session
        or record.entry_id != fork.forked_at_entry
        or record.fork_session_id != current_session_id
        or record.project_id != current_project_id
        or source_metadata.project_id != current_project_id
    ):
        raise ValueError("attention fork is not bound to this source record")
    return ValidatedAttentionFork(fork, record)


def attention_decision_message_id(attention_id: str) -> str:
    """Return the stable inbox identity for one attention decision."""
    return hashlib.sha256(f"attention-decision:{attention_id}".encode()).hexdigest()[
        :32
    ]


@contextmanager
def _fork_allocation_lock(
    attention_store: AttentionStore, attention_id: str
) -> Iterator[None]:
    if len(attention_id) != 32 or any(
        character not in "0123456789abcdef" for character in attention_id
    ):
        raise ValueError("invalid attention id")
    with (
        session_directory(
            attention_store.session_dir.parent, attention_store.session_dir.name
        ) as (_, session_fd),
        child_directory(session_fd, "attention") as attention_fd,
    ):
        fcntl.flock(attention_fd, fcntl.LOCK_EX)
        try:
            lock_fd = open_session_file(
                attention_fd, f".{attention_id}.lock", os.O_RDWR | os.O_CREAT
            )
        finally:
            fcntl.flock(attention_fd, fcntl.LOCK_UN)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def _create_fork_session(
    *,
    manager: SessionManager,
    source: OpenedSession,
    record: AttentionRecord,
    branch: list[ConversationEntry],
    anchor: int,
) -> str:
    metadata = source.metadata
    fork = manager.create(
        provider=metadata.provider,
        model=metadata.model,
        cwd=metadata.cwd,
        retained_tail=metadata.retained_tail,
        compaction_budget=metadata.compaction_budget,
        compaction=metadata.compaction,
        compaction_pinned=metadata.compaction_pinned,
        system_prompt=metadata.system_prompt,
        context_files=metadata.context_files,
        skill_catalog=(
            SkillCatalog.from_snapshot(metadata.skill_catalog)
            if metadata.skill_catalog is not None
            else None
        ),
        agent_catalog=(
            AgentCatalog.from_snapshot(metadata.agent_catalog)
            if metadata.agent_catalog is not None
            else None
        ),
        vim_mode=metadata.vim_mode,
        budget_pinned=metadata.budget_pinned,
        name=f"Discussion: {record.title}",
        project_id=metadata.project_id,
        project_role="session",
        parent_session_id=record.session_id,
        tool_allow=ATTENTION_FORK_POLICY.allow,
        auto_project=False,
    )
    fork_id = fork.store.session_id
    try:
        try:
            with session_directory(manager.sessions_dir, fork_id) as (_, fork_fd):
                write_session_file(
                    fork_fd,
                    "attention_fork.json",
                    (
                        json.dumps(
                            {
                                "forked_from_session": record.session_id,
                                "forked_at_entry": record.entry_id,
                                "attention_id": record.id,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode(),
                )
        finally:
            fork.store.close()
        log_path = fork.store.session_dir / "conversation.jsonl"
        header = log_path.read_bytes().splitlines(keepends=True)[0]
        log_path.write_bytes(
            header
            + b"".join(
                (
                    json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
                for entry in branch[: anchor + 1]
            )
        )
        reopened = manager.open(fork_id)
        try:
            note = (
                f"You are a discussion fork for attention item {record.id}: "
                f"{record.title}. The original session keeps running. Read, search, "
                "and discuss only. When the user reaches a decision, send it with "
                "resolve_attention."
            )
            reopened.store.append_message(
                Message(
                    MessageRole.SYSTEM,
                    [TextContent(note)],
                    metadata={"origin": "harness", "kind": "attention_fork"},
                )
            )
        finally:
            reopened.store.close()
    except BaseException:
        manager.delete(fork_id)
        raise
    return fork_id


def create_discussion_fork(
    home: Path, source_session_id: str, attention_id: str
) -> str:
    """Create or reuse the read-only discussion fork for one attention record."""
    manager = SessionManager(home)
    source = manager.open(source_session_id, _read_only=True)
    try:
        attention_store = AttentionStore(source.store.session_dir)
        with _fork_allocation_lock(attention_store, attention_id):
            record = attention_store.get(attention_id)
            if record.session_id != source_session_id:
                raise ValueError("attention record does not belong to the source session")
            if record.project_id != source.metadata.project_id:
                raise ValueError("attention record does not belong to the source project")
            if record.fork_session_id:
                return record.fork_session_id
            branch = source.store.replay()
            anchor_index = next(
                (
                    index
                    for index, entry in enumerate(branch)
                    if entry.id == record.entry_id
                ),
                None,
            )
            if anchor_index is None:
                raise ValueError("attention anchor is not on the active branch")
            fork_id = _create_fork_session(
                manager=manager,
                source=source,
                record=record,
                branch=branch,
                anchor=anchor_index,
            )
            bound_record = AttentionRecord.from_dict(
                {**record.to_dict(), "fork_session_id": fork_id}
            )
            try:
                attention_store.replace(bound_record)
            except BaseException:
                try:
                    publication_succeeded = (
                        attention_store.get(attention_id).fork_session_id == fork_id
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    publication_succeeded = False
                if publication_succeeded:
                    return fork_id
                manager.delete(fork_id)
                raise
            return fork_id
    finally:
        source.store.close()
