"""Creation and validation of isolated attention discussion forks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .attention_records import (
    AttentionRecord,
    AttentionStore,
    read_bounded_session_json,
)
from .config.tool_policy import ToolPolicy
from .core.session import SessionManager
from .core.session_files import session_directory, write_session_file
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
            manager = SessionManager(Path(session_dir).parent.parent)
            with session_directory(manager.sessions_dir, Path(session_dir).name) as (
                _,
                fd,
            ):
                value = read_bounded_session_json(fd, "attention_fork.json")
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


def create_discussion_fork(
    home: Path, source_session_id: str, attention_id: str
) -> str:
    """Create or reuse the read-only discussion fork for one attention record."""
    manager = SessionManager(home)
    source = manager.open(source_session_id, _read_only=True)
    try:
        attention_store = AttentionStore(source.store.session_dir)
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
            parent_session_id=source_session_id,
            tool_allow=ATTENTION_FORK_POLICY.allow,
            auto_project=False,
        )
        fork_id = fork.store.session_id
        with session_directory(manager.sessions_dir, fork_id) as (_, fork_fd):
            write_session_file(
                fork_fd,
                "attention_fork.json",
                (
                    json.dumps(
                        {
                            "forked_from_session": source_session_id,
                            "forked_at_entry": record.entry_id,
                            "attention_id": record.id,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode(),
            )
        fork.store.close()
        log_path = fork.store.session_dir / "conversation.jsonl"
        header = log_path.read_bytes().splitlines(keepends=True)[0]
        selected = branch[: anchor_index + 1]
        log_path.write_bytes(
            header
            + b"".join(
                (
                    json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
                for entry in selected
            )
        )
        reopened = manager.open(fork_id)
        note = (
            f"You are a discussion fork for attention item {record.id}: {record.title}. "
            "The original session keeps running. Read, search, and discuss only. "
            "When the user reaches a decision, send it with resolve_attention."
        )
        reopened.store.append_message(
            Message(
                MessageRole.SYSTEM,
                [TextContent(note)],
                metadata={"origin": "harness", "kind": "attention_fork"},
            )
        )
        reopened.store.close()
        attention_store.replace(
            AttentionRecord.from_dict({**record.to_dict(), "fork_session_id": fork_id})
        )
        return fork_id
    finally:
        source.store.close()
