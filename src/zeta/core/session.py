"""Versioned session directories and discovery."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from ..compaction import (
    COMPACTION_MODES,
    DEFAULT_SESSION_COMPACTION,
    LEGACY_SESSION_COMPACTION,
)
from ..skills import SkillCatalog
from ..skills.agent_catalog import AgentCatalog
from .checkpoints import ConversationIntegrityError, load_session_json
from .project_context import discover_or_find_project
from .store import ConversationStore
from .session_files import (
    SessionError,
    SessionInUseError,
    SessionNotFoundError,
    open_session_file,
    read_session_file,
    session_directory,
    session_root,
    child_directory,
    write_session_json,
)
from ..project_registry import ProjectRegistry, ProjectRegistryError
from .store.session_preferences import SessionPreferenceMixin
from ..config.tool_policy import validate_tool_patterns
from ..transcript_search.index import delete_indexed_session
from ..session_display import (
    SESSION_NAME_MAX_LENGTH,
    format_relative_age,
    normalize_session_name,
    preview_text as _preview_text,
)
from .session_links import (
    _PROJECT_ROLES,
    persist_pending_root_link,
    reconcile_child_links,
    valid_pending_link,
)


logger = logging.getLogger(__name__)


META_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionPreview:
    """A session row suitable for the interactive resume picker."""

    session_id: str
    updated_at: str
    preview: str
    name: str = ""


_MEMORY_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _valid_memory_digest(value: object) -> bool:
    return type(value) is str and _MEMORY_DIGEST.match(value) is not None


def env_home() -> Path:
    """Return zeta's home, honoring the test and local override."""

    return Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class SessionMetadata:
    version: int
    session_id: str
    created_at: str
    updated_at: str
    provider: str
    model: str
    cwd: str
    retained_tail: int
    compaction_budget: int
    compaction: str = DEFAULT_SESSION_COMPACTION
    compaction_pinned: bool = False
    override_audit: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    skill_catalog: list[dict[str, Any]] | None = None
    agent_catalog: list[dict[str, Any]] | None = None
    vim_mode: bool = True
    budget_pinned: bool = False
    plan_mode: bool = False
    name: str = ""
    approval_mode: str | None = None
    # Previous provider, model, and budget until a frontend selection succeeds.
    model_fallback: tuple[str, str, int] | None = None
    project_id: str | None = None
    project_role: str | None = None
    parent_session_id: str | None = None
    # Structured location of the owned project-memory block within
    # ``system_prompt`` (see project_context.refresh_project_memory).  Absent on
    # legacy sessions, which are refreshed by never touching their prompt.
    project_memory_offset: int | None = None
    project_memory_length: int | None = None
    project_memory_digest: str | None = None
    tool_allow: tuple[str, ...] | None = None
    tool_deny: tuple[str, ...] = ()
    tool_allow_layers: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        provider: str,
        model: str,
        cwd: str,
        retained_tail: int,
        compaction_budget: int,
        compaction: str = DEFAULT_SESSION_COMPACTION,
        compaction_pinned: bool = False,
        system_prompt: str = "",
        context_files: list[str] | tuple[str, ...] = (),
        skill_catalog: SkillCatalog | None = None,
        agent_catalog: AgentCatalog | None = None,
        vim_mode: bool = True,
        budget_pinned: bool = False,
        plan_mode: bool = False,
        name: str = "",
        project_id: str | None = None,
        project_role: str | None = None,
        parent_session_id: str | None = None,
        project_memory_offset: int | None = None,
        project_memory_length: int | None = None,
        project_memory_digest: str | None = None,
        tool_allow: tuple[str, ...] | None = None,
        tool_deny: tuple[str, ...] = (),
        tool_allow_layers: tuple[tuple[str, ...], ...] = (),
    ) -> SessionMetadata:
        timestamp = _now()
        return cls(
            version=META_VERSION,
            session_id=session_id,
            created_at=timestamp,
            updated_at=timestamp,
            provider=provider,
            model=model,
            cwd=cwd,
            retained_tail=retained_tail,
            compaction_budget=compaction_budget,
            compaction=compaction,
            compaction_pinned=compaction_pinned,
            system_prompt=system_prompt,
            context_files=list(context_files),
            skill_catalog=skill_catalog.to_snapshot()
            if skill_catalog is not None
            else None,
            agent_catalog=agent_catalog.to_snapshot()
            if agent_catalog is not None
            else None,
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
            project_id=project_id,
            project_role=project_role,
            parent_session_id=parent_session_id,
            project_memory_offset=project_memory_offset,
            project_memory_length=project_memory_length,
            project_memory_digest=project_memory_digest,
            tool_allow=tool_allow,
            tool_deny=tool_deny,
            tool_allow_layers=tool_allow_layers,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: Path) -> SessionMetadata:
        if type(value.get("version")) is not int:
            raise SessionError(f"session metadata version is missing: {path}")
        if value["version"] != META_VERSION:
            raise SessionError(
                f"unsupported session metadata version {value['version']} at {path}; "
                f"expected {META_VERSION}"
            )
        required_strings = (
            "session_id",
            "created_at",
            "updated_at",
            "provider",
            "model",
            "cwd",
        )
        if any(
            type(value.get(key)) is not str or not value[key]
            for key in required_strings
        ):
            raise SessionError(f"session metadata is incomplete: {path}")
        retained_tail = value.get("retained_tail")
        compaction_budget = value.get("compaction_budget")
        # Sessions created before compaction modes were persisted must keep
        # their original summary behavior when they resume.
        compaction = value.get("compaction", LEGACY_SESSION_COMPACTION)
        compaction_pinned = value.get("compaction_pinned", False)
        if (
            type(retained_tail) is not int
            or retained_tail < 1
            or type(compaction_budget) is not int
            or compaction_budget < 1
            or compaction not in COMPACTION_MODES
            or type(compaction_pinned) is not bool
        ):
            raise SessionError(f"session metadata budgets are invalid: {path}")
        audit = value.get("override_audit", [])
        if type(audit) is not list or any(type(item) is not dict for item in audit):
            raise SessionError(f"session metadata override audit is invalid: {path}")
        fallback = value.get("model_fallback")
        if fallback is not None and (
            type(fallback) is not list
            or len(fallback) != 3
            or type(fallback[0]) is not str
            or fallback[0] not in {"claude", "codex"}
            or type(fallback[1]) is not str
            or not fallback[1].strip()
            or type(fallback[2]) is not int
            or fallback[2] <= 0
        ):
            raise SessionError(f"session model fallback is invalid: {path}")
        has_context_snapshot = "system_prompt" in value and "context_files" in value
        system_prompt = value.get("system_prompt", "") if has_context_snapshot else ""
        context_files = value.get("context_files", []) if has_context_snapshot else []
        skill_catalog = value.get("skill_catalog")
        agent_catalog = value.get("agent_catalog")
        vim_mode = value.get("vim_mode", True)
        budget_pinned = value.get("budget_pinned", False)
        plan_mode = value.get("plan_mode", False)
        name = value.get("name", "")
        raw_tool_allow = value.get("tool_allow")
        raw_tool_deny = value.get("tool_deny", [])
        raw_tool_allow_layers = value.get("tool_allow_layers")
        try:
            tool_allow = validate_tool_patterns(raw_tool_allow, field="tool_allow")
            tool_deny = validate_tool_patterns(raw_tool_deny, field="tool_deny")
            tool_allow_layers = (
                tuple(
                    validate_tool_patterns(layer, field="tool_allow_layers") or ()
                    for layer in raw_tool_allow_layers
                )
                if raw_tool_allow_layers is not None
                else (() if tool_allow is None else (tool_allow,))
            )
        except (TypeError, ValueError) as exc:
            raise SessionError(f"session metadata tool policy is invalid: {path}") from exc
        assert tool_deny is not None
        project_id = value.get("project_id")
        project_role = value.get("project_role")
        parent_session_id = value.get("parent_session_id")
        if any(
            item is not None and (type(item) is not str or not item)
            for item in (project_id, project_role, parent_session_id)
        ) or (project_role is not None and project_role not in _PROJECT_ROLES):
            raise SessionError(f"session project linkage is invalid: {path}")
        raw_offset = value.get("project_memory_offset") if has_context_snapshot else None
        raw_length = value.get("project_memory_length") if has_context_snapshot else None
        raw_digest = value.get("project_memory_digest") if has_context_snapshot else None
        # The owned-memory span is all-or-none: offset, length, and digest must
        # arrive together.  A partial triple (e.g. an offset/length with no
        # digest) is treated as a legacy/absent span rather than authorizing an
        # unverifiable replacement, so all three fields drop to None.
        present = (raw_offset is not None, raw_length is not None, raw_digest is not None)
        if not all(present):
            memory_offset = memory_length = memory_digest = None
        else:
            if (
                type(raw_offset) is not int
                or raw_offset < 0
                or type(raw_length) is not int
                or raw_length < 0
                or not _valid_memory_digest(raw_digest)
            ):
                raise SessionError(f"session project memory span is invalid: {path}")
            memory_offset, memory_length, memory_digest = raw_offset, raw_length, raw_digest
        if (
            type(system_prompt) is not str
            or type(context_files) is not list
            or any(type(item) is not str for item in context_files)
            or (
                skill_catalog is not None
                and (
                    type(skill_catalog) is not list
                    or any(type(item) is not dict for item in skill_catalog)
                )
            )
            or (
                agent_catalog is not None
                and (
                    type(agent_catalog) is not list
                    or any(type(item) is not dict for item in agent_catalog)
                )
            )
            or type(vim_mode) is not bool
            or type(budget_pinned) is not bool
            or type(plan_mode) is not bool
            or value.get("approval_mode") not in (None, "ask", "allow", "deny")
            or type(name) is not str
        ):
            raise SessionError(f"session metadata context is invalid: {path}")
        if skill_catalog is not None:
            try:
                SkillCatalog.from_snapshot(skill_catalog)
            except ValueError as exc:
                raise SessionError(
                    f"session metadata skill catalog is invalid: {path}"
                ) from exc
        if agent_catalog is not None:
            try:
                AgentCatalog.from_snapshot(agent_catalog)
            except ValueError as exc:
                raise SessionError(
                    f"session metadata agent catalog is invalid: {path}"
                ) from exc
        return cls(
            version=value["version"],
            session_id=value["session_id"],
            created_at=value["created_at"],
            updated_at=value["updated_at"],
            provider=value["provider"],
            model=value["model"],
            cwd=value["cwd"],
            retained_tail=retained_tail,
            compaction_budget=compaction_budget,
            compaction=compaction,
            compaction_pinned=compaction_pinned,
            override_audit=[dict(item) for item in audit],
            system_prompt=system_prompt,
            context_files=list(context_files),
            skill_catalog=[dict(item) for item in skill_catalog]
            if skill_catalog is not None
            else None,
            agent_catalog=[dict(item) for item in agent_catalog]
            if agent_catalog is not None
            else None,
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
            approval_mode=value.get("approval_mode"),
            model_fallback=tuple(fallback) if fallback is not None else None,
            project_id=project_id,
            project_role=project_role,
            parent_session_id=parent_session_id,
            project_memory_offset=memory_offset,
            project_memory_length=memory_length,
            project_memory_digest=memory_digest,
            tool_allow=tool_allow,
            tool_deny=tool_deny,
            tool_allow_layers=tool_allow_layers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "cwd": self.cwd,
            "retained_tail": self.retained_tail,
            "compaction_budget": self.compaction_budget,
            "compaction": self.compaction,
            "compaction_pinned": self.compaction_pinned,
            "override_audit": self.override_audit,
            "system_prompt": self.system_prompt,
            "context_files": self.context_files,
            "skill_catalog": self.skill_catalog,
            "agent_catalog": self.agent_catalog,
            "vim_mode": self.vim_mode,
            "budget_pinned": self.budget_pinned,
            "plan_mode": self.plan_mode,
            "name": self.name,
            "approval_mode": self.approval_mode,
            "project_id": self.project_id,
            "project_role": self.project_role,
            "parent_session_id": self.parent_session_id,
            "project_memory_offset": self.project_memory_offset,
            "project_memory_length": self.project_memory_length,
            "project_memory_digest": self.project_memory_digest,
            "tool_allow": list(self.tool_allow) if self.tool_allow is not None else None,
            "tool_deny": list(self.tool_deny),
            **(
                {
                    "tool_allow_layers": [
                        list(layer) for layer in self.tool_allow_layers
                    ]
                }
                if len(self.tool_allow_layers) > 1
                else {}
            ),
        }

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "model_fallback": list(self.model_fallback)
            if self.model_fallback
            else None,
        }


@dataclass(frozen=True, slots=True)
class OpenedSession:
    metadata: SessionMetadata
    store: ConversationStore


class SessionManager(SessionPreferenceMixin):
    """Create, validate, open, and discover zeta sessions."""
    def __init__(self, home: str | Path | None = None, *, user_home: str | Path | None = None) -> None:
        self.home = Path(home) if home is not None else env_home()
        self.user_home = (Path(user_home) if user_home is not None else Path.home()).expanduser().resolve()
        self.sessions_dir = self.home / "sessions"
        self.project_registry = ProjectRegistry(self.home / "projects")
    def create(
        self,
        *,
        provider: str,
        model: str,
        cwd: str | Path | None = None,
        retained_tail: int = 8,
        compaction_budget: int = 200_000,
        compaction: str = DEFAULT_SESSION_COMPACTION,
        compaction_pinned: bool = False,
        system_prompt: str = "",
        context_files: list[str] | tuple[str, ...] = (),
        skill_catalog: SkillCatalog | None = None,
        agent_catalog: AgentCatalog | None = None,
        vim_mode: bool = True,
        budget_pinned: bool = False,
        name: str = "",
        project_id: str | None = None,
        project_role: str | None = None,
        parent_session_id: str | None = None,
        project_memory_offset: int | None = None,
        project_memory_length: int | None = None,
        project_memory_digest: str | None = None,
        tool_allow: tuple[str, ...] | None = None,
        tool_deny: tuple[str, ...] = (),
        tool_allow_layers: tuple[tuple[str, ...], ...] = (),
        auto_project: bool = True,
    ) -> OpenedSession:
        resolved_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        if project_role is not None and project_role not in _PROJECT_ROLES:
            raise SessionError("invalid project role")
        if project_id is None and auto_project:
            try:
                project = discover_or_find_project(
                    self.project_registry, resolved_cwd, self.user_home
                )
                project_id = project.project_id if project is not None else None
            except (ProjectRegistryError, OSError, ValueError) as exc:
                logger.warning("project discovery unavailable; continuing without project: %s", exc)
                project_id = None
        with session_root(self.sessions_dir, create=True):
            pass
        for _ in range(8):
            session_id = uuid.uuid4().hex
            metadata = SessionMetadata.new(
                session_id=session_id,
                provider=provider,
                model=model,
                cwd=resolved_cwd,
                retained_tail=retained_tail,
                compaction_budget=compaction_budget,
                compaction=compaction,
                compaction_pinned=compaction_pinned,
                system_prompt=system_prompt,
                context_files=context_files,
                skill_catalog=skill_catalog,
                agent_catalog=agent_catalog,
                vim_mode=vim_mode,
                budget_pinned=budget_pinned,
                name=name,
                project_id=project_id,
                project_role=project_role,
                parent_session_id=parent_session_id,
                project_memory_offset=project_memory_offset,
                project_memory_length=project_memory_length,
                project_memory_digest=project_memory_digest,
                tool_allow=tool_allow,
                tool_deny=tool_deny,
                tool_allow_layers=tool_allow_layers,
            )
            # Finish all writes outside discovery before claiming the final ID.
            with TemporaryDirectory(prefix=".session-", dir=self.home) as temporary:
                staged = SessionManager(temporary)
                ConversationStore(
                    staged.sessions_dir,
                    session_id=session_id,
                    cwd=resolved_cwd,
                ).close()
                staged._write(metadata)
                # mkdir atomically claims the ID without replacing any existing
                # path, including a non-cooperating creator's empty directory.
                with session_root(self.sessions_dir) as root_fd:
                    try:
                        os.mkdir(session_id, mode=0o700, dir_fd=root_fd)
                    except FileExistsError:
                        continue
                    with (
                        child_directory(root_fd, session_id) as destination_fd,
                        session_directory(staged.sessions_dir, session_id) as (
                            _,
                            source_fd,
                        ),
                    ):
                        names = os.listdir(source_fd)
                        # Publish metadata last so discovery skips incomplete sessions.
                        for filename in sorted(
                            names, key=lambda item: item == "meta.json"
                        ):
                            os.replace(
                                filename,
                                filename,
                                src_dir_fd=source_fd,
                                dst_dir_fd=destination_fd,
                            )
            if project_id is not None:
                link = {
                    "project_id": project_id,
                    "role": project_role or "session",
                    "parent_session_id": parent_session_id,
                    "transcript_path": str(self.sessions_dir / session_id),
                }
                # The intent is the durable source of truth.  Publish it before
                # touching the registry, so a crash at either boundary is
                # recoverable after restart.
                try:
                    with session_directory(self.sessions_dir, session_id) as (
                        _,
                        directory_fd,
                    ):
                        write_session_json(
                            directory_fd, "project_link_pending.json", link
                        )
                        # Durably land the intent's directory entry before the
                        # registry is published; write_session_json only fsyncs
                        # the file and its rename, not the containing directory.
                        os.fsync(directory_fd)
                except (SessionError, OSError) as exc:
                    logger.warning("could not persist project linkage intent: %s", exc)
                self._reconcile_project_link(session_id)
            opened = self.open(session_id)
            return opened
        raise SessionError("could not allocate a unique session id")
    def associate_project(self, metadata: SessionMetadata, project_id: str) -> SessionMetadata:
        """Associate an existing session with a project."""
        previous_project_id = metadata.project_id
        self.project_registry.show_project(project_id)
        bind = lambda item: setattr(item, "project_id", project_id) or item
        current = self._mutate(metadata.session_id, bind)
        self._copy_metadata(metadata, current)
        try:
            persist_pending_root_link(
                self.sessions_dir, current.session_id, project_id, current.project_role or "session", current.parent_session_id,
            )
        except (SessionError, OSError) as exc:
            logger.warning("could not persist project linkage intent: %s", exc)
        self._reconcile_project_link(current.session_id)
        if previous_project_id is not None and previous_project_id != project_id:
            delete_indexed_session(
                self.project_registry.root, previous_project_id, current.session_id
            )
        return current

    def read_metadata(self, session_id: str) -> SessionMetadata:
        """Read validated metadata without opening or repairing the conversation."""
        self._validate_id(session_id)
        return self._read(session_id)

    def open(self, session_id: str, *, _read_only: bool = False) -> OpenedSession:
        metadata = self.read_metadata(session_id)
        if not _read_only and metadata.project_id is not None:
            # Reconcile durable linkage intent before publishing a live store;
            # this also repairs links after a process restart.
            self._reconcile_project_link(session_id)
            # Child/grandchild lineage lives inside the root's agent subtree and
            # is never visited as a top-level session; publish any durable child
            # intents idempotently on every open.
            self._reconcile_child_links(session_id)
        try:
            store = ConversationStore(
                self.sessions_dir,
                session_id=session_id,
                _read_only=_read_only,
                _must_exist=True,
            )
        except (OSError, ValueError) as exc:
            raise SessionError(f"session {session_id} could not be opened") from exc
        if store.cwd != metadata.cwd:
            store.close()
            raise SessionError(f"session {session_id} cwd does not match its metadata")
        return OpenedSession(metadata, store)

    def _reconcile_project_link(self, session_id: str) -> None:
        try:
            with session_directory(self.sessions_dir, session_id) as (_, directory_fd):
                pending: dict[str, object] | None = None
                try:
                    value = json.loads(
                        read_session_file(directory_fd, "project_link_pending.json")
                    )
                    if valid_pending_link(value):
                        pending = value
                except (
                    FileNotFoundError,
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    OSError,
                    SessionError,
                ):
                    pass
                # Immutable meta.json is the second source of truth.  This
                # closes the gap where intent publication itself was torn,
                # unwritable, or left semantically incomplete (e.g. ``{}``) but
                # the session directory was durable.
                if pending is None:
                    metadata = self._read(session_id)
                    if metadata.project_id is None:
                        return
                    pending = {
                        "project_id": metadata.project_id,
                        "role": metadata.project_role or "session",
                        "parent_session_id": metadata.parent_session_id,
                        "transcript_path": str(self.sessions_dir / session_id),
                    }
                self.project_registry.record_session(
                    pending["project_id"],
                    session_id=session_id,
                    transcript_path=pending["transcript_path"],
                    role=pending["role"],
                    parent_session_id=pending.get("parent_session_id"),
                )
                try:
                    os.unlink("project_link_pending.json", dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except FileNotFoundError:
                    pass
        except (
            KeyError,
            ProjectRegistryError,
            OSError,
            SessionError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "project session linkage remains pending for %s: %s", session_id, exc
            )

    def _reconcile_child_links(self, session_id: str) -> None:
        reconcile_child_links(self.project_registry, self.sessions_dir, session_id)

    def list_sessions(self) -> list[SessionMetadata]:
        sessions = self.list_sessions_read_only()
        for metadata in sessions:
            self._reconcile_project_link(metadata.session_id)
        return sessions

    def list_sessions_read_only(self) -> list[SessionMetadata]:
        """List valid sessions without repairing or changing durable state."""
        try:
            with session_root(self.sessions_dir) as root_fd:
                names = os.listdir(root_fd)
        except FileNotFoundError:
            return []
        sessions: list[SessionMetadata] = []
        for name in names:
            try:
                opened = self.open(name, _read_only=True)
                opened.store.close()
                sessions.append(opened.metadata)
            except (SessionError, ConversationIntegrityError) as exc:
                logger.warning("Skipping session %s: %s", name, exc)
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def list_session_previews(
        self, *, limit: int = 20, sessions: list[SessionMetadata] | None = None
    ) -> list[SessionPreview]:
        """Return recent sessions with safe, single-line first-message previews."""

        if sessions is None:
            sessions = self.list_sessions()
        previews: list[SessionPreview] = []
        for metadata in sessions:
            if len(previews) >= limit:
                break
            try:
                opened = self.open(metadata.session_id, _read_only=True)
                with opened.store:
                    entries = opened.store.replay()
                first_message = ""
                for entry in entries:
                    if entry.type != "message":
                        continue
                    message = entry.data.get("message")
                    if not isinstance(message, dict):
                        continue
                    if message.get("role") != "user":
                        continue
                    parts = []
                    for block in message.get("content", []):
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") in {
                            "text",
                            "thinking",
                        } and isinstance(block.get("text"), str):
                            parts.append(block["text"])
                    first_message = "".join(parts)
                    break
                previews.append(
                    SessionPreview(
                        session_id=metadata.session_id,
                        updated_at=metadata.updated_at,
                        preview=_preview_text(first_message),
                        name=metadata.name,
                    )
                )
            except (SessionError, ConversationIntegrityError) as exc:
                logger.warning(
                    "Skipping session preview %s: %s", metadata.session_id, exc
                )
        return previews

    def find_most_recent(self, *, cwd: str | Path | None = None) -> SessionMetadata:
        resolved_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        matches = [item for item in self.list_sessions() if item.cwd == resolved_cwd]
        if not matches:
            raise SessionError(f"no prior zeta session found in {resolved_cwd}")
        return matches[0]

    def touch(self, metadata: SessionMetadata) -> None:
        current = self._mutate(metadata.session_id, lambda item: self._touch(item))
        self._copy_metadata(metadata, current)

    def persist_context_snapshot(
        self,
        metadata: SessionMetadata,
        *,
        system_prompt: str,
        context_files: list[str] | tuple[str, ...],
        overwrite: bool = False,
    ) -> SessionMetadata:
        """Snapshot the composed system prompt for future resumes.

        By default this is first-write-wins: once a session has a stored
        system_prompt, subsequent calls no-op so plain resume replays the
        same cached prefix. Pass ``overwrite=True`` on the explicit
        resume-with-``--system-prompt``/``--append-system-prompt`` path
        so the new prompt replaces the snapshot; the caller is
        responsible for warning the user that the prompt cache rebuilds.

        ``updated_at`` only advances on ``overwrite=True`` (explicit user
        action). Automatic legacy hydration — filling the snapshot in
        first-write-wins style during resume — must not bump ``updated_at``,
        or the sidebar reorders a session the user did not touch.
        """

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.system_prompt and not overwrite:
                return item
            item.system_prompt = system_prompt
            item.context_files = list(context_files)
            if overwrite:
                return self._touch(item)
            return item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current

    def persist_tool_policy(
        self,
        metadata: SessionMetadata,
        *,
        tool_allow: tuple[str, ...] | None,
        tool_deny: tuple[str, ...],
        tool_allow_layers: tuple[tuple[str, ...], ...],
    ) -> SessionMetadata:
        """Persist a narrowed effective policy without changing session recency."""

        def update(item: SessionMetadata) -> SessionMetadata:
            item.tool_allow = tool_allow
            item.tool_deny = tool_deny
            item.tool_allow_layers = tool_allow_layers
            return item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current

    def persist_skill_catalog(
        self,
        metadata: SessionMetadata,
        catalog: SkillCatalog,
        *,
        system_prompt: str | None = None,
        context_files: list[str] | tuple[str, ...] | None = None,
    ) -> SessionMetadata:
        """Snapshot the session catalog once so resumes do not rediscover it."""

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.skill_catalog is not None:
                return item
            item.skill_catalog = catalog.to_snapshot()
            if system_prompt is not None:
                item.system_prompt = system_prompt
            if context_files is not None:
                item.context_files = list(context_files)
            return item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current

    def persist_agent_catalog(
        self, metadata: SessionMetadata, catalog: AgentCatalog
    ) -> SessionMetadata:
        """Snapshot the session agent catalog so resumes do not rediscover it."""

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.agent_catalog is not None:
                return item
            item.agent_catalog = catalog.to_snapshot()
            return item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current

    def record_override(
        self,
        metadata: SessionMetadata,
        *,
        provider: str | None,
        model: str | None,
    ) -> None:
        expected_provider = metadata.provider
        expected_model = metadata.model
        expected_audit = copy.deepcopy(metadata.override_audit)

        def update(item: SessionMetadata) -> SessionMetadata:
            if (
                item.provider != expected_provider
                or item.model != expected_model
                or item.override_audit != expected_audit
            ):
                raise SessionError(
                    "session override changed before commit; winner: "
                    f"provider={item.provider!r}, model={item.model!r}, "
                    f"audit={item.override_audit[-1] if item.override_audit else None}"
                )
            item.override_audit.append(
                {
                    "at": _now(),
                    "provider": {"from": item.provider, "to": provider}
                    if provider is not None and provider != item.provider
                    else None,
                    "model": {"from": item.model, "to": model}
                    if model is not None and model != item.model
                    else None,
                }
            )
            if provider is not None:
                item.provider = provider
            if model is not None:
                item.model = model
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    def record_session_settings(
        self,
        metadata: SessionMetadata,
        *,
        model: str,
        approval_mode: str,
        budget: int,
        provider: str,
        model_fallback: tuple[str, str, int] | None = None,
    ) -> None:
        """Persist active-session settings together, without changing global config."""
        if (
            approval_mode not in {"ask", "allow", "deny"}
            or not model.strip()
            or budget <= 0
        ):
            raise SessionError("invalid session settings")
        expected = (
            metadata.provider,
            metadata.model,
            metadata.approval_mode,
            metadata.compaction_budget,
            metadata.budget_pinned,
            metadata.model_fallback,
        )

        def update(item: SessionMetadata) -> SessionMetadata:
            if (
                item.provider,
                item.model,
                item.approval_mode,
                item.compaction_budget,
                item.budget_pinned,
                item.model_fallback,
            ) != expected:
                raise SessionError("session settings changed before commit")
            if item.model != model or item.provider != provider:
                item.override_audit.append(
                    {
                        "at": _now(),
                        "provider": {"from": item.provider, "to": provider}
                        if item.provider != provider
                        else None,
                        "model": {"from": item.model, "to": model}
                        if item.model != model
                        else None,
                    }
                )
            item.provider = provider
            item.model = model
            item.approval_mode = approval_mode
            item.model_fallback = model_fallback
            item.compaction_budget = budget
            return self._touch(item)

        self._copy_metadata(metadata, self._mutate(metadata.session_id, update))

    def record_vim_mode(self, metadata: SessionMetadata, *, enabled: bool) -> None:
        """Persist the composer editing mode with optimistic concurrency."""

        expected = metadata.vim_mode

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.vim_mode != expected:
                raise SessionError(
                    "session vim mode changed before commit; winner: "
                    f"vim_mode={item.vim_mode!r}"
                )
            item.vim_mode = enabled
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    def record_plan_mode(self, metadata: SessionMetadata, *, enabled: bool) -> None:
        """Persist plan mode with optimistic concurrency."""

        expected = metadata.plan_mode

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.plan_mode != expected:
                raise SessionError(
                    "session plan mode changed before commit; winner: "
                    f"plan_mode={item.plan_mode!r}"
                )
            item.plan_mode = enabled
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    def record_name(self, metadata: SessionMetadata, *, name: str) -> None:
        """Persist a session label with optimistic concurrency."""

        expected = metadata.name

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.name != expected:
                raise SessionError(
                    f"session name changed before commit; winner: name={item.name!r}"
                )
            item.name = name
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    def rename(self, session_id: str, name: str) -> SessionMetadata:
        """Set a display name; whitespace clears it to the derived preview."""
        full_id = self.resolve_id(session_id)
        metadata = self.read_metadata(full_id)
        self.record_name(
            metadata, name=normalize_session_name(name) if name.strip() else ""
        )
        return metadata

    def resolve_id(self, session_id: str) -> str:
        """Return the full id for an exact match or unambiguous prefix."""

        self._validate_id(session_id)
        try:
            with (
                session_root(self.sessions_dir) as root_fd,
                os.scandir(root_fd) as entries,
            ):
                candidates = [
                    entry.name
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False) or entry.is_symlink()
                ]
            if session_id in candidates:
                return session_id
            matches = sorted(name for name in candidates if name.startswith(session_id))
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"session {session_id} was not found") from exc
        except OSError as exc:
            raise SessionError(
                f"session {session_id} could not be resolved: {exc.strerror}"
            ) from exc
        if not matches:
            raise SessionNotFoundError(f"session {session_id} was not found")
        if len(matches) > 1:
            raise SessionError(
                f"session id {session_id!r} is ambiguous ({len(matches)} matches)"
            )
        return matches[0]

    def delete(self, session_id: str) -> None:
        """Remove a session directory and its contents."""

        import shutil

        full_id = session_id
        project_id: str | None = None
        try:
            full_id = self.resolve_id(session_id)
            try:
                project_id = self._read(full_id).project_id
            except SessionError:
                # Deletion intentionally remains available for incomplete or
                # corrupt sessions; index cleanup is best effort.
                pass
            with session_directory(self.sessions_dir, full_id, exclusive=True) as (
                root_fd,
                session_fd,
            ):
                lock_fd = open_session_file(session_fd, ".lock", os.O_RDWR | os.O_CREAT)
                try:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise SessionInUseError(
                            "session is currently open or in use"
                        ) from exc
                    # fd-based rmtree unlinks nested symlinks without following them.
                    shutil.rmtree(full_id, dir_fd=root_fd)
                finally:
                    os.close(lock_fd)
            if project_id is not None:
                delete_indexed_session(self.project_registry.root, project_id, full_id)
        except SessionInUseError:
            raise
        except (OSError, SessionError) as exc:
            raise SessionError(
                f"session {full_id} could not be deleted: {exc}"
            ) from exc

    def export(self, session_id: str) -> str:
        """Return the session as portable JSONL (metadata header + entries)."""

        full_id = self.resolve_id(session_id)
        metadata = self._read(full_id)
        header = {"type": "session_export", "metadata": metadata.to_dict()}
        lines = [json.dumps(header, separators=(",", ":"), sort_keys=True)]
        try:
            with (
                session_directory(self.sessions_dir, full_id) as (_, directory_fd),
                os.fdopen(
                    open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY),
                    "rb",
                ) as handle,
            ):
                for line in handle:
                    if line.strip():
                        row = load_session_json(line)
                        lines.append(json.dumps(row, separators=(",", ":")))
        except (ConversationIntegrityError, OSError) as exc:
            raise SessionError(f"session {full_id} could not be exported") from exc
        return "\n".join(lines) + "\n"

    @staticmethod
    def _touch(metadata: SessionMetadata) -> SessionMetadata:
        metadata.updated_at = _now()
        return metadata

    def _mutate(
        self,
        session_id: str,
        update: Callable[[SessionMetadata], SessionMetadata],
    ) -> SessionMetadata:
        with self._metadata_lock(session_id) as directory_fd:
            current = self._read(session_id, directory_fd=directory_fd)
            updated = update(current)
            self._write_unlocked(updated, directory_fd=directory_fd)
            return updated

    @staticmethod
    def _copy_metadata(target: SessionMetadata, source: SessionMetadata) -> None:
        target.version = source.version
        target.session_id = source.session_id
        target.created_at = source.created_at
        target.updated_at = source.updated_at
        target.provider = source.provider
        target.model = source.model
        target.cwd = source.cwd
        target.retained_tail = source.retained_tail
        target.compaction_budget = source.compaction_budget
        target.compaction = source.compaction
        target.compaction_pinned = source.compaction_pinned
        target.override_audit = [dict(item) for item in source.override_audit]
        target.system_prompt = source.system_prompt
        target.context_files = list(source.context_files)
        target.skill_catalog = (
            [dict(item) for item in source.skill_catalog]
            if source.skill_catalog is not None
            else None
        )
        target.agent_catalog = (
            [dict(item) for item in source.agent_catalog]
            if source.agent_catalog is not None
            else None
        )
        target.vim_mode = source.vim_mode
        target.budget_pinned = source.budget_pinned
        target.plan_mode = source.plan_mode
        target.name = source.name
        target.approval_mode = source.approval_mode
        target.model_fallback = source.model_fallback
        target.tool_allow = source.tool_allow
        target.tool_deny = source.tool_deny
        target.tool_allow_layers = source.tool_allow_layers

    def _read(
        self, session_id: str, *, directory_fd: int | None = None
    ) -> SessionMetadata:
        if directory_fd is None:
            with session_directory(self.sessions_dir, session_id) as (_, opened_fd):
                return self._read(session_id, directory_fd=opened_fd)
        path = self.sessions_dir / session_id / "meta.json"
        try:
            with os.fdopen(
                open_session_file(directory_fd, "meta.json", os.O_RDONLY), "rb"
            ) as handle:
                value = load_session_json(handle.read())
        except FileNotFoundError as exc:
            raise SessionError(f"session {session_id} has no meta.json") from exc
        except (ConversationIntegrityError, OSError) as exc:
            raise SessionError(f"session metadata could not be read: {path}") from exc
        if not isinstance(value, Mapping):
            raise SessionError(f"session metadata is not an object: {path}")
        metadata = SessionMetadata.from_dict(value, path=path)
        if metadata.session_id != session_id:
            raise SessionError(
                f"session metadata id mismatch for {session_id}: {metadata.session_id}"
            )
        return metadata

    def _write(self, metadata: SessionMetadata) -> None:
        with self._metadata_lock(metadata.session_id) as directory_fd:
            self._write_unlocked(metadata, directory_fd=directory_fd)

    def _write_unlocked(self, metadata: SessionMetadata, *, directory_fd: int) -> None:
        write_session_json(directory_fd, "meta.json", metadata.to_storage_dict())

    @contextmanager
    def _metadata_lock(self, session_id: str):
        self._validate_id(session_id)
        with session_directory(self.sessions_dir, session_id) as (_, directory_fd):
            try:
                lock_fd = open_session_file(
                    directory_fd, ".meta.lock", os.O_RDWR | os.O_CREAT
                )
            except OSError as exc:
                raise SessionError("session metadata lock could not be opened") from exc
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield directory_fd
            finally:
                os.close(lock_fd)

    @staticmethod
    def _validate_id(session_id: str) -> None:
        path = Path(session_id)
        if (
            not session_id
            or session_id in {".", ".."}
            or "\x00" in session_id
            or path.is_absolute()
            or path.parts != (session_id,)
        ):
            raise SessionError(f"invalid session id: {session_id!r}")


__all__ = [
    "META_VERSION",
    "SESSION_NAME_MAX_LENGTH",
    "OpenedSession",
    "SessionError",
    "SessionInUseError",
    "SessionManager",
    "SessionMetadata",
    "SessionPreview",
    "env_home",
    "find_most_recent",
    "format_relative_age",
    "list_sessions",
    "normalize_session_name",
]


def list_sessions(home: str | Path | None = None) -> list[SessionMetadata]:
    return SessionManager(home).list_sessions()


def find_most_recent(
    *,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> SessionMetadata:
    return SessionManager(home).find_most_recent(cwd=cwd)
