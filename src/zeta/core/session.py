"""Versioned session directories and discovery."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import logging
import os
import uuid
import weakref
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

from ..skills import SkillCatalog
from ..storage_format import advance, mapping_mutations, open_writer, release_later
from ..skills.agent_catalog import AgentCatalog
from .checkpoints import ConversationIntegrityError, load_session_json
from .project_context import associate_project_discovery, discover_project
from .store import ConversationStore, TranscriptLoadError
from .transcript_recovery import (
    RepairResult,
    RepairTarget,
    recovery_status,
    set_memory_review_required,
)
from .transcript_recovery.runtime import format_repair_transcript_command
from .store._child_runs import child_run_segments
from .session_execution import (
    RUNTIME_LEASE_NAME, OwnerHint, OwnerSurface,
    SessionExecutionLease,
    SessionOwnedError,
    session_owner_status,
)
from .store.prompt_composition import PromptCompositionMixin
from .session_files import (
    SessionError,
    SessionIdAmbiguousError,
    SessionInUseError,
    SessionNotFoundError,
    SessionReferenceAuthority,
    reference_publication,
    retained_reference_publication,
    open_session_file,
    mutate_session_metadata,
    read_session_file,
    session_directory,
    session_root,
    session_metadata_lock,
    child_directory,
    copy_session_metadata,
    write_session_file,
)
from ..project_registry import ProjectRegistryError
from ..runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)
from .store.session_preferences import SessionPreferenceMixin
from ..transcript_search.index import delete_indexed_session
from ..session_display import (
    SESSION_NAME_MAX_LENGTH,
    format_relative_age,
    normalize_session_name,
    preview_text as _preview_text,
)
from .store.preview import first_user_message
from .session_metadata import META_VERSION, OpenedSession, SessionMetadata
from .session_links import (
    _PROJECT_ROLES,
    associate_project as associate_project_link,
    _persist_pending_root_link,
    project_links_lock,
    reconcile_child_links,
    valid_pending_link,
)

logger = logging.getLogger(__name__)
class SessionLoopThreadLockError(SessionError):
    pass

@dataclass(frozen=True, slots=True)
class SessionPreview:
    """A session row suitable for the interactive resume picker."""

    session_id: str
    updated_at: str
    preview: str
    name: str = ""
    in_use: bool = False
    owner_pid: int | None = None


def env_home() -> Path:
    """Return zeta's home, honoring the test and local override."""

    return Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))


def account_home(home: str | Path | None = None) -> Path:
    """Return the shared provider-account home, or the supplied zeta home."""

    value = os.environ.get("ZETA_ACCOUNT_HOME")
    if value is None:
        return Path(home) if home is not None else env_home()
    path = Path(value).expanduser()
    if not value or not path.is_absolute():
        raise ValueError("ZETA_ACCOUNT_HOME must be an absolute path")
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager(PromptCompositionMixin, SessionPreferenceMixin):
    """Create, validate, open, and discover zeta sessions."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        user_home: str | Path | None = None,
        project_registry_state: ProjectRegistryState | None = None,
        owner_surface: OwnerSurface = "tui",
    ) -> None:
        self.home = Path(home) if home is not None else env_home()
        self._owner_surface = owner_surface
        self._storage_lease = open_writer(self.home)
        self._release_storage_lease = weakref.finalize(
            self, release_later, self._storage_lease.close
        )
        self.user_home = (
            (Path(user_home) if user_home is not None else Path.home())
            .expanduser()
            .resolve()
        )
        self.sessions_dir = self.home / "sessions"
        state = project_registry_state or available_project_registry(
            self.home / "projects"
        )
        self.project_registry_state = state
        self.project_registry = state.registry
        self.project_registry_unavailable = state.unavailable

    @property
    def observed_storage_epoch(self) -> int:
        return self._storage_lease.observed_epoch

    def create(
        self,
        *,
        provider: str,
        model: str,
        cwd: str | Path | None = None,
        retained_tail: int = 8,
        compaction_budget: int = 200_000,
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
        prompt_recipe: str | None = None,
        prompt_components: dict[str, dict[str, int | str]] | None = None,
        tool_allow: tuple[str, ...] | None = None,
        tool_deny: tuple[str, ...] = (),
        tool_allow_layers: tuple[tuple[str, ...], ...] = (),
        auto_project: bool = True,
        _session_id: str | None = None,
        _reference_authority: SessionReferenceAuthority | None = None,
        _reference_stack: ExitStack | None = None,
        _reference_authorities: list[SessionReferenceAuthority] | None = None,
        _before_reference_publication: Callable[[OpenedSession], None] | None = None,
    ) -> OpenedSession:
        resolved_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        if project_role is not None and project_role not in _PROJECT_ROLES:
            raise SessionError("invalid project role")
        if project_id is None and auto_project and self.project_registry is not None:
            try:
                discovery = associate_project_discovery(
                    discover_project(resolved_cwd, user_home=self.user_home),
                    self.project_registry,
                )
                project_id = (
                    discovery.project.project_id
                    if discovery.project is not None
                    else None
                )
            except (ProjectRegistryError, OSError, ValueError) as exc:
                logger.warning(
                    "project discovery unavailable; continuing without project: %s", exc
                )
                project_id = None
        with session_root(self.sessions_dir, create=True):
            pass
        for _ in range(8):
            session_id = _session_id or uuid.uuid4().hex
            metadata = SessionMetadata.new(
                session_id=session_id,
                provider=provider,
                model=model,
                cwd=resolved_cwd,
                retained_tail=retained_tail,
                compaction_budget=compaction_budget,
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
                prompt_recipe=prompt_recipe,
                prompt_components=prompt_components,
                tool_allow=tool_allow,
                tool_deny=tool_deny,
                tool_allow_layers=tool_allow_layers,
            )
            # Finish all writes outside discovery before claiming the final ID.
            with TemporaryDirectory(prefix=".session-", dir=self.home) as temporary:
                advance(temporary, expected_epoch=0)
                staged = SessionManager(temporary)
                ConversationStore(
                    staged.sessions_dir,
                    session_id=session_id,
                    cwd=resolved_cwd,
                    _storage_lease=staged._storage_lease,
                ).close()
                staged._write(metadata)
                # mkdir atomically claims the ID without replacing any existing
                # path, including a non-cooperating creator's empty directory.
                with session_root(self.sessions_dir) as root_fd:
                    try:
                        os.mkdir(session_id, mode=0o700, dir_fd=root_fd)
                    except FileExistsError:
                        continue
                    with child_directory(root_fd, session_id) as destination_fd:
                        execution_lease = SessionExecutionLease.acquire(
                            destination_fd,
                            session_id=session_id,
                            owner=OwnerHint(self._owner_surface),
                            home=self.home,
                        )
                    referenced_ids = (parent_session_id,) if parent_session_id else ()
                    if _reference_authority is not None:
                        execution_authority, publication = (
                            nullcontext(),
                            nullcontext(_reference_authority),
                        )
                    else:
                        execution_authority, publication = retained_reference_publication(
                            self.home,
                            self._storage_lease,
                            _reference_stack,
                            _reference_authorities,
                            installed_ids=(session_id,),
                            referenced_ids=referenced_ids,
                        )
                    try:
                        with execution_authority, publication as authority:
                            assert authority is not None
                            return self._publish_created_session(
                                root_fd=root_fd, staged=staged, metadata=metadata,
                                session_id=session_id,
                                parent_session_id=parent_session_id,
                                project_id=project_id, project_role=project_role,
                                execution_lease=execution_lease, authority=authority,
                                before_reference_publication=_before_reference_publication,
                            )
                    finally:
                        execution_lease.close()
        raise SessionError("could not allocate a unique session id")

    def _publish_created_session(
        self,
        *,
        root_fd: int, staged: SessionManager, metadata: SessionMetadata,
        session_id: str, parent_session_id: str | None,
        project_id: str | None, project_role: str | None,
        execution_lease: SessionExecutionLease,
        authority: SessionReferenceAuthority,
        before_reference_publication: Callable[[OpenedSession], None] | None,
    ) -> OpenedSession:
        authority.validate((session_id, *((parent_session_id,) if parent_session_id else ())))
        with child_directory(root_fd, session_id) as destination_fd, session_directory(
            staged.sessions_dir, session_id
        ) as (_, source_fd):
            for filename in sorted(
                (name for name in os.listdir(source_fd) if name != RUNTIME_LEASE_NAME),
                key=lambda item: item == "meta.json",
            ):
                os.replace(
                    filename,
                    filename,
                    src_dir_fd=source_fd,
                    dst_dir_fd=destination_fd,
                )
            os.fsync(destination_fd)
            os.fsync(root_fd)
        store = ConversationStore(
            self.sessions_dir,
            session_id=session_id,
            _must_exist=True,
            _storage_lease=self._storage_lease,
            _execution_lease=execution_lease,
            _owner_surface=self._owner_surface,
            _initialization_execution_ranked=False,
        )
        metadata._execution_lease = store.execution_lease
        try:
            if before_reference_publication is not None:
                before_reference_publication(OpenedSession(metadata, store))
            store._rank_execution = True
            store.activate_runtime_lease()
            if project_id is not None:
                try:
                    _persist_pending_root_link(
                        self.sessions_dir, session_id, project_id,
                        project_role or "session", parent_session_id,
                        self._storage_lease, authority=authority,
                    )
                except (SessionError, OSError) as exc:
                    logger.warning("could not persist project linkage intent: %s", exc)
                self._reconcile_project_link(session_id, _reference_authority=authority)
            return OpenedSession(metadata, store)
        except BaseException:
            store.close()
            self.delete(session_id)
            raise

    def associate_project(self, metadata: SessionMetadata, project_id: str) -> SessionMetadata:
        return associate_project_link(self, metadata, project_id)

    def read_metadata(self, session_id: str, *, lock_timeout: float | None = None) -> SessionMetadata:
        """Read validated metadata without opening or repairing the conversation."""
        self._validate_id(session_id)
        if lock_timeout is None:
            return self._read(session_id)
        with self._metadata_lock(session_id, timeout=lock_timeout) as directory_fd:
            assert directory_fd is not None
            return self._read(session_id, directory_fd=directory_fd)

    def open(
        self,
        session_id: str,
        *,
        _read_only: bool = False,
        _reference_authority: SessionReferenceAuthority | None = None,
    ) -> OpenedSession:
        self._validate_id(session_id)
        if _reference_authority is not None and not _read_only:
            raise SessionError("an already-gated open must be read-only")
        if _read_only:
            if _reference_authority is not None:
                _reference_authority.validate((session_id,))
            metadata = self.read_metadata(session_id)
            try:
                store = ConversationStore(
                    self.sessions_dir,
                    session_id=session_id,
                    _read_only=True,
                    _must_exist=True,
                    _storage_lease=self._storage_lease,
                    _owner_surface=self._owner_surface,
                )
            except (TranscriptLoadError, OSError, ValueError) as exc:
                raise SessionError(f"session {session_id} could not be opened") from exc
            if store.cwd != metadata.cwd:
                store.close()
                raise SessionError(
                    f"session {session_id} cwd does not match its metadata"
                )
            return OpenedSession(metadata, store)

        existing_metadata = self.read_metadata(session_id)
        with session_root(self.sessions_dir) as root_fd, child_directory(
            root_fd, session_id
        ) as directory_fd:
            execution_lease = SessionExecutionLease.acquire(
                directory_fd,
                session_id=session_id,
                owner=OwnerHint(self._owner_surface),
                home=self.home,
            )
        opened_metadata: SessionMetadata | None = None
        reference_authority: SessionReferenceAuthority | None = None
        @contextmanager
        def initialize_authority():
            nonlocal reference_authority
            with reference_publication(
                self.home,
                self._storage_lease,
                referenced_ids=(session_id,),
            ) as authority:
                reference_authority = authority
                yield authority

        def initialize(store: ConversationStore) -> None:
            nonlocal opened_metadata
            assert reference_authority is not None
            metadata = existing_metadata
            if store.cwd != metadata.cwd:
                raise SessionError(
                    f"session {session_id} cwd does not match its metadata"
                )
            metadata._execution_lease = store.execution_lease
            if metadata.project_id is not None and self.project_registry is not None:
                self._reconcile_project_link(
                    session_id, _reference_authority=reference_authority
                )
                self._reconcile_child_links(
                    session_id, _reference_authority=reference_authority
                )
            opened_metadata = metadata

        try:
            store = ConversationStore(
                self.sessions_dir,
                session_id=session_id,
                _must_exist=True,
                _storage_lease=self._storage_lease,
                _execution_lease=execution_lease,
                _owner_surface=self._owner_surface,
                _initialization_authority=initialize_authority(),
                _on_initialized=initialize,
            )
        except (SessionError, SessionOwnedError, TranscriptLoadError):
            raise
        except (OSError, ValueError) as exc:
            raise SessionError(f"session {session_id} could not be opened") from exc
        finally:
            execution_lease.close()
        assert opened_metadata is not None
        self._complete_root_recovery(opened_metadata, store)
        return OpenedSession(opened_metadata, store)

    def repair_transcript(
        self,
        session_id: str,
        child_run_id: str | None = None,
        *,
        resident_memory_commit: Callable[[str, int, bool], None] | None = None,
    ) -> RepairResult:
        """Repair one diagnosed root or child transcript."""
        metadata = self.read_metadata(session_id)
        target = RepairTarget(session_id, child_run_id)
        segments = child_run_segments(session_id, child_run_id)
        session_dir = self.sessions_dir / session_id
        for segment in segments:
            session_dir = session_dir / "agents" / str(segment)
        store = ConversationStore(
            session_dir.parent,
            session_id=session_dir.name,
            cwd=metadata.cwd,
            _must_exist=True,
            _storage_lease=self._storage_lease,
            _owner_surface="recovery",
            _recovery_owner=True,
            _repair_explicit=True,
            _repair_target=target,
        )
        try:
            result = store.last_repair_result
            if result is None:
                raise SessionError(
                    f"session {session_id} transcript does not need repair"
                )
            if child_run_id is None:
                return self._complete_root_recovery(
                    metadata,
                    store,
                    result,
                    resident_memory_commit=resident_memory_commit,
                )
            return result
        finally:
            store.close()

    def _complete_root_recovery(
        self,
        metadata: SessionMetadata,
        store: ConversationStore,
        result: RepairResult | None = None,
        *,
        resident_memory_commit: Callable[[str, int, bool], None] | None = None,
    ) -> RepairResult | None:
        """Close memory state before a repaired root store is published."""
        status = recovery_status(store.directory_fd, RepairTarget(metadata.session_id))
        record = status.latest_committed
        if record is None or record.memory_step == "resumed":
            return result

        overlaps: tuple[str, ...] = ()
        registry = self.project_registry
        if metadata.project_id is not None and registry is not None:
            overlaps = registry.overlapping_automatic_versions(
                metadata.project_id,
                session_id=metadata.session_id,
                retained_last_sequence=record.cutoff_sequence,
                diagnosed_at=record.diagnosis_timestamp,
            )
            if resident_memory_commit is not None:
                resident_memory_commit(
                    record.quarantine_id,
                    record.cutoff_sequence,
                    bool(overlaps),
                )
            else:
                self._commit_detached_memory_recovery(
                    metadata,
                    store,
                    quarantine_id=record.quarantine_id,
                    retained_last_sequence=record.cutoff_sequence,
                    paused=bool(overlaps),
                )
        set_memory_review_required(
            store.directory_fd, record.quarantine_id, required=bool(overlaps)
        )
        if result is None or result.quarantine_id != record.quarantine_id:
            result = RepairResult(
                record.quarantine_id,
                record.target,
                record.cutoff_sequence,
                record.expected_source_digest,
                record.repaired_digest,
                bool(overlaps),
            )
        return replace(
            result,
            memory_review_required=bool(overlaps),
            overlapping_memory_versions=overlaps,
        )

    def _commit_detached_memory_recovery(
        self,
        metadata: SessionMetadata,
        store: ConversationStore,
        *,
        quarantine_id: str,
        retained_last_sequence: int,
        paused: bool,
    ) -> None:
        from ..memory.reconciliation_state import ReconciliationState

        state = ReconciliationState(
            store.session_dir / "memory-reconcile.json",
            project_id=metadata.project_id,
            session_id=metadata.session_id,
            diagnostics_path=self.home / "logs" / "memory-reconciliation.jsonl",
            storage_lease=self._storage_lease,
        )
        if state.recovery_quarantine_id == quarantine_id:
            return
        state.reset_for_recovery(
            generation=state.generation + 1,
            retained_last_sequence=retained_last_sequence,
            quarantine_id=quarantine_id,
            paused=paused,
        )

    def _reconcile_project_link(
        self,
        session_id: str,
        *,
        _reference_authority: SessionReferenceAuthority | None = None,
    ) -> None:
        publication = (
            nullcontext(_reference_authority)
            if _reference_authority is not None
            else reference_publication(
                self.home,
                self._storage_lease,
                referenced_ids=(session_id,),
            )
        )
        with publication as authority:
            assert authority is not None
            self._reconcile_project_link_gated(session_id, authority)

    def _reconcile_project_link_gated(
        self, session_id: str, authority: SessionReferenceAuthority
    ) -> None:
        registry = self.project_registry
        if registry is None:
            return
        try:
            while True:
                with (
                    session_directory(self.sessions_dir, session_id) as (
                        _,
                        directory_fd,
                    ),
                    project_links_lock(directory_fd),
                ):
                    try:
                        value = json.loads(
                            read_session_file(
                                directory_fd, "project_link_pending.json"
                            )
                        )
                    except (
                        FileNotFoundError,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                        OSError,
                        SessionError,
                    ):
                        value = None
                    if valid_pending_link(value):
                        pending = value
                    else:
                        metadata = self._read(
                            session_id, directory_fd=directory_fd
                        )
                        if metadata.project_id is None:
                            return
                        pending = {
                            "project_id": metadata.project_id,
                            "previous_project_id": None,
                            "role": metadata.project_role or "session",
                            "parent_session_id": metadata.parent_session_id,
                            "transcript_path": str(self.sessions_dir / session_id),
                        }
                        write_session_file(
                            directory_fd,
                            "project_link_pending.json",
                            json.dumps(pending, sort_keys=True).encode("utf-8"),
                        )

                authority.validate(
                    tuple(
                        value
                        for value in (session_id, pending.get("parent_session_id"))
                        if isinstance(value, str)
                    )
                )
                project_id = str(pending["project_id"])
                previous = pending.get("previous_project_id")
                project_ids = {project_id}
                if previous is not None and previous != project_id:
                    project_ids.add(str(previous))
                domains = tuple(
                    (value, "sessions") for value in sorted(project_ids)
                )
                with registry._project_locks(
                    domains, exclusive=True
                ) as project_fds, session_directory(
                    self.sessions_dir, session_id
                ) as (_, directory_fd), project_links_lock(directory_fd):
                    try:
                        current = json.loads(
                            read_session_file(
                                directory_fd, "project_link_pending.json"
                            )
                        )
                    except (
                        FileNotFoundError,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                        OSError,
                        SessionError,
                    ):
                        continue
                    if current != pending or not valid_pending_link(current):
                        continue
                    fd_by_project = {
                        value: project_fd
                        for (value, _domain), project_fd in zip(
                            domains, project_fds, strict=True
                        )
                    }
                    prepared = registry._prepare_session_records(
                        project_id,
                        [
                            {
                                "session_id": session_id,
                                "transcript_path": pending["transcript_path"],
                                "role": pending["role"],
                                "parent_session_id": pending.get(
                                    "parent_session_id"
                                ),
                            }
                        ],
                    )
                    if previous is not None and previous != project_id:
                        registry._update_sessions_locked(
                            fd_by_project[str(previous)], {session_id}, []
                        )
                    registry._update_sessions_locked(
                        fd_by_project[project_id], set(), prepared
                    )
                    os.unlink(
                        "project_link_pending.json", dir_fd=directory_fd
                    )
                    os.fsync(directory_fd)
                    return
        except (
            KeyError,
            ProjectRegistryError,
            OSError,
            SessionError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "project session linkage remains pending for %s (%s): %s",
                session_id,
                type(exc).__name__,
                exc,
            )

    def _reconcile_child_links(
        self,
        session_id: str,
        *,
        _reference_authority: SessionReferenceAuthority | None = None,
    ) -> None:
        if self.project_registry is not None:
            reconcile_child_links(
                self.project_registry,
                self.sessions_dir,
                session_id,
                _reference_authority=_reference_authority,
            )

    def list_sessions(self) -> list[SessionMetadata]:
        sessions = self.list_sessions_read_only()
        for metadata in sessions:
            self._reconcile_project_link(metadata.session_id)
        return sessions

    def list_sessions_read_only(self) -> list[SessionMetadata]:
        """List valid sessions from metadata without opening conversation stores."""
        try:
            with session_root(self.sessions_dir) as root_fd:
                names = os.listdir(root_fd)
        except FileNotFoundError:
            return []
        sessions: list[SessionMetadata] = []
        for name in names:
            try:
                sessions.append(self.read_metadata(name))
            except (SessionError, ConversationIntegrityError) as exc:
                logger.warning("Skipping session %s: %s", name, exc)
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def list_session_previews(
        self, *, limit: int = 20, sessions: list[SessionMetadata] | None = None
    ) -> list[SessionPreview]:
        """Return recent sessions with safe, single-line first-message previews."""

        if limit <= 0:
            return []
        if sessions is None:
            sessions = self.list_sessions()
        previews: list[SessionPreview] = []
        for metadata in sessions[:limit]:
            try:
                first_message = first_user_message(
                    self.sessions_dir, metadata.session_id
                )
            except (OSError, SessionError, ConversationIntegrityError) as exc:
                first_message = ""
                logger.warning(
                    "Empty session preview for %s: %s", metadata.session_id, exc
                )
            try:
                in_use, owner_pid = self.session_owner_status(metadata.session_id)
            except SessionError as exc:
                in_use, owner_pid = False, None
                logger.warning(
                    "Missing owner hint for %s: %s", metadata.session_id, exc
                )
            previews.append(
                SessionPreview(
                    session_id=metadata.session_id,
                    updated_at=metadata.updated_at,
                    preview=_preview_text(first_message),
                    name=metadata.name,
                    in_use=in_use,
                    owner_pid=owner_pid,
                )
            )
        return previews

    def find_most_recent(self, *, cwd: str | Path | None = None) -> SessionMetadata:
        resolved_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        matches = [item for item in self.list_sessions() if item.cwd == resolved_cwd]
        if not matches:
            raise SessionError(f"no prior zeta session found in {resolved_cwd}")
        return matches[0]

    def touch(self, metadata: SessionMetadata) -> None:
        current = self._mutate(
            metadata.session_id,
            lambda item: self._touch(item),
            execution_lease=metadata._execution_lease,
        )
        self._copy_metadata(metadata, current)

    def refresh_recency(
        self,
        session_id: str,
        execution_lease: SessionExecutionLease | None = None,
    ) -> bool:
        """Best-effort refresh recency without waiting for metadata writers."""

        with (
            self._execution_owner(session_id, execution_lease),
            self._metadata_lock(session_id, blocking=False) as directory_fd,
        ):
            if directory_fd is None:
                return False
            current = self._touch(self._read(session_id, directory_fd=directory_fd))
            self._publish_metadata(directory_fd, current, sync=False)
            return True

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

        current = self._mutate(
            metadata.session_id, update, execution_lease=metadata._execution_lease
        )
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

        current = self._mutate(
            metadata.session_id, update, execution_lease=metadata._execution_lease
        )
        self._copy_metadata(metadata, current)

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

        current = self._mutate(
            metadata.session_id, update, execution_lease=metadata._execution_lease
        )
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

        current = self._mutate(
            metadata.session_id, update, execution_lease=metadata._execution_lease
        )
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

        current = self._mutate(
            metadata.session_id, update, execution_lease=metadata._execution_lease
        )
        self._copy_metadata(metadata, current)

    def rename(
        self,
        session_id: str,
        name: str,
        execution_lease: SessionExecutionLease | None = None,
    ) -> SessionMetadata:
        """Set a display name; whitespace clears it to the derived preview."""
        full_id = self.resolve_id(session_id)
        metadata = self.read_metadata(full_id)
        metadata._execution_lease = execution_lease
        self.record_name(
            metadata, name=normalize_session_name(name) if name.strip() else ""
        )
        return metadata

    def resolve_id(
        self, session_id: str, *, min_prefix_length: int | None = None
    ) -> str:
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
        if min_prefix_length is not None and len(session_id) < min_prefix_length:
            raise SessionError(
                f"session id prefix must be at least {min_prefix_length} characters"
            )
        if len(matches) > 1:
            def unique_prefix(match: str) -> str:
                for length in range(8, len(match) + 1):
                    prefix = match[:length]
                    if sum(name.startswith(prefix) for name in candidates) == 1:
                        return prefix
                return match

            candidates = tuple(unique_prefix(match) for match in matches[:8])
            shown = ", ".join(candidates)
            if len(matches) > 8:
                shown += ", …"
            raise SessionIdAmbiguousError(
                f"session id {session_id!r} is ambiguous; matches: {shown}; "
                "please provide more characters",
                candidates,
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
                # Keep deletion available for corrupt sessions; cleanup is best effort.
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
            if project_id is not None and self.project_registry is not None:
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
            try:
                ConversationStore(
                    self.sessions_dir,
                    session_id=full_id,
                    _read_only=True,
                    _must_exist=True,
                ).close()
            except TranscriptLoadError as diagnosis:
                repair_hint = (
                    f"; run {format_repair_transcript_command(full_id)}"
                    if diagnosis.repair_available
                    else ""
                )
                raise SessionError(
                    f"session {full_id} could not be exported: {diagnosis}{repair_hint}"
                ) from exc
            except (ConversationIntegrityError, OSError):
                pass
            raise SessionError(f"session {full_id} could not be exported") from exc
        return "\n".join(lines) + "\n"

    @staticmethod
    def _touch(metadata: SessionMetadata) -> SessionMetadata:
        metadata.updated_at = _now()
        return metadata

    def session_owner_status(self, session_id: str) -> tuple[bool, int | None]:
        with session_directory(self.sessions_dir, session_id) as (_, directory_fd):
            return session_owner_status(directory_fd)

    @contextmanager
    def _execution_owner(
        self,
        session_id: str,
        execution_lease: SessionExecutionLease | None,
    ):
        if execution_lease is not None:
            with execution_lease.serialize_mutation():
                yield execution_lease
            return
        with session_directory(
            self.sessions_dir, session_id, lifetime=True
        ) as (_, directory_fd):
            lease = SessionExecutionLease.retain_resident(directory_fd)
            if lease is None:
                lease = SessionExecutionLease.acquire(
                    directory_fd,
                    session_id=session_id,
                    owner=OwnerHint(self._owner_surface),
                    home=self.home,
                )
            try:
                with lease.serialize_mutation():
                    yield lease
            finally:
                lease.close()

    def _mutate(
        self,
        session_id: str,
        update: Callable[[SessionMetadata], SessionMetadata],
        *,
        lock_timeout: float | None = None,
        execution_lease: SessionExecutionLease | None = None,
        authority=None,
        before_release: Callable[[SessionMetadata], None] | None = None,
    ) -> SessionMetadata:
        return mutate_session_metadata(
            self,
            session_id,
            update,
            lock_timeout=lock_timeout,
            execution_lease=execution_lease,
            authority=authority,
            before_release=before_release,
        )

    _copy_metadata = staticmethod(copy_session_metadata)

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
        with (
            self._execution_owner(metadata.session_id, metadata._execution_lease),
            self._metadata_lock(metadata.session_id) as directory_fd,
        ):
            self._write_unlocked(metadata, directory_fd=directory_fd)

    def _write_unlocked(self, metadata: SessionMetadata, *, directory_fd: int) -> None:
        self._publish_metadata(directory_fd, metadata)

    def _publish_metadata(
        self, directory_fd: int, metadata: SessionMetadata, *, sync: bool = True
    ) -> None:
        value = metadata.to_storage_dict()
        # Refuse a shape the reader would reject, so no writer leaves an
        # unreadable meta.json behind.
        SessionMetadata.from_dict(
            value, path=self.sessions_dir / metadata.session_id / "meta.json"
        )
        delete = () if "tool_allow_layers" in value else ("tool_allow_layers",)
        self._storage_lease.publish_json(
            directory_fd,
            "meta.json",
            mapping_mutations(
                value,
                delete=delete,
                broad=("prompt_components",),
            ),
            lambda fd, name, data: write_session_file(fd, name, data, sync=sync),
        )

    def _metadata_lock(self, session_id: str, **kwargs):
        self._validate_id(session_id)
        return session_metadata_lock(
            self.sessions_dir, session_id, loop_error=SessionLoopThreadLockError,
            blocking=kwargs.pop("blocking", True), timeout=kwargs.pop("timeout", None),
            authority=kwargs.pop("authority", None),
            after_shape=kwargs.pop("after_shape", None),
        )

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


async def persist_metadata_change(
    manager: SessionManager,
    metadata: SessionMetadata,
    change: Callable[[SessionMetadata], None],
) -> None:
    """Run durable metadata work off-loop, then apply its result on-loop."""

    candidate = copy.deepcopy(metadata)
    candidate._execution_lease = metadata._execution_lease
    persistence = asyncio.create_task(asyncio.to_thread(change, candidate))
    try:
        await asyncio.shield(persistence)
    except asyncio.CancelledError:
        await persistence
        manager._copy_metadata(metadata, candidate)
        raise
    manager._copy_metadata(metadata, candidate)


__all__ = [
    "META_VERSION",
    "SESSION_NAME_MAX_LENGTH",
    "OpenedSession",
    "SessionError",
    "SessionInUseError",
    "SessionLoopThreadLockError",
    "SessionManager",
    "SessionMetadata",
    "SessionOwnedError",
    "SessionPreview",
    "env_home",
    "find_most_recent",
    "format_relative_age",
    "list_sessions",
    "normalize_session_name",
    "persist_metadata_change",
]


def list_sessions(home: str | Path | None = None) -> list[SessionMetadata]:
    return SessionManager(home).list_sessions()


def find_most_recent(
    *,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> SessionMetadata:
    return SessionManager(home).find_most_recent(cwd=cwd)
