"""Versioned session directories and discovery."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import re
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from rich.cells import cell_len

from .checkpoints import ConversationIntegrityError, load_session_json
from .store import ConversationStore


logger = logging.getLogger(__name__)

META_VERSION = 1


class SessionError(ValueError):
    """Raised when a session cannot be created or resumed."""


@dataclass(frozen=True, slots=True)
class SessionPreview:
    """A session row suitable for the interactive resume picker."""

    session_id: str
    updated_at: str
    preview: str
    name: str = ""


SESSION_NAME_MAX_LENGTH = 60


def normalize_session_name(value: str) -> str:
    """Return a validated session label or raise ``SessionError``."""

    cleaned = _ANSI_SEQUENCE.sub("", value)
    cleaned = "".join(
        character
        for character in cleaned
        if character not in _PREVIEW_STRIPPED_CHARACTERS
        and unicodedata.category(character) != "Cc"
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise SessionError("session name must be a nonempty label")
    if cell_len(cleaned) > SESSION_NAME_MAX_LENGTH:
        raise SessionError(
            f"session name is too long (max {SESSION_NAME_MAX_LENGTH} cells)"
        )
    return cleaned


_ANSI_SEQUENCE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x9b[0-?]*[ -/]*[@-~])"
    r"|(?:\x1b\][^\x07]*(?:\x07|\x1b\\)|\x9d[^\x07]*(?:\x07|\x1b\\))"
    r"|\x1b[ -/]*[@-~]"
)
_PREVIEW_CODEPOINT_LIMIT = 512
_PREVIEW_STRIPPED_CHARACTERS = frozenset(
    chr(codepoint)
    for start, end in ((0x200B, 0x200D), (0x202A, 0x202E), (0x2066, 0x2069))
    for codepoint in range(start, end + 1)
) | {"\ufeff"}


def _preview_text(value: str, *, limit: int = 80) -> str:
    clean = _ANSI_SEQUENCE.sub("", value)
    clean = "".join(
        character
        for character in clean
        if (
            character not in _PREVIEW_STRIPPED_CHARACTERS
            and (character in "\t\n\r" or unicodedata.category(character) != "Cc")
        )
    )
    clean = clean[:_PREVIEW_CODEPOINT_LIMIT]
    clean = " ".join(clean.split())
    if cell_len(clean) <= limit:
        return clean
    suffix = "..."
    available = max(0, limit - cell_len(suffix))
    result = ""
    for character in clean:
        candidate = result + character
        if cell_len(candidate) > available:
            break
        result = candidate
    return result + suffix


def env_home() -> Path:
    """Return zeta's home, honoring the test and local override."""

    return Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def format_relative_age(updated_at: str, *, now: datetime | None = None) -> str:
    """Return a compact human-readable age string like ``2h ago``."""

    try:
        parsed = datetime.fromisoformat(updated_at)
    except ValueError:
        return "unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    delta_seconds = int((reference - parsed).total_seconds())
    if delta_seconds < 5:
        return "just now"
    if delta_seconds < 60:
        return f"{delta_seconds}s ago"
    minutes = delta_seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


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
    override_audit: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    vim_mode: bool = True
    budget_pinned: bool = False
    plan_mode: bool = False
    name: str = ""
    approval_mode: str | None = None

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
        system_prompt: str = "",
        context_files: list[str] | tuple[str, ...] = (),
        vim_mode: bool = True,
        budget_pinned: bool = False,
        plan_mode: bool = False,
        name: str = "",
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
            system_prompt=system_prompt,
            context_files=list(context_files),
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
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
        if any(type(value.get(key)) is not str or not value[key] for key in required_strings):
            raise SessionError(f"session metadata is incomplete: {path}")
        retained_tail = value.get("retained_tail")
        compaction_budget = value.get("compaction_budget")
        if (
            type(retained_tail) is not int
            or retained_tail < 1
            or type(compaction_budget) is not int
            or compaction_budget < 1
        ):
            raise SessionError(f"session metadata budgets are invalid: {path}")
        audit = value.get("override_audit", [])
        if type(audit) is not list or any(type(item) is not dict for item in audit):
            raise SessionError(f"session metadata override audit is invalid: {path}")
        has_context_snapshot = "system_prompt" in value and "context_files" in value
        system_prompt = value.get("system_prompt", "") if has_context_snapshot else ""
        context_files = value.get("context_files", []) if has_context_snapshot else []
        vim_mode = value.get("vim_mode", True)
        budget_pinned = value.get("budget_pinned", False)
        plan_mode = value.get("plan_mode", False)
        name = value.get("name", "")
        if (
            type(system_prompt) is not str
            or type(context_files) is not list
            or any(type(item) is not str for item in context_files)
            or type(vim_mode) is not bool
            or type(budget_pinned) is not bool
            or type(plan_mode) is not bool
            or value.get("approval_mode") not in (None, "ask", "allow", "deny")
            or type(name) is not str
        ):
            raise SessionError(f"session metadata context is invalid: {path}")
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
            override_audit=[dict(item) for item in audit],
            system_prompt=system_prompt,
            context_files=list(context_files),
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
            approval_mode=value.get("approval_mode"),
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
            "override_audit": self.override_audit,
            "system_prompt": self.system_prompt,
            "context_files": self.context_files,
            "vim_mode": self.vim_mode,
            "budget_pinned": self.budget_pinned,
            "plan_mode": self.plan_mode,
            "name": self.name,
            "approval_mode": self.approval_mode,
        }


@dataclass(frozen=True, slots=True)
class OpenedSession:
    metadata: SessionMetadata
    store: ConversationStore


class SessionManager:
    """Create, validate, open, and discover zeta sessions."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home) if home is not None else env_home()
        self.sessions_dir = self.home / "sessions"

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
        vim_mode: bool = True,
        budget_pinned: bool = False,
        name: str = "",
    ) -> OpenedSession:
        resolved_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            session_id = uuid.uuid4().hex
            session_dir = self.sessions_dir / session_id
            if session_dir.exists():
                continue
            metadata = SessionMetadata.new(
                session_id=session_id,
                provider=provider,
                model=model,
                cwd=resolved_cwd,
                retained_tail=retained_tail,
                compaction_budget=compaction_budget,
                system_prompt=system_prompt,
                context_files=context_files,
                vim_mode=vim_mode,
                budget_pinned=budget_pinned,
                name=name,
            )
            # Finish all writes outside discovery before claiming the final ID.
            with TemporaryDirectory(prefix=".session-", dir=self.home) as temporary:
                staged = SessionManager(temporary)
                ConversationStore(
                    staged.sessions_dir,
                    session_id=session_id,
                    cwd=resolved_cwd,
                )
                staged._write(metadata)
                # mkdir atomically claims the ID without replacing any existing
                # path, including a non-cooperating creator's empty directory.
                try:
                    session_dir.mkdir()
                except FileExistsError:
                    continue
                staged_dir = staged.sessions_dir / session_id
                for path in staged_dir.iterdir():
                    if path.name != "meta.json":
                        path.rename(session_dir / path.name)
                # Publish metadata last: a crash before this leaves an incomplete
                # directory that discovery skips and future creators preserve.
                (staged_dir / "meta.json").rename(session_dir / "meta.json")
            return self.open(session_id)
        raise SessionError("could not allocate a unique session id")

    def open(self, session_id: str, *, _read_only: bool = False) -> OpenedSession:
        self._validate_id(session_id)
        metadata = self._read(session_id)
        if metadata.session_id != session_id:
            raise SessionError(
                f"session metadata id mismatch for {session_id}: {metadata.session_id}"
            )
        session_path = self.sessions_dir / session_id
        conversation_path = session_path / "conversation.jsonl"
        if not conversation_path.exists():
            raise SessionError(f"session {session_id} has no conversation.jsonl")
        try:
            store = ConversationStore(
                self.sessions_dir, session_id=session_id, _read_only=_read_only
            )
        except (OSError, ValueError) as exc:
            raise SessionError(f"session {session_id} could not be opened") from exc
        if store.cwd != metadata.cwd:
            raise SessionError(f"session {session_id} cwd does not match its metadata")
        return OpenedSession(metadata, store)

    def list_sessions(self) -> list[SessionMetadata]:
        if not self.sessions_dir.exists():
            return []
        sessions: list[SessionMetadata] = []
        for session_path in self.sessions_dir.iterdir():
            try:
                if not session_path.is_dir():
                    continue
                opened = self.open(session_path.name, _read_only=True)
                sessions.append(opened.metadata)
            except (SessionError, ConversationIntegrityError) as exc:
                logger.warning("Skipping session %s: %s", session_path.name, exc)
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def list_session_previews(self, *, limit: int = 20) -> list[SessionPreview]:
        """Return recent sessions with safe, single-line first-message previews."""

        sessions = self.list_sessions()
        previews: list[SessionPreview] = []
        for metadata in sessions:
            if len(previews) >= limit:
                break
            try:
                opened = self.open(metadata.session_id, _read_only=True)
                first_message = ""
                for entry in opened.store.replay():
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
                        preview=_preview_text(first_message) or "(no user message)",
                        name=metadata.name,
                    )
                )
            except (SessionError, ConversationIntegrityError) as exc:
                logger.warning("Skipping session preview %s: %s", metadata.session_id, exc)
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
        """

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.system_prompt and not overwrite:
                return item
            item.system_prompt = system_prompt
            item.context_files = list(context_files)
            return self._touch(item)

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

    def record_session_settings(self, metadata: SessionMetadata, *, model: str, approval_mode: str, budget: int) -> None:
        """Persist active-session settings together, without changing global config."""
        if approval_mode not in {"ask", "allow", "deny"} or not model.strip() or budget <= 0:
            raise SessionError("invalid session settings")
        expected = (metadata.model, metadata.approval_mode, metadata.compaction_budget, metadata.budget_pinned)

        def update(item: SessionMetadata) -> SessionMetadata:
            if (item.model, item.approval_mode, item.compaction_budget, item.budget_pinned) != expected:
                raise SessionError("session settings changed before commit")
            if item.model != model:
                item.override_audit.append({"at": _now(), "provider": None, "model": {"from": item.model, "to": model}})
            item.model = model
            item.approval_mode = approval_mode
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
                    "session name changed before commit; winner: "
                    f"name={item.name!r}"
                )
            item.name = name
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    def resolve_id(self, session_id: str) -> str:
        """Return the full id for an exact match or unambiguous prefix."""

        self._validate_id(session_id)
        if (self.sessions_dir / session_id).is_dir():
            return session_id
        if not self.sessions_dir.exists():
            raise SessionError(f"session {session_id} was not found")
        matches = sorted(
            entry.name
            for entry in self.sessions_dir.iterdir()
            if entry.is_dir() and entry.name.startswith(session_id)
        )
        if not matches:
            raise SessionError(f"session {session_id} was not found")
        if len(matches) > 1:
            raise SessionError(
                f"session id {session_id!r} is ambiguous "
                f"({len(matches)} matches)"
            )
        return matches[0]

    def delete(self, session_id: str) -> None:
        """Remove a session directory and its contents."""

        full_id = self.resolve_id(session_id)
        session_dir = self.sessions_dir / full_id
        import shutil

        lock_path = session_dir / ".lock"
        if lock_path.exists():
            with lock_path.open("a+") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise SessionError(
                        "session is currently open in another process"
                    ) from exc
                shutil.rmtree(session_dir)
        else:
            shutil.rmtree(session_dir)

    def export(self, session_id: str) -> str:
        """Return the session as portable JSONL (metadata header + entries)."""

        full_id = self.resolve_id(session_id)
        metadata = self._read(full_id)
        conversation_path = self.sessions_dir / full_id / "conversation.jsonl"
        if not conversation_path.exists():
            raise SessionError(f"session {full_id} has no conversation.jsonl")
        header = {"type": "session_export", "metadata": metadata.to_dict()}
        lines = [json.dumps(header, separators=(",", ":"), sort_keys=True)]
        try:
            with conversation_path.open("rb") as handle:
                for line in handle:
                    if line.strip():
                        row = load_session_json(line)
                        lines.append(json.dumps(row, separators=(",", ":")))
        except (ConversationIntegrityError, OSError) as exc:
            raise SessionError(f"session {full_id} could not be exported") from exc
        return "\n".join(lines) + "\n"

    def record_budget(
        self,
        metadata: SessionMetadata,
        *,
        budget: int,
        pinned: bool,
    ) -> None:
        """Persist the compaction budget with optimistic concurrency."""

        expected = metadata.compaction_budget

        def update(item: SessionMetadata) -> SessionMetadata:
            if item.compaction_budget != expected:
                raise SessionError(
                    "session budget changed before commit; winner: "
                    f"compaction_budget={item.compaction_budget!r}"
                )
            item.compaction_budget = budget
            item.budget_pinned = pinned
            return self._touch(item)

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)

    @staticmethod
    def _touch(metadata: SessionMetadata) -> SessionMetadata:
        metadata.updated_at = _now()
        return metadata

    def _mutate(
        self,
        session_id: str,
        update: Callable[[SessionMetadata], SessionMetadata],
    ) -> SessionMetadata:
        with self._metadata_lock(session_id):
            current = self._read(session_id)
            updated = update(current)
            self._write_unlocked(updated)
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
        target.override_audit = [dict(item) for item in source.override_audit]
        target.system_prompt = source.system_prompt
        target.context_files = list(source.context_files)
        target.vim_mode = source.vim_mode
        target.budget_pinned = source.budget_pinned
        target.plan_mode = source.plan_mode
        target.name = source.name
        target.approval_mode = source.approval_mode

    def _read(self, session_id: str) -> SessionMetadata:
        path = self.sessions_dir / session_id / "meta.json"
        try:
            value = load_session_json(path)
        except ConversationIntegrityError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                if path.parent.is_dir():
                    raise SessionError(f"session {session_id} has no meta.json") from exc
                raise SessionError(f"session {session_id} was not found") from exc
            raise SessionError(f"session metadata could not be read: {path}") from exc
        if not isinstance(value, Mapping):
            raise SessionError(f"session metadata is not an object: {path}")
        return SessionMetadata.from_dict(value, path=path)

    def _write(self, metadata: SessionMetadata) -> None:
        with self._metadata_lock(metadata.session_id):
            self._write_unlocked(metadata)

    def _write_unlocked(self, metadata: SessionMetadata) -> None:
        path = self.sessions_dir / metadata.session_id / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w") as handle:
                json.dump(metadata.to_dict(), handle, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _metadata_lock(self, session_id: str):
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with (session_dir / ".meta.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
