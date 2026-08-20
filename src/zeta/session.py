"""Versioned session directories and discovery."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .store import ConversationStore


META_VERSION = 1


class SessionError(ValueError):
    """Raised when a session cannot be created or resumed."""


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
    override_audit: list[dict[str, Any]] = field(default_factory=list)

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
        compaction_budget: int = 100_000,
    ) -> OpenedSession:
        session_id = uuid.uuid4().hex
        resolved_cwd = str(cwd or Path.cwd())
        metadata = SessionMetadata.new(
            session_id=session_id,
            provider=provider,
            model=model,
            cwd=resolved_cwd,
            retained_tail=retained_tail,
            compaction_budget=compaction_budget,
        )
        store = ConversationStore(
            self.sessions_dir,
            session_id=session_id,
            cwd=resolved_cwd,
        )
        self._write(metadata)
        return OpenedSession(metadata, store)

    def open(self, session_id: str) -> OpenedSession:
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
            store = ConversationStore(self.sessions_dir, session_id=session_id)
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
            if not session_path.is_dir():
                continue
            self._validate_id(session_path.name)
            sessions.append(self._read(session_path.name))
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def find_most_recent(self, *, cwd: str | Path | None = None) -> SessionMetadata:
        resolved_cwd = str(cwd or Path.cwd())
        matches = [item for item in self.list_sessions() if item.cwd == resolved_cwd]
        if not matches:
            raise SessionError(f"no prior zeta session found in {resolved_cwd}")
        return matches[0]

    def touch(self, metadata: SessionMetadata) -> None:
        metadata.updated_at = _now()
        self._write(metadata)

    def record_override(
        self,
        metadata: SessionMetadata,
        *,
        provider: str | None,
        model: str | None,
    ) -> None:
        audit: dict[str, Any] = {
            "at": _now(),
            "provider": {"from": metadata.provider, "to": provider}
            if provider is not None and provider != metadata.provider
            else None,
            "model": {"from": metadata.model, "to": model}
            if model is not None and model != metadata.model
            else None,
        }
        metadata.override_audit.append(audit)
        if provider is not None:
            metadata.provider = provider
        if model is not None:
            metadata.model = model
        self.touch(metadata)

    def _read(self, session_id: str) -> SessionMetadata:
        path = self.sessions_dir / session_id / "meta.json"
        if not path.exists():
            raise SessionError(f"session {session_id} was not found")
        try:
            with path.open() as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"session metadata could not be read: {path}") from exc
        if not isinstance(value, Mapping):
            raise SessionError(f"session metadata is not an object: {path}")
        return SessionMetadata.from_dict(value, path=path)

    def _write(self, metadata: SessionMetadata) -> None:
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
    "OpenedSession",
    "SessionError",
    "SessionManager",
    "SessionMetadata",
    "env_home",
    "find_most_recent",
    "list_sessions",
]


def list_sessions(home: str | Path | None = None) -> list[SessionMetadata]:
    return SessionManager(home).list_sessions()


def find_most_recent(
    *,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> SessionMetadata:
    return SessionManager(home).find_most_recent(cwd=cwd)
