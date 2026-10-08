"""Immutable built-in schemas for dormant format-2 project memory.

Profiles are trusted code data. Activation commands do not import this module;
PR 2 uses them only through format-2 test fixtures and the dormant updater.
"""

from __future__ import annotations

from types import MappingProxyType

from .entry_store import MemoryKind, MemorySchema, validate_schema


def _kind(
    key: str,
    name: str,
    description: str,
    mode: str,
    priority: int,
    maximum: int,
    expiry: int | None = None,
) -> MemoryKind:
    return MemoryKind(
        key=key,
        name=name,
        description=description,
        prompt_mode=mode,  # type: ignore[arg-type]
        prompt_priority=priority,
        prompt_max_entries=maximum,
        default_expiry_days=expiry,
    )


_ZETA = MemorySchema(
    version=3,
    profile="zeta",
    kinds=(
        _kind("brief", "Brief", "Stable project purpose, architecture, and invariants.", "always", 100, 100),
        _kind("decisions", "Decisions", "Binding user decisions, validated procedures, and failure lessons.", "always", 90, 100),
        _kind("state", "Current state", "Current project and active work state, not progress narration. Supersede or resolve progress reports when completion evidence arrives.", "recent", 80, 100, 30),
        _kind("backlog", "Backlog", "Open follow-up work that remains actionable. Resolve completed items instead of retaining progress narration.", "recent", 70, 100, 60),
        _kind("changelog", "Changelog", "Completed durable outcomes.", "recent", 40, 100, 180),
    ),
)

_MESSAGING = MemorySchema(
    version=3,
    profile="messaging",
    kinds=(
        _kind("people", "People", "Stable facts needed to identify or understand people.", "always", 100, 100),
        _kind("preferences", "Preferences", "Stated likes, dislikes, communication preferences, and corrections.", "always", 90, 100),
        _kind("routines", "Routines", "Repeated schedules and habits.", "recent", 70, 100, 90),
        _kind("threads", "Threads", "Open conversation topics and follow-ups.", "recent", 60, 100, 30),
        _kind("commitments", "Commitments", "Promises, requests, deadlines, and obligations until completed.", "always", 95, 100, 60),
    ),
)

for _profile in (_ZETA, _MESSAGING):
    validate_schema(_profile)

BUILTIN_PROFILES: MappingProxyType[str, MemorySchema] = MappingProxyType(
    {"zeta": _ZETA, "messaging": _MESSAGING}
)

_EARLY_UPDATE_DEBOUNCE_SECONDS: MappingProxyType[str, float] = MappingProxyType(
    {"zeta": 60.0, "messaging": 15.0}
)
_EARLY_UPDATE_MAX_WAIT_SECONDS: MappingProxyType[str, float] = MappingProxyType(
    {"zeta": 300.0, "messaging": 120.0}
)


def early_update_debounce_seconds(profile: str) -> float:
    """Return the completed-turn debounce selected by a memory profile."""
    try:
        return _EARLY_UPDATE_DEBOUNCE_SECONDS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown memory profile: {profile}") from exc


def early_update_max_wait_seconds(profile: str) -> float:
    """Return the maximum completed-turn coalescing delay for a profile."""
    try:
        return _EARLY_UPDATE_MAX_WAIT_SECONDS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown memory profile: {profile}") from exc


def memory_profile(name: str) -> MemorySchema:
    """Return one immutable built-in schema by name."""
    try:
        return BUILTIN_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown memory profile: {name}") from exc
