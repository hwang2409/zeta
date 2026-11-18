"""Validated, metadata-only state reads backed by the serve read pool."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import storage_format
from ..core.slash import home_slash_registry
from ..read_page import InvalidParams, Key, page_result, page_room, resume
from ..skills.agent_catalog import discover_session_agents
from ..skills.catalog import discover_session_skills
from . import project_state
from .protocol import ErrorDescriptor, ProtocolError

_ORDER = (Key("name", str), Key("source", str))
_DIRECTORIES = {"profiles": "agents", "skills": "skills", "commands": "commands"}
_MAX_ITEMS = 1_000


def _owner_rows(home: Path, kind: str) -> list[dict[str, str]]:
    def row(name: str, description: str, source: str) -> dict[str, str]:
        return {"name": name, "description": description, "source": source}

    if kind == "profiles":
        return [
            row(item.name, item.selection_guidance, "home" if item.source == "home" else "built_in")
            for item in discover_session_agents(home=home).agents
        ]
    if kind == "skills":
        return [
            row(item.name, item.description, "home" if item.source == "home" else "built_in")
            for item in discover_session_skills(home=home).skills
        ]
    registry = home_slash_registry(home)
    return [
        row(name, description, source or "built_in")
        for name, description, source in registry.completion_entries
    ]


def _generation(home: Path, kind: str) -> object:
    # These owners have no generation pointer. Use file identities and change times,
    # including the directory itself to catch additions and removals during discovery.
    root = home / _DIRECTORIES[kind]
    paths = [home / "storage-format.json", root, *sorted(root.glob("*.md"))]
    if kind == "skills":
        paths.extend(sorted(root.glob("*/SKILL.md")))
    values = []
    for path in paths:
        try:
            info = path.stat()
        except FileNotFoundError:
            values.append((path, None))
        else:
            values.append((path, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)))
    return values


def _stable_snapshot[T](generation: Callable[[], object], read: Callable[[], T]) -> T:
    for _ in range(2):
        before = generation()
        try:
            result = read()
        except (OSError, ValueError, ProtocolError):
            if generation() == before:
                raise
        else:
            if generation() == before:
                return result
    raise ProtocolError(
        ErrorDescriptor.STALE_STATE_SNAPSHOT, "snapshot changed; retry", {"retryable": True}
    )


def validated_state_read[T](
    home: Path, kind: str, generation: Callable[[], object], owner_read: Callable[[int], T],
) -> T:
    def read() -> T:
        try:
            with storage_format.open_reader(home) as epoch:
                return owner_read(epoch)
        except storage_format.MissingManifest:
            raise ProtocolError(
                ErrorDescriptor.STATE_READ_UNAVAILABLE, "state unavailable", {"retryable": True}
            ) from None
        except storage_format.StorageFormatError:
            raise ProtocolError(
                ErrorDescriptor.CORRUPT_STATE, "stored state is invalid", {"resource": "storage-format.json"}
            ) from None
        except ValueError:
            raise ProtocolError(
                ErrorDescriptor.CORRUPT_STATE, "stored state is invalid", {"resource": kind}
            ) from None

    try:
        return _stable_snapshot(generation, read)
    except OSError:
        raise ProtocolError(
            ErrorDescriptor.STATE_READ_UNAVAILABLE, "state unavailable", {"retryable": True}
        ) from None


def _snapshot(home: Path, kind: str) -> list[dict[str, str]]:
    rows = validated_state_read(home, kind, lambda: _generation(home, kind), lambda _epoch: _owner_rows(home, kind))
    return sorted(rows, key=_position)[:_MAX_ITEMS]


def _position(row: dict[str, str]) -> tuple[str, str]:
    return row["name"], row["source"]


class ReadCatalog:
    """Serve state projections; one active owner read and one snapshot per in-flight kind."""

    def __init__(self, run: Callable[[Callable[[], Any]], Awaitable[Any]]) -> None:
        self._run = run
        self._slot = asyncio.Lock()
        self._pending: dict[tuple, asyncio.Task[Any]] = {}

    def close(self) -> None:
        for task in self._pending.values():
            task.cancel()

    async def catalog(
        self, home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        kind = params.get("kind")
        limit = params.get("limit", 50)
        if (
            params.keys() - {"kind", "limit", "cursor"}
            or not isinstance(kind, str) or kind not in _DIRECTORIES
            or type(limit) is not int or not 1 <= limit <= 200
        ):
            raise InvalidParams("invalid catalog params")
        start = resume(params.get("cursor"), ("state_catalog", kind), (), _ORDER)
        room = page_room(start, limit, _MAX_ITEMS)
        rows = await self._rows(home, kind)

        def page() -> dict[str, Any]:
            after = [row for row in rows if start.position is None or _position(row) > start.position]
            return page_result(
                after[:room], lambda index: _position(after[index]), start,
                more=len(after) > room, total=_MAX_ITEMS, fits=fits,
                shape=lambda items, cursor: {"kind": kind, "items": items, "next_cursor": cursor},
            )

        async with self._slot:
            return await self._run(page)

    async def project_state(
        self, home: Path, collection: str, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        scope, start, limit = project_state.request(collection, params)
        rows = await self._share((home, "project_state", *scope), lambda: project_state.snapshot(home, scope, _stable_snapshot))
        async with self._slot:
            return await self._run(lambda: project_state.page(collection, rows, start, limit, fits))

    async def _rows(self, home: Path, kind: str) -> list[dict[str, str]]:
        return await self._share((home, kind), lambda: _snapshot(home, kind))

    async def _share(self, key: tuple, snapshot: Callable[[], Any]) -> Any:
        task = self._pending.get(key)
        if task is None or task.done():
            async def read() -> Any:
                async with self._slot:
                    return await self._run(snapshot)

            def finished(done: asyncio.Task[Any]) -> None:
                if self._pending.get(key) is done:
                    del self._pending[key]
                if not done.cancelled():
                    done.exception()

            task = self._pending[key] = asyncio.create_task(read())
            task.add_done_callback(finished)
        return await asyncio.shield(task)
