"""Negotiated format-2 project-memory responses for serve."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from zeta.memory.entry_store import MemoryEntry
from zeta.memory.entry_views import entry_value, kind_views, render_all_kinds
from zeta.project_registry import Project, ProjectRegistry
from zeta.server.protocol import FrameCodec

MAX_DIFF_BYTES = 64 * 1024
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1_000


class EntryProjectViews:
    """Build bounded entry responses behind one negotiated serve seam."""

    def __init__(
        self,
        *,
        registry: ProjectRegistry,
        runtime: Any,
        codec: FrameCodec,
        invalid: Callable[[str], Exception],
    ) -> None:
        self.registry = registry
        self.runtime = runtime
        self.codec = codec
        self.invalid = invalid

    def show(self, request_id: str | int, project: Project) -> dict[str, object]:
        sessions = self.runtime.manager.list_sessions_read_only()
        snapshot = self.registry._entry_memory_state(project.project_id)
        kinds = kind_views(snapshot.state)
        result: dict[str, object] = {
            "project": self._project_metadata(project, sessions),
            "memory": {
                "version_id": snapshot.version,
                "digest": snapshot.digest,
                "schema_version": snapshot.state.schema.version,
                "profile": snapshot.state.schema.profile,
                "kinds": kinds,
            },
        }
        while not self.codec.response_fits(request_id, result):
            largest = max(kinds, key=lambda item: len(str(item["rendered"]).encode()))
            rendered = str(largest["rendered"])
            size = len(rendered.encode())
            if size == 0:
                raise RuntimeError("project metadata exceeds the protocol frame limit")
            largest["rendered"], _ = _truncate_utf8(rendered, size // 2)
            largest["rendered_truncated"] = True
        return result

    def memory_log(
        self,
        request_id: str | int,
        project: Project,
        params: dict[str, Any],
    ) -> dict[str, object]:
        self._only(
            params,
            {"project_id", "offset", "limit", "version_id", "entry_id"},
        )
        version = params.get("version_id")
        entry_id = params.get("entry_id")
        if entry_id is not None and (not isinstance(entry_id, str) or not entry_id):
            raise self.invalid("entry_id must be a non-empty string")
        if version is not None:
            return self._version(request_id, project, params, version, entry_id)
        offset = self._integer(params, "offset", 0, minimum=0)
        limit = self._integer(
            params, "limit", DEFAULT_PAGE_LIMIT, minimum=1, maximum=MAX_PAGE_LIMIT
        )
        records = self.registry._entry_memory_log(
            project.project_id, entry_id=entry_id, limit=10_000
        )
        return self._page(request_id, "versions", records, offset, limit)

    def _version(
        self,
        request_id: str | int,
        project: Project,
        params: dict[str, Any],
        version: object,
        entry_id: object,
    ) -> dict[str, object]:
        if not isinstance(version, str) or not version:
            raise self.invalid("version_id must be a non-empty string")
        if "offset" in params or "limit" in params:
            raise self.invalid("offset and limit cannot be used with version_id")
        record, before, after = self.registry._entry_memory_version(
            project.project_id, version
        )
        touched = {
            item
            for operation in record["operations"]
            for item in (
                *operation.get("target_ids", []),
                *operation.get("result_ids", []),
            )
        }
        if entry_id is not None and entry_id not in touched:
            raise self.invalid(f"memory version does not touch entry: {entry_id}")
        if isinstance(entry_id, str):
            touched = {entry_id}
        before_entries = self._entries(before.entries, touched)
        after_entries = self._entries(after.entries, touched)
        before_text = "\n".join(render_all_kinds(before).values())
        after_text = "\n".join(render_all_kinds(after).values())
        diff = "".join(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"memory@{version}:before",
                tofile=f"memory@{version}",
            )
        )
        diff, truncated = _truncate_utf8(diff, MAX_DIFF_BYTES)
        result = {
            "version": {
                **record,
                "before_entries": before_entries,
                "after_entries": after_entries,
                "diff": diff,
                "diff_truncated": truncated,
            }
        }
        self._fit_version(request_id, result)
        return result

    @staticmethod
    def _entries(entries: dict[str, object], touched: set[str]) -> list[dict[str, object]]:
        return [
            entry_value(entry)
            for key in sorted(touched)
            if isinstance((entry := entries.get(key)), MemoryEntry)
        ]

    def _fit_version(self, request_id: str | int, result: dict[str, object]) -> None:
        detail = result["version"]
        assert isinstance(detail, dict)
        while not self.codec.response_fits(request_id, result):
            if detail["diff"]:
                value = str(detail["diff"])
                detail["diff"], _ = _truncate_utf8(value, len(value.encode()) // 2)
                detail["diff_truncated"] = True
                continue
            entries = [*detail["before_entries"], *detail["after_entries"]]
            texts = [entry for entry in entries if entry.get("text")]
            if not texts:
                raise RuntimeError("memory version metadata exceeds the frame limit")
            largest = max(texts, key=lambda item: len(str(item["text"]).encode()))
            text = str(largest["text"])
            largest["text"], _ = _truncate_utf8(text, len(text.encode()) // 2)
            largest["text_truncated"] = True

    def _page(
        self,
        request_id: str | int,
        key: str,
        available: list[dict[str, object]],
        offset: int,
        limit: int,
    ) -> dict[str, object]:
        selected = available[offset : offset + limit]
        page: list[dict[str, object]] = []
        for item in selected:
            candidate = [*page, item]
            result = {key: candidate, "next_offset": None}
            if not self.codec.response_fits(request_id, result):
                break
            page = candidate
        while True:
            end = offset + len(page)
            result = {
                key: page,
                "next_offset": end if end < len(available) else None,
                **({"truncated": True} if len(page) < len(selected) else {}),
            }
            if self.codec.response_fits(request_id, result) or not page:
                return result
            page.pop()

    def _only(self, params: dict[str, Any], allowed: set[str]) -> None:
        unknown = set(params) - allowed
        if unknown:
            raise self.invalid(f"unknown parameter: {min(unknown)}")

    def _integer(
        self,
        params: dict[str, Any],
        name: str,
        default: int,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        value = params.get(name, default)
        if type(value) is not int or value < minimum or (
            maximum is not None and value > maximum
        ):
            raise self.invalid(f"{name} is out of range")
        return value

    @staticmethod
    def _project_metadata(project: Project, sessions: list[Any]) -> dict[str, object]:
        linked = [item for item in sessions if item.project_id == project.project_id]
        return {
            "id": project.project_id,
            "name": project.name,
            "scope": project.scope,
            "roots": [project.canonical_integration_root]
            if project.canonical_integration_root is not None
            else [],
            "session_count": len(linked),
            "last_activity": max(
                [project.updated_at, *(item.updated_at for item in linked)]
            ),
            "created_at": project.created_at,
            "updated_at": project.updated_at,
        }


def _truncate_utf8(value: str, maximum: int) -> tuple[str, bool]:
    payload = value.encode("utf-8")
    if len(payload) <= maximum:
        return value, False
    return payload[:maximum].decode("utf-8", errors="ignore"), True


__all__ = ["EntryProjectViews"]
