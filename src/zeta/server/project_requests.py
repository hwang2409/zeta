"""Read-only project views for the serve protocol."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..project_errors import ProjectRegistryError
from ..project_inbox import ProjectInbox
from ..project_memory_history import PROJECT_MEMORY_FILES
from ..project_registry import Project, ProjectRegistry
from .protocol import FrameCodec

if TYPE_CHECKING:
    from .runtime import ServerRuntime


class ProjectNotFound(ValueError):
    def __init__(self, project_id: str) -> None:
        super().__init__(f"project not found: {project_id}")
        self.project_id = project_id


PROJECT_REQUESTS = (
    "list_projects",
    "project_show",
    "project_memory_log",
    "project_inbox",
)
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1_000
MAX_DIFF_BYTES = 64 * 1024


class ProjectRequests:
    """Build bounded project views without changing project or session state."""

    def __init__(
        self, *, home: Path, runtime: ServerRuntime, codec: FrameCodec
    ) -> None:
        self.registry = ProjectRegistry(home / "projects")
        self.inbox = ProjectInbox(self.registry, sessions_root=home / "sessions")
        self.runtime = runtime
        self.codec = codec

    def dispatch(
        self,
        request_id: str | int,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, object]:
        if method == "list_projects":
            return self._list_projects(request_id, params)
        project_id = self._project_id(params)
        project = self.require_project(project_id)
        if method == "project_show":
            self._only(params, {"project_id"})
            return self._show(request_id, project)
        if method == "project_memory_log":
            return self._memory_log(request_id, project, params)
        if method == "project_inbox":
            return self._inbox(request_id, project, params)
        raise AssertionError(f"unknown project request: {method}")

    def _list_projects(
        self, request_id: str | int, params: dict[str, Any]
    ) -> dict[str, object]:
        self._only(params, {"offset", "limit"})
        offset = self._integer(params, "offset", 0, minimum=0)
        limit = self._integer(
            params, "limit", DEFAULT_PAGE_LIMIT, minimum=1, maximum=MAX_PAGE_LIMIT
        )
        sessions = self.runtime.manager.list_sessions_read_only()
        projects = [
            self._project_metadata(project, sessions)
            for project in self.registry.list_projects()
        ]
        return self._page(request_id, "projects", projects, offset, limit)

    def _show(
        self, request_id: str | int, project: Project
    ) -> dict[str, object]:
        sessions = self.runtime.manager.list_sessions_read_only()
        state = self.registry.memory_state(project.project_id)
        automatic = set(state.automatic_files)
        files = [
            {
                "name": name,
                "content": state.contents.get(name, ""),
                "automatic": name in automatic,
                "content_truncated": False,
            }
            for name in PROJECT_MEMORY_FILES
        ]
        result = {
            "project": self._project_metadata(project, sessions, include_dates=True),
            "memory": {
                "version_id": state.version,
                "digest": state.digest,
                "files": files,
            },
        }
        while not self.codec.response_fits(request_id, result):
            largest = max(files, key=lambda item: len(str(item["content"]).encode()))
            content = str(largest["content"])
            size = len(content.encode())
            if size == 0:
                raise RuntimeError("project metadata exceeds the protocol frame limit")
            largest["content"], _ = _truncate_utf8(content, size // 2)
            largest["content_truncated"] = True
        return result

    def _memory_log(
        self,
        request_id: str | int,
        project: Project,
        params: dict[str, Any],
    ) -> dict[str, object]:
        self._only(params, {"project_id", "offset", "limit", "version_id", "file"})
        version = params.get("version_id")
        name = params.get("file")
        if version is not None or name is not None:
            if not isinstance(version, str) or not version:
                raise ValueError("version_id must be a non-empty string")
            if not isinstance(name, str) or name not in PROJECT_MEMORY_FILES:
                raise ValueError("file must name a project memory file")
            if "offset" in params or "limit" in params:
                raise ValueError("offset and limit cannot be used with version_id")
            item = self.registry.memory_version_file(project.project_id, version, name)
            diff = "".join(
                difflib.unified_diff(
                    item.parent_content.splitlines(keepends=True),
                    item.content.splitlines(keepends=True),
                    fromfile=f"{name}@parent",
                    tofile=f"{name}@{version}",
                )
            )
            diff, truncated = _truncate_utf8(diff, MAX_DIFF_BYTES)
            result = {
                "version": {
                    **self._record(item.record),
                    "file": name,
                    "content": item.content,
                    "content_truncated": False,
                    "diff": diff,
                    "diff_truncated": truncated,
                }
            }
            while not self.codec.response_fits(request_id, result):
                detail = result["version"]
                assert isinstance(detail, dict)
                field = "diff" if detail["diff"] else "content"
                value = str(detail[field])
                size = len(value.encode())
                if size == 0:
                    raise RuntimeError("memory version metadata exceeds the frame limit")
                detail[field], _ = _truncate_utf8(value, size // 2)
                detail[f"{field}_truncated"] = True
            return result
        offset = self._integer(params, "offset", 0, minimum=0)
        limit = self._integer(
            params, "limit", DEFAULT_PAGE_LIMIT, minimum=1, maximum=MAX_PAGE_LIMIT
        )
        records = [self._record(item) for item in self.registry.memory_log(project.project_id, limit=10_000)]
        return self._page(request_id, "versions", records, offset, limit)

    def _inbox(
        self,
        request_id: str | int,
        project: Project,
        params: dict[str, Any],
    ) -> dict[str, object]:
        self._only(params, {"project_id", "status", "offset", "limit"})
        status = params.get("status", "new")
        if status not in {"new", "claimed", "done"}:
            raise ValueError("status must be new, claimed, or done")
        offset = self._integer(params, "offset", 0, minimum=0)
        limit = self._integer(
            params, "limit", DEFAULT_PAGE_LIMIT, minimum=1, maximum=MAX_PAGE_LIMIT
        )
        state = self.inbox.read(project.project_id)
        messages = [
            self._bounded_message(request_id, item) for item in state[status]
        ]
        return self._page(
            request_id,
            "messages",
            messages,
            offset,
            limit,
            extra={"status": status, "untrusted": True},
        )

    def _bounded_message(
        self, request_id: str | int, message: dict[str, Any]
    ) -> dict[str, object]:
        result: dict[str, object] = dict(message)
        truncated: list[str] = []
        envelope = {"status": "new", "untrusted": True, "messages": [result]}
        while not self.codec.response_fits(request_id, envelope):
            strings = [
                (key, value)
                for key, value in result.items()
                if isinstance(value, str) and value
            ]
            if not strings:
                raise RuntimeError("inbox message metadata exceeds the frame limit")
            key, value = max(strings, key=lambda item: len(item[1].encode()))
            result[key], _ = _truncate_utf8(value, len(value.encode()) // 2)
            if key not in truncated:
                truncated.append(key)
            result["truncated_fields"] = truncated
        return result

    def _page(
        self,
        request_id: str | int,
        key: str,
        available: list[dict[str, object]],
        offset: int,
        limit: int,
        *,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        selected = available[offset : offset + limit]
        page: list[dict[str, object]] = []
        for item in selected:
            candidate = [*page, item]
            result = {**(extra or {}), key: candidate, "next_offset": None}
            if not self.codec.response_fits(request_id, result):
                break
            page = candidate
        while True:
            end = offset + len(page)
            result = {
                **(extra or {}),
                key: page,
                "next_offset": end if end < len(available) else None,
                **({"truncated": True} if len(page) < len(selected) else {}),
            }
            if self.codec.response_fits(request_id, result) or not page:
                return result
            page.pop()

    @staticmethod
    def _record(record: dict[str, object]) -> dict[str, object]:
        raw_provenance = record.get("provenance", {})
        provenance = (
            {
                key: value
                for key, value in raw_provenance.items()
                if key
                in {
                    "session_id",
                    "seq_start",
                    "seq_end",
                    "model",
                    "accepted_by",
                    "source",
                    "peer",
                }
            }
            if isinstance(raw_provenance, dict)
            else {}
        )
        provenance_truncated = False
        for key, value in list(provenance.items()):
            if isinstance(value, str):
                provenance[key], truncated = _truncate_utf8(value, 4096)
                provenance_truncated = provenance_truncated or truncated
        result: dict[str, object] = {
            "version_id": record.get("version"),
            "timestamp": record.get("created_at"),
            "kind": record.get("kind"),
            "files_changed": list(record.get("files", [])),
            "provenance": provenance,
        }
        if provenance_truncated:
            result["provenance_truncated"] = True
        if "target_version" in record:
            result["target_version_id"] = record["target_version"]
        return result

    @staticmethod
    def _project_metadata(
        project: Project, sessions: list[Any], *, include_dates: bool = False
    ) -> dict[str, object]:
        linked = [item for item in sessions if item.project_id == project.project_id]
        last_activity = max(
            [project.updated_at, *(item.updated_at for item in linked)]
        )
        result: dict[str, object] = {
            "id": project.project_id,
            "name": project.name,
            "scope": project.scope,
            "roots": [project.canonical_integration_root]
            if project.canonical_integration_root is not None
            else [],
            "session_count": len(linked),
            "last_activity": last_activity,
        }
        if include_dates:
            result.update(created_at=project.created_at, updated_at=project.updated_at)
        return result

    def require_project(self, project_id: str) -> Project:
        try:
            return self.registry.show_project(project_id)
        except ProjectRegistryError as exc:
            raise ProjectNotFound(project_id) from exc

    @staticmethod
    def _project_id(params: dict[str, Any]) -> str:
        value = params.get("project_id")
        if not isinstance(value, str) or not value:
            raise ValueError("project_id must be a non-empty string")
        return value

    @staticmethod
    def _only(params: dict[str, Any], allowed: set[str]) -> None:
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unknown parameter: {min(unknown)}")

    @staticmethod
    def _integer(
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
            raise ValueError(f"{name} is out of range")
        return value


def _truncate_utf8(value: str, maximum: int) -> tuple[str, bool]:
    payload = value.encode("utf-8")
    if len(payload) <= maximum:
        return value, False
    return payload[:maximum].decode("utf-8", errors="ignore"), True


__all__ = ["PROJECT_REQUESTS", "ProjectNotFound", "ProjectRequests"]
