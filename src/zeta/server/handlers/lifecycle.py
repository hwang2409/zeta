"""Session lifecycle request bodies: new/resume, listing, status, snapshots."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from ...core.session_execution import SessionOwnedError
from ...core.session_files import SessionIdAmbiguousError, SessionNotFoundError
from ...mcp.config import (
    MCPServerConfig,
    home_config_path,
    load_mcp_config,
    load_mcp_config_overlay,
)
from ...protocol.types import StreamEvent, StreamEventType
from .. import model_selection
from ..host_commands import SessionHostCommands
from ..protocol import MAX_FRAME_BYTES, ErrorDescriptor, ProtocolError
from . import validation

if TYPE_CHECKING:
    from ...runtime.loop import AgentLoop


_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_RESERVED_HEADERS = frozenset({
    "connection",
    "content-length",
    "expect",
    "forwarded",
    "host",
    "http2-settings",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "via",
    "x-real-ip",
})
_RESERVED_HEADER_PREFIXES = ("proxy-", "sec-websocket-", "x-forwarded-")


async def _publish_project_registry_notice(self) -> None:
    runtime = self.server.runtime
    if runtime.project_registry_busy:
        await self.server.control._event(
            StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={
                    "notification_id": f"project-registry-busy:{runtime.session_id}",
                    "kind": "project_registry_busy",
                    "text": (
                        "project memory and inbox are off for this session: "
                        "the project registry was busy; restart to retry"
                    ),
                },
            ),
            session_id=runtime.session_id,
        )


async def new_session(self, params: dict[str, Any]) -> dict[str, object]:
    try:
        bindings = await mcp_bindings(self, params)
        initial_model = validation.optional_string(params, "model")
        initial_mode = validation.optional_string(params, "approval_mode")
        if initial_mode is not None:
            validation.require_feature(self, "new_session_settings", "approval_mode")
            if initial_model is None:
                initial_model = validation.required_string(params, "model")
            model_selection.validate(initial_model, initial_mode)
        candidate = host_commands(self, params)
        with self.server.control.reserve_session_swap():
            self.server.connections.replay_session_replaced()
            cwd = None
            if "cwd" in params:
                validation.require_feature(self, "session_cwd", "cwd")
                cwd = validation.directory(params, "cwd")
            await self.server.control._terminate_pending_approvals()
            await self.server.runtime.create_session(
                provider=validation.optional_string(params, "provider"),
                model=initial_model,
                cwd=cwd,
            )
            if initial_model is not None and initial_mode is not None:
                await model_selection.apply_rpc(
                    self.server.runtime, initial_model, initial_mode
                )
            if bindings is not None:
                headers, configs = bindings
                await self.server.runtime.loop.set_mcp_bindings(headers, configs)
                self.server.control._active_mcp_bindings = headers
                self.server.control._mcp_binding_servers = frozenset(headers)
                self.server.control._mcp_binding_required = bool(headers)
            elif self.server.runtime.loop is not None:
                self.server.control._active_mcp_bindings = {}
                self.server.control._mcp_binding_servers = frozenset()
                self.server.control._mcp_binding_required = False
            self.server.control._active_host_commands = candidate
            self.server.control._host_binding_required = (
                "session_host_commands" in self.features
            )
            self.server.control.session_replaced()
            if self.server._retention_owner is not None:
                self.server._retention_owner.set_session_available()
            self.server.control.clear_approvals()
            await self.server.control.controller_ready()
            await _publish_project_registry_notice(self)
            await self.server.control.publish_session_grant_reset()
            return {"session": session_snapshot(self)}
    except BaseException:
        if self.server.runtime.state is None:
            self.server.control._active_host_commands = None
            self.server.control._host_binding_required = False
        raise


async def resume(self, params: dict[str, Any]) -> dict[str, object]:
    try:
        bindings = await mcp_bindings(self, params)
        state = self.server.runtime.state
        previous_session_id = state.metadata.session_id if state is not None else None
        candidate = host_commands(self, params)
        with self.server.control.reserve_session_swap():
            self.server.connections.replay_session_replaced()
            session_id = validation.required_string(params, "session_id")
            try:
                session_id = await asyncio.to_thread(
                    self.server.runtime.manager.resolve_id,
                    session_id,
                    min_prefix_length=4,
                )
            except SessionIdAmbiguousError as exc:
                raise ProtocolError(
                    ErrorDescriptor.SESSION_ID_AMBIGUOUS,
                    str(exc),
                    {"session_id": session_id, "candidates": list(exc.candidates)},
                ) from exc
            except SessionNotFoundError as exc:
                raise ProtocolError(
                    ErrorDescriptor.SESSION_NOT_FOUND,
                    str(exc),
                    {"session_id": session_id},
                ) from exc
            unhealthy_same_session = (
                self.server.control.unhealthy()
                and self.server.runtime.session_id == session_id
            )
            await self.server.control._terminate_pending_approvals()
            if unhealthy_same_session:
                try:
                    await self.server.runtime.resume_session(
                        session_id, force=True
                    )
                except SessionOwnedError as exc:
                    raise session_owned_protocol_error(exc) from exc
                except SessionNotFoundError as exc:
                    raise ProtocolError(
                        ErrorDescriptor.SESSION_NOT_FOUND,
                        str(exc),
                        {"session_id": session_id},
                    ) from exc
                except Exception as exc:
                    raise ProtocolError(
                        ErrorDescriptor.INTERNAL_ERROR,
                        "approval persistence failed",
                    ) from exc
            else:
                try:
                    await self.server.runtime.resume_session(session_id)
                except SessionOwnedError as exc:
                    raise session_owned_protocol_error(exc) from exc
                except SessionNotFoundError as exc:
                    raise ProtocolError(
                        ErrorDescriptor.SESSION_NOT_FOUND,
                        str(exc),
                        {"session_id": session_id},
                    ) from exc
            if bindings is not None:
                headers, configs = bindings
                await self.server.runtime.loop.set_mcp_bindings(headers, configs)
                self.server.control._active_mcp_bindings = headers
                self.server.control._mcp_binding_servers = frozenset(headers)
                self.server.control._mcp_binding_required = bool(headers)
            elif previous_session_id != session_id:
                self.server.control._active_mcp_bindings = {}
                self.server.control._mcp_binding_servers = frozenset()
                self.server.control._mcp_binding_required = False
            self.server.control._active_host_commands = candidate
            self.server.control._host_binding_required = (
                "session_host_commands" in self.features
            )
            self.server.control.session_replaced()
            if self.server._retention_owner is not None:
                self.server._retention_owner.set_session_available()
            self.server.control.clear_approvals()
            await self.server.control.controller_ready()
            await _publish_project_registry_notice(self)
            await self.server.control.publish_session_grant_reset()
            return {"session": session_snapshot(self)}
    except BaseException:
        if self.server.runtime.state is None:
            self.server.control._active_host_commands = None
            self.server.control._host_binding_required = False
        raise


async def mcp_bindings(
    self, params: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None:
    if "mcp_bindings" not in params:
        return None
    validation.require_feature(self, "session_mcp_bindings", "mcp_bindings")
    parsed = parse_mcp_bindings(params["mcp_bindings"])
    configs = await resolve_mcp_bindings(self, parsed, params.get("cwd"))
    return parsed, configs


async def reconnect_mcp_bindings(
    self, params: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None:
    if "mcp_bindings" not in params:
        return None
    validation.require_feature(self, "approval_reconnect", "mcp_bindings")
    bindings = await mcp_bindings(self, params)
    assert bindings is not None
    session_id = validation.required_string(params, "session_id")
    state = self.server.runtime.state
    control = self.server.control
    if (
        state is None
        or state.session_id != session_id
        or not control._mcp_binding_required
        or frozenset(bindings[0]) != control._mcp_binding_servers
    ):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "mcp_bindings must match the active session",
        )
    return bindings


def _valid_mcp_header(header: object, value: object) -> bool:
    if type(header) is not str or type(value) is not str or not header:
        return False
    folded = header.casefold()
    return (
        _HEADER_NAME.fullmatch(header) is not None
        and folded not in _RESERVED_HEADERS
        and not folded.startswith(_RESERVED_HEADER_PREFIXES)
        and not any(ord(char) < 32 and char not in "\t" for char in value)
        and not any(ord(char) > 126 for char in value)
        and len(header.encode()) <= 256
        and len(value.encode()) <= 8192
    )


def parse_mcp_bindings(raw: object) -> dict[str, dict[str, str]]:
    if type(raw) is not list:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "mcp_bindings must be a list")
    result: dict[str, dict[str, str]] = {}
    for item in raw:
        if type(item) is not dict or type(item.get("server")) is not str:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "invalid MCP binding")
        name = item["server"]
        headers = item.get("headers")
        if name in result or type(headers) is not dict or len(headers) > 64:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "invalid MCP binding")
        parsed: dict[str, str] = {}
        for header, value in headers.items():
            if not _valid_mcp_header(header, value):
                raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "invalid MCP binding")
            parsed[header] = value
        result[name] = parsed
    return result


async def resolve_mcp_bindings(
    self, bindings: dict[str, dict[str, str]], project_dir: object = None
) -> dict[str, MCPServerConfig]:
    home_path = home_config_path(self.server.runtime.home)
    loop = self.server.runtime.loop
    project_dir = (
        loop.tool_registry.cwd
        if loop is not None and project_dir is None
        else project_dir if project_dir is not None else self.server.runtime.cwd
    )
    config, effective = await asyncio.gather(
        asyncio.to_thread(load_mcp_config, home_path),
        asyncio.to_thread(
            load_mcp_config_overlay,
            home=self.server.runtime.home,
            project_dir=project_dir,
        ),
    )
    resolved: dict[str, MCPServerConfig] = {}
    for name in bindings:
        source = config.sources.get(name, config.path)
        server = config.configured_servers.get(name)
        effective_source = effective.sources.get(name, effective.path)
        if (
            server is None
            or source != config.path
            or effective_source != config.path
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                f"MCP server is not a trusted global server: {name}",
            )
        resolved[name] = server
    return resolved


def host_commands(self, params: dict[str, Any]) -> SessionHostCommands:
    if "host_commands" not in params:
        return SessionHostCommands.parse([])
    validation.require_feature(self, "session_host_commands", "host_commands")
    return SessionHostCommands.parse(params["host_commands"])


def _active_runtime_snapshot(
    persisted: dict[str, object], loop: AgentLoop | None
) -> dict[str, object]:
    if loop is None:
        return persisted
    snapshot = dict(persisted)
    prompt = loop._runtime_prompt.system_prompt
    if isinstance(prompt, str):
        snapshot["system_prompt"] = prompt
    snapshot["prompt_components"] = {
        key: dict(component)
        for key, component in loop._runtime_prompt.prompt_components.items()
    }
    snapshot["skill_catalog"] = (
        snapshot.get("skill_catalog")
        if loop._live_skill_catalog_snapshot is None
        else loop._live_skill_catalog_snapshot
    )
    snapshot["agent_catalog"] = (
        snapshot.get("agent_catalog")
        if loop._live_agent_catalog_snapshot is None
        else loop._live_agent_catalog_snapshot
    )
    return snapshot


def session_snapshot(self) -> dict[str, object] | None:
    runtime = self.server.runtime
    if runtime.opened is None:
        return None
    snapshot = _active_runtime_snapshot(runtime.metadata.to_dict(), runtime.loop)
    if runtime.policy is not None:
        snapshot["approval_mode"] = runtime.policy.default.value
    return snapshot


def session_runtime_state(self) -> str:
    runtime = self.server.runtime
    control = self.server.control
    return (
        "idle"
        if control.unhealthy() or runtime.state is None
        else "running" if control.turn_busy() else runtime.state.status
    )


async def live_status_baseline(self) -> dict[str, object]:
    pending, truncated = await self.server.control.reconnect_baseline()
    return {
        "state": session_runtime_state(self),
        "pending_approvals": pending,
        "pending_approvals_truncated": truncated,
    }


async def status(
    self,
    *,
    include_connection_setup_counts: bool = False,
    fallback_config: object | None = None,
) -> dict[str, object]:
    runtime = self.server.runtime
    session = session_snapshot(self)
    reconnect = "approval_reconnect" in self.features
    if reconnect:
        baseline = await live_status_baseline(self)
        state = baseline["state"]
        pending_approvals = baseline["pending_approvals"]
        pending_approvals_truncated = baseline["pending_approvals_truncated"]
    else:
        state = session_runtime_state(self)
        pending_approvals = await self.server.control.pending_approvals()
        pending_approvals_truncated = None
    loop = runtime.loop
    if loop is None and fallback_config is None:
        fallback_config = await asyncio.to_thread(runtime._config, None, None)
    project_context_health = (
        (
            loop.project_context_health.summary
            if loop is not None and loop.project_context_health is not None
            else "healthy"
        )
        if self.protocol_version != "1.0"
        else None
    )
    transcript_index_failure = (
        loop.transcript_index_failure(runtime.session_id)
        if self.protocol_version != "1.0" and loop is not None
        else None
    )
    status = {
        "session": session,
        "state": state,
        "explicit_finish": (
            loop.explicit_finish
            if loop is not None
            else fallback_config.agents_explicit_finish  # type: ignore[union-attr]
        ),
    }
    if self.protocol_version != "1.0":
        status["max_agent_depth"] = (
            loop.max_agent_depth
            if loop is not None
            else fallback_config.agents_max_depth  # type: ignore[union-attr]
        )
    suffix = {
        "pending_approvals": pending_approvals,
        **(
            {"pending_approvals_truncated": pending_approvals_truncated}
            if reconnect
            else {}
        ),
        "usage": dict(runtime.usage),
        **(
            {
                "connection_setup_counts": (
                    runtime.loop.connection_setup_counts()
                    if runtime.loop is not None
                    else ()
                )
            }
            if include_connection_setup_counts
            else {}
        ),
        **(
            {"transcript_index_failure": transcript_index_failure}
            if transcript_index_failure is not None
            else {}
        ),
        **(
            {"project_context_health": project_context_health}
            if project_context_health is not None
            else {}
        ),
        "store": runtime.opened.store if runtime.opened is not None else None,
    }
    status.update(suffix)
    slash_command = self.server.control.slash_lock.status()
    if slash_command is not None:
        status["slash_command"] = slash_command
    return status


def list_sessions(
    self, request_id: str | int, params: dict[str, Any]
) -> tuple[dict[str, object], list[str]]:
    return _list_sessions(self, request_id, params)


def _list_sessions(
    self, request_id: str | int, params: dict[str, Any]
) -> tuple[dict[str, object], list[str]]:
    paging = "list_sessions_paging" in self.features
    for name in ("offset", "limit"):
        if name in params:
            validation.require_feature(self, "list_sessions_paging", name)
    offset = validation.integer(params, "offset", 0, minimum=0)
    limit = validation.integer(params, "limit", None, minimum=1)
    if "project_id" in params:
        validation.require_feature(self, "projects", "project_id")
        assert self.projects is not None
        available = self.projects.project_sessions(params["project_id"])
    else:
        available = self.server.runtime.list_sessions()
    metadata = available[offset : None if limit is None else offset + limit]
    previews = {
        item.session_id: item
        for item in self.server.runtime.manager.list_session_previews(
            limit=len(metadata), sessions=metadata
        )
    }
    sessions = []
    for item in metadata:
        preview = previews[item.session_id]
        sessions.append(
            {
                **item.to_catalog_dict(),
                "first_message_preview": preview.preview,
                "in_use": preview.in_use,
                **(
                    {"owner_pid": preview.owner_pid}
                    if preview.owner_pid is not None
                    else {}
                ),
            }
        )
    if self.protocol_version == "1.0":
        for item in sessions:
            item.pop("name", None)
    page: list[dict[str, Any]] = []
    for index, item in enumerate(sessions):
        candidate = [*page, item]
        if not self.codec.response_fits(request_id, {"sessions": candidate}) or (
            len(self.codec.response(request_id, {"sessions": candidate}))
            > MAX_FRAME_BYTES - 128
        ):
            return {
                "sessions": page,
                "truncated": True,
                "next_offset": offset + index,
            }, [item["session_id"] for item in page]
        page = candidate
    page_ids = [item["session_id"] for item in page]
    if not paging:
        return {"sessions": page}, page_ids
    end = offset + len(page)
    return {
        "sessions": page,
        "next_offset": end if end < len(available) else None,
    }, page_ids


def session_owned_protocol_error(exc: SessionOwnedError) -> ProtocolError:
    data: dict[str, Any] = {
        "code": "session_owned",
        "session_id": exc.session_id,
    }
    owner = {
        key: value
        for key, value in {
            "surface": exc.owner_surface,
            "pid": exc.owner_pid,
        }.items()
        if value is not None
    }
    if owner:
        data["owner"] = owner
    return ProtocolError(
        ErrorDescriptor.OPERATION_BUSY,
        "session is active in another zeta process",
        data,
    )
