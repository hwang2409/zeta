"""Session lifecycle request bodies: new/resume, listing, status, snapshots."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
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
from .. import control as control_module
from .. import model_selection
from ..host_commands import SessionHostCommands
from ..protocol import MAX_FRAME_BYTES, ErrorDescriptor, ProtocolError
from . import validation
from .telemetry import live_view

if TYPE_CHECKING:
    from ...runtime.loop import AgentLoop
    from ..control import ServeControl
    from ..session_controllers import SessionController, SessionUse


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


async def _publish_project_registry_notice(self, controller: SessionUse | SessionController) -> None:
    runtime = controller.runtime
    if runtime.project_registry_busy:
        await controller.control._event(
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


@dataclass(frozen=True, slots=True)
class _Outgoing:
    """What a multi swap does with the connection's current session: nothing
    (``record`` None), park it, or close it."""

    record: SessionController | None = None
    park: bool = False

    async def leave(self, client: Any) -> None:
        """Run the decision once the swap target is committed."""
        if self.record is None:
            return
        # storage that failed during setup closes it too: never park an unhealthy session
        if self.park and not self.record.control.unhealthy():
            # a swap is not a disconnect: park as legacy attach does; grace starts at disconnect
            client.server.parking.leave(self.record, client)
            return
        try:
            cancelled = await _settle_controller_release(client, self.record.control)
        finally:
            client.server.controllers.retire(self.record)
        if cancelled:
            raise asyncio.CancelledError


@contextmanager
def _outgoing(
    self: Any, target_id: str | None, target: SessionController | None = None
) -> Iterator[_Outgoing]:
    """Decide once what a multi swap does with the connection's current session
    and hold that decision's reservation until the swap ends:

    - nothing: no open session (the shared sessionless record), or it is the target;
    - park: with ``session_park``, a healthy session of another id that holds
      work; only this case needs parked room, and ``target`` does not count;
    - close: anything else, as legacy does in place; its background children and
      tasks stop. A busy turn refuses the swap with -32004 first, unless
      ``session_park`` closes an unhealthy or same-id session."""
    controllers = self.server.controllers
    record = controllers.session_of(self)
    if record is None or record is target or record.runtime.opened is None:
        yield _Outgoing()
        return
    control = record.control
    session_park = "session_park" in self.features
    retire = control.unhealthy() or record.runtime.session_id == target_id
    if session_park and not retire and control.holds_work():
        with controllers.reserve_park(self, record, target=target):
            yield _Outgoing(record, park=True)
        return
    with contextlib.nullcontext() if session_park and retire else control.reserve_session_swap():
        yield _Outgoing(record)


@asynccontextmanager
async def _session_swap(
    self, park: bool, controller: SessionUse | SessionController, target_id: str | None = None
) -> AsyncIterator[tuple[SessionController | None, _Outgoing]]:
    """Hold a session swap and yield ``(incoming, outgoing)``. ``incoming`` is
    None to swap the attached pair in place, as without ``session_park``; or,
    in multi mode or when ``park`` and the attached pair holds work, a new pair
    to set up beside it. The caller commits that pair (``server.attach``, or
    the multi acquisition) right after its setup returns; a setup that fails
    first is closed here. ``outgoing`` is the multi decision for the
    connection's current session against ``target_id`` (empty in legacy)."""
    server = self.server
    outgoing = controller.control
    multi = server.controllers.controller_mode == "multi"
    if not multi and (not park or not outgoing.holds_work()):
        with outgoing.reserve_session_swap():
            server.connections.replay_session_replaced()
            yield None, _Outgoing()
        return
    # Keep the outgoing charge and incoming use across setup and rollback.
    with (
        _outgoing(self, target_id) if multi else contextlib.nullcontext(_Outgoing()) as leaving,
        (server.new_pair(controller, connection=self) if multi else server.new_pair()) as reservation,
    ):
        incoming = reservation.record
        with incoming.control.reserve_session_swap():
            try:
                yield incoming, leaving
            except BaseException:
                if (
                    not incoming.control.attached
                    and server.controllers.controller_for(self) is not incoming
                ):
                    closed = server.controllers.retire(incoming)
                    reservation.release()
                    await asyncio.shield(closed)
                raise


async def new_session(self, params: dict[str, Any], controller: SessionUse) -> dict[str, object]:
    return await _setup(self, _new_session, params, controller)


async def resume(self, params: dict[str, Any], controller: SessionUse) -> dict[str, object]:
    return await _setup(self, _resume, params, controller)


async def _setup(self, body: Any, params: dict[str, Any], controller: SessionUse) -> dict[str, object]:
    """Run a create or resume as the one setup completion that owns the
    request's use (and, inside it, the new pair's reservation): a cancel of the
    request reaches the setup once and its cleanup still holds both records."""

    async def setup() -> dict[str, object]:
        # a queued wake may not start its setup while this state change runs
        with controller.control.hold_wake_start():
            return await body(self, params, controller)

    return await controller.run(setup())


async def _settle_controller_release(
    self: Any, control: ServeControl
) -> bool:
    """Finish one semantic release and report deferred caller cancellation."""
    release = control.begin_controller_release(
        reconnect="approval_reconnect" in self.features,
        keep_turn="turn_reconnect" in self.features,
    )
    cancelled = False
    while True:
        try:
            await asyncio.shield(release)
            return cancelled
        except asyncio.CancelledError:
            if release.cancelled():
                raise
            cancelled = True


async def _new_session(
    self, params: dict[str, Any], controller: SessionUse | SessionController
) -> dict[str, object]:
    acquisition = None
    try:
        bindings = await mcp_bindings(self, params, controller)
        initial_model = validation.optional_string(params, "model")
        initial_mode = validation.optional_string(params, "approval_mode")
        if initial_mode is not None:
            validation.require_feature(self, "new_session_settings", "approval_mode")
            if initial_model is None:
                initial_model = validation.required_string(params, "model")
            model_selection.validate(initial_model, initial_mode)
        candidate = host_commands(self, params)
        async with _session_swap(self, "session_park" in self.features, controller) as (incoming, leaving):
            cwd = None
            if "cwd" in params:
                validation.require_feature(self, "session_cwd", "cwd")
                cwd = validation.directory(params, "cwd")
            if incoming is None:
                await controller.control._terminate_pending_approvals()
            await (incoming.control if incoming is not None else controller.control).runtime.create_session(
                provider=validation.optional_string(params, "provider"),
                model=initial_model,
                cwd=cwd,
            )
            if incoming is not None:
                if self.server.controllers.controller_mode == "multi":
                    acquisition = self.server.controllers.reserve_session(
                        self, incoming.runtime.session_id, incoming
                    )
                    controller = incoming
                else:
                    controller = self.server.attach(incoming, self)
            if initial_model is not None and initial_mode is not None:
                await model_selection.apply_rpc(
                    controller.runtime, initial_model, initial_mode
                )
            if bindings is not None:
                headers, configs = bindings
                # assign the control's binding state before the await so a cancel
                # inside set_mcp_bindings still lets the release clear the loop (F6).
                controller.control._active_mcp_bindings = headers
                controller.control._mcp_binding_servers = frozenset(headers)
                controller.control._mcp_binding_required = bool(headers)
                await controller.runtime.loop.set_mcp_bindings(headers, configs)
            elif controller.runtime.loop is not None:
                controller.control._active_mcp_bindings = {}
                controller.control._mcp_binding_servers = frozenset()
                controller.control._mcp_binding_required = False
            controller.control._active_host_commands = candidate
            controller.control._host_binding_required = (
                "session_host_commands" in self.features
            )
            controller.control.session_replaced()
            if self.server._retention_owner is not None:
                self.server._retention_owner.set_session_available()
                self.server._retention_owner.start()
            controller.control.clear_approvals()
            await controller.control.controller_ready()
            await _publish_project_registry_notice(self, controller)
            await controller.control.publish_session_grant_reset()
            snapshot = {"session": session_snapshot(self, controller)}
            if acquisition is not None:
                self.server.controllers.commit_session(acquisition).release()
                await leaving.leave(self)
            return snapshot
    except BaseException:
        if acquisition is not None:
            closed = self.server.controllers.abort_session(acquisition)
            if closed is not None:
                await asyncio.shield(closed)
        _clear_unopened_host_commands(controller)
        raise


def _clear_unopened_host_commands(controller: SessionUse | SessionController) -> None:
    if controller.runtime.state is None:
        controller.control._active_host_commands = None
        controller.control._host_binding_required = False


async def _resume(
    self, params: dict[str, Any], controller: SessionUse | SessionController
) -> dict[str, object]:
    resident = None
    try:
        bindings = await mcp_bindings(self, params, controller)
        state = controller.runtime.state
        previous_session_id = state.metadata.session_id if state is not None else None
        commands = host_commands(self, params)
        park = "session_park" in self.features
        session_id = None
        if self.server.controllers.controller_mode == "multi":
            session_id = await _resolve_session_id(self, params, controller)
            resident = await _multi_resident(self, params, session_id, controller)
        else:
            attached, park = await _resume_live_or_parked(
                self, params, bindings, commands, park, controller
            )
            if attached is not None:
                return attached
    except BaseException:
        _clear_unopened_host_commands(controller)
        raise
    if resident is not None:
        return await _resume_resident(self, session_id, resident, bindings, commands, controller)
    return await _resume_reopen(
        self, params, session_id, bindings, commands, park, previous_session_id, controller
    )


async def _multi_resident(
    self, params: dict[str, Any], session_id: str, controller: SessionUse | SessionController
) -> SessionController | None:
    """Return the healthy resident record a multi resume takes in place (one
    record per session id); retire an unhealthy one so the resume reopens it."""
    controllers = self.server.controllers
    await controllers.wait_closed(session_id)
    exact_attach = "resume_attach" in self.features and params["session_id"] == session_id
    resident = controllers.acquisition_target(self, session_id, reconnect=exact_attach)
    if resident is None:
        return None
    if resident.control.unhealthy():
        if resident.control is not controller.control and controllers.owner_of(resident) in (None, self):
            # a parked one also gets its closed row, as on the legacy path
            self.server.parking.close(resident.control)
            await asyncio.shield(controllers.retire_for_reopen(resident))
        return None
    if "resume_attach" not in self.features and resident.control.state_change_busy():
        raise ProtocolError(ErrorDescriptor.OPERATION_BUSY, "a turn is already running")
    return resident


async def _resume_resident(
    self,
    session_id: str,
    resident: SessionController,
    bindings: tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None,
    commands: SessionHostCommands,
    controller: SessionUse | SessionController,
) -> dict[str, object]:
    """Attach this multi controller to the session's resident record."""
    acquisition = None
    target_use = None
    resident_claimed = False
    try:
        with _outgoing(self, session_id, resident) as leaving:
            acquisition = self.server.controllers.reserve_existing(self, session_id)
            target_use = self.server.controllers.use_acquisition(acquisition)
            if acquisition.previous is not acquisition.target:
                resident_claimed = True
                claimed = await target_use.control.controller_claimed(
                    reconnect="approval_reconnect" in self.features,
                    require_host_binding="session_host_commands" in self.features,
                    deadline=control_module.CONTROLLER_HANDOFF_DEADLINE_SECONDS,
                )
                if not claimed:
                    raise ProtocolError(
                        ErrorDescriptor.CONNECTION_CAPACITY, "another client is already connected"
                    )
            attached = await _attach_live_turn(
                self, bindings, commands, controller=target_use
            )
            if acquisition.previous is not acquisition.target:
                await target_use.control.publish_session_grant_reset()
            self.server.controllers.commit_session(acquisition).release()
            resident_claimed = False
            await leaving.leave(self)
            return attached
    except BaseException:
        deferred_cancel = False
        if resident_claimed and acquisition is not None:
            deferred_cancel = await _settle_controller_release(
                self, acquisition.target.control
            )
        if acquisition is not None:
            closed = self.server.controllers.abort_session(acquisition)
            if closed is not None:
                await asyncio.shield(closed)
        _clear_unopened_host_commands(controller)
        if deferred_cancel:
            raise asyncio.CancelledError
        raise
    finally:
        if target_use is not None:
            target_use.release()


async def _resume_reopen(
    self,
    params: dict[str, Any],
    session_id: str | None,
    bindings: tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None,
    commands: SessionHostCommands,
    park: bool,
    previous_session_id: str | None,
    controller: SessionUse | SessionController,
) -> dict[str, object]:
    """Open the session from disk, in place or in a new pair."""
    acquisition = None
    multi = self.server.controllers.controller_mode == "multi"
    try:
        async with _session_swap(self, park, controller, session_id) as (incoming, leaving):
            if session_id is None:
                session_id = await _resolve_session_id(self, params, controller)
                await self.server.controllers.wait_closed(session_id)
            if multi:
                acquisition = self.server.controllers.reserve_session(
                    self, session_id, incoming or controller.record
                )
            unhealthy_same_session = (
                incoming is None
                and controller.control.unhealthy()
                and controller.runtime.session_id == session_id
            )
            if incoming is None:
                await controller.control._terminate_pending_approvals()
            runtime = (incoming.control if incoming is not None else controller.control).runtime
            if unhealthy_same_session:
                try:
                    await runtime.resume_session(session_id, force=True)
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
                    await runtime.resume_session(session_id)
                except SessionOwnedError as exc:
                    raise session_owned_protocol_error(exc) from exc
                except SessionNotFoundError as exc:
                    raise ProtocolError(
                        ErrorDescriptor.SESSION_NOT_FOUND,
                        str(exc),
                        {"session_id": session_id},
                    ) from exc
            if incoming is not None:
                controller = incoming if multi else self.server.attach(incoming, self)
            if bindings is not None:
                headers, configs = bindings
                # assign the control's binding state before the await so a cancel
                # inside set_mcp_bindings still lets the release clear the loop (F6).
                controller.control._active_mcp_bindings = headers
                controller.control._mcp_binding_servers = frozenset(headers)
                controller.control._mcp_binding_required = bool(headers)
                await controller.runtime.loop.set_mcp_bindings(headers, configs)
            elif previous_session_id != session_id:
                controller.control._active_mcp_bindings = {}
                controller.control._mcp_binding_servers = frozenset()
                controller.control._mcp_binding_required = False
            controller.control._active_host_commands = commands
            controller.control._host_binding_required = (
                "session_host_commands" in self.features
            )
            controller.control.session_replaced()
            if self.server._retention_owner is not None:
                self.server._retention_owner.set_session_available()
                self.server._retention_owner.start()
            controller.control.clear_approvals()
            await controller.control.controller_ready()
            await _publish_project_registry_notice(self, controller)
            await controller.control.publish_session_grant_reset()
            snapshot = {"session": session_snapshot(self, controller)}
            if acquisition is not None:
                self.server.controllers.commit_session(acquisition).release()
                await leaving.leave(self)
            return snapshot
    except BaseException:
        if acquisition is not None:
            closed = self.server.controllers.abort_session(acquisition)
            if closed is not None:
                await asyncio.shield(closed)
        _clear_unopened_host_commands(controller)
        raise


async def _resume_live_or_parked(
    self,
    params: dict[str, Any],
    bindings: tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None,
    candidate: SessionHostCommands,
    park: bool,
    controller: SessionUse | SessionController,
) -> tuple[dict[str, object] | None, bool]:
    """Attach to the open session's busy turn or to a parked pair and return the
    result; else return None and whether the reopen takes a new pair."""
    # a wake in setup counts too: resuming the open session attaches, never swaps
    if "resume_attach" not in self.features or not (
        park or controller.control.state_change_busy()
    ):
        return None, park
    session_id = await _resolve_session_id(self, params, controller)
    parking = self.server.parking
    await self.server.controllers.wait_closed(session_id)
    state = controller.runtime.state
    control = controller.control
    if (
        state is not None
        and state.session_id == session_id
        and control.state_change_busy()
        and not control._session_swap_active
        and not control.unhealthy()
    ):
        return await _attach_live_turn(self, bindings, candidate, controller=controller), park
    parked = parking.get(session_id)
    if park and parked is not None:
        if not parked.control.unhealthy():
            return await _attach_live_turn(self, bindings, candidate, parked=parked, controller=controller), park
        # as for the open session: close it (its turn ends disconnect), reopen from disk
        parking.close(parked.control)
        await self.server.controllers.wait_closed(session_id)
    # resuming the attached session itself never takes a new pair
    return None, park and (state is None or state.session_id != session_id)


async def _resolve_session_id(
    self, params: dict[str, Any], controller: SessionUse | SessionController
) -> str:
    session_id = validation.required_string(params, "session_id")
    try:
        return await asyncio.to_thread(
            controller.runtime.manager.resolve_id,
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


async def _attach_live_turn(
    self,
    bindings: tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None,
    candidate: SessionHostCommands,
    *,
    parked: SessionController | None = None,
    controller: SessionUse | SessionController,
) -> dict[str, object]:
    """Attach to the active session's busy turn, or to a ``parked`` pair, without
    a reopen: the turn, its approvals and children stay; only this connection's
    bindings are re-supplied."""
    control = (parked or controller).control
    reattach = None
    rebound = None
    if bindings is not None:
        headers, configs = bindings
        # an idle session takes new bindings, as a reopen does
        idle = not control.holds_work()
        if idle:
            rebound = frozenset(headers)
        elif (
            not control._mcp_binding_required
            or frozenset(headers) != control._mcp_binding_servers
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "mcp_bindings must match the active session",
            )
        if not control._active_mcp_bindings or (idle and control._active_mcp_bindings != headers):
            reattach = (headers, configs)
        elif control._active_mcp_bindings != headers:
            # A running tool call may hold the current headers; swap them only when idle.
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY,
                "mcp_bindings cannot change during a live turn",
            )
    if parked is not None:
        # one step after the last await: the pair cannot start closing first
        controller = self.server.attach(parked, self)
    # Bind host commands before the await, so a turn that ends meanwhile never
    # sees a half-attached session. The running turn keeps its snapshot.
    with control.hold_binding_transition():
        previous = (control._active_host_commands, control._host_binding_required)
        control._active_host_commands = candidate
        control._host_binding_required = "session_host_commands" in self.features
        try:
            if reattach is not None:
                await control.reattach_mcp_bindings(*reattach)
            if rebound is not None:
                # with the new headers bound: a failed rebind keeps the old server set
                control._mcp_binding_servers = rebound
                control._mcp_binding_required = bool(rebound)
            if parked is not None:
                # the attached session changed, as on any resume
                await control.publish_session_grant_reset()
            if not control.state_change_busy():
                # the turn ended during the rebind, or a parked pair only held background
                # work: this is a plain resume of an idle session
                await control.controller_ready()
                return {"session": session_snapshot(self, controller)}
            return {"session": session_snapshot(self, controller), "attached": "live_turn"}
        except BaseException:
            control._active_host_commands, control._host_binding_required = previous
            raise


async def mcp_bindings(
    self, params: dict[str, Any], controller: SessionUse | SessionController
) -> tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None:
    if "mcp_bindings" not in params:
        return None
    validation.require_feature(self, "session_mcp_bindings", "mcp_bindings")
    parsed = parse_mcp_bindings(params["mcp_bindings"])
    configs = await resolve_mcp_bindings(self, parsed, params.get("cwd"), controller)
    return parsed, configs


async def reconnect_mcp_bindings(
    self, params: dict[str, Any], controller: SessionUse | SessionController
) -> tuple[dict[str, dict[str, str]], dict[str, MCPServerConfig]] | None:
    if "mcp_bindings" not in params:
        return None
    validation.require_feature(self, "approval_reconnect", "mcp_bindings")
    bindings = await mcp_bindings(self, params, controller)
    assert bindings is not None
    session_id = validation.required_string(params, "session_id")
    state = controller.runtime.state
    control = controller.control
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
    self,
    bindings: dict[str, dict[str, str]],
    project_dir: object,
    controller: SessionUse | SessionController,
) -> dict[str, MCPServerConfig]:
    runtime = controller.runtime
    home_path = home_config_path(runtime.home)
    loop = runtime.loop
    project_dir = (
        loop.tool_registry.cwd
        if loop is not None and project_dir is None
        else project_dir if project_dir is not None else runtime.cwd
    )
    config, effective = await asyncio.gather(
        asyncio.to_thread(load_mcp_config, home_path),
        asyncio.to_thread(
            load_mcp_config_overlay,
            home=runtime.home,
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


def session_snapshot(self, controller: SessionUse | SessionController) -> dict[str, object] | None:
    runtime = controller.runtime
    if runtime.opened is None:
        return None
    snapshot = _active_runtime_snapshot(runtime.metadata.to_dict(), runtime.loop)
    if runtime.policy is not None:
        snapshot["approval_mode"] = runtime.policy.default.value
    return snapshot


def session_runtime_state(self, controller: SessionUse | SessionController) -> str:
    return controller.control.pair_state()


async def live_status_baseline(self, controller: SessionUse | SessionController) -> dict[str, object]:
    pending, truncated = await controller.control.reconnect_baseline()
    return {
        "state": session_runtime_state(self, controller),
        "pending_approvals": pending,
        "pending_approvals_truncated": truncated,
    }


async def status(
    self,
    *,
    include_connection_setup_counts: bool = False,
    fallback_config: object | None = None,
    controller: SessionUse | SessionController,
) -> dict[str, object]:
    runtime = controller.runtime
    session = session_snapshot(self, controller)
    reconnect = "approval_reconnect" in self.features
    if reconnect:
        baseline = await live_status_baseline(self, controller)
        state = baseline["state"]
        pending_approvals = baseline["pending_approvals"]
        pending_approvals_truncated = baseline["pending_approvals_truncated"]
    else:
        state = session_runtime_state(self, controller)
        pending_approvals = await controller.control.pending_approvals()
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
    if "session_park" in self.features:
        status["parked_sessions"] = self.server.parking.rows(self)
    if "telemetry_read_v1" in self.features:
        status["telemetry"] = live_view(self.server.telemetry)
    slash_command = controller.control.slash_lock.status()
    if slash_command is not None:
        status["slash_command"] = slash_command
    return status


def list_sessions(
    self, request_id: str | int, params: dict[str, Any], resident: frozenset[str]
) -> tuple[dict[str, object], list[str]]:
    return _list_sessions(self, request_id, params, resident)


def _list_sessions(
    self, request_id: str | int, params: dict[str, Any], resident: frozenset[str]
) -> tuple[dict[str, object], list[str]]:
    paging = "list_sessions_paging" in self.features
    for name in ("offset", "limit"):
        if name in params:
            validation.require_feature(self, "list_sessions_paging", name)
    offset = validation.integer(params, "offset", 0, minimum=0)
    limit = validation.integer(params, "limit", None, minimum=1)
    if "project_id" in params:
        validation.require_feature(self, "projects", "project_id")
        assert self.server.projects is not None
        available = self.server.projects.project_sessions(params["project_id"])
    else:
        available = self.server.manager.list_sessions_read_only()
    metadata = available[offset : None if limit is None else offset + limit]
    previews = {
        item.session_id: item
        for item in self.server.manager.list_session_previews(
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
                "resident": item.session_id in resident,
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
            item.pop("resident")
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
