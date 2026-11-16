from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import logging
import os
import socket
import stat
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from ..config.settings import load_settings
from ..core.session import SessionError
from ..core.store import ConversationIntegrityError, TranscriptLoadError
from ..core.transcript_recovery.runtime import TranscriptRepairRequiresUpgrade
from ..locking import socket_lock
from ..project_errors import ProjectRegistryBusyError
from ..runtime.foreground_events import ForegroundEventPublisher
from ..session_retention.activation import make_retention_owner
from ..storage_format import (
    StorageFeature,
    drain_releases,
    feature_enabled,
    pending_releases,
)
from . import (
    accounts,
    login,
)
from . import control as control_module
from .connections import ConnectionRegistry
from .control import (
    NotificationPresentation,
    ServeControl,
)
from .diagnostics import capture_peer_identity, log_stuck_holder
from .handlers.projects import (
    PROJECT_REQUEST_EXCEPTIONS,
    ProjectRequests,
    project_request_error,
)
from .handlers.registry import dispatch as _dispatch_request
from .parking import ParkedPairs
from .protocol import (
    CHILD_RUN_EVENTS_PREREQUISITES,
    FEATURES,
    MAX_FRAME_BYTES,
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    FrameWriteOwner,
    MethodDescriptor,
    ProtocolError,
    _Frame,
    hello_result,
    negotiate_protocol_version,
)
from .runtime import BackendFactory, ServerRuntime
from .session_controllers import SessionController, SessionControllers
from .transport import WebSocketServer
from .turn_context import PendingTurnContexts

logger = logging.getLogger(__name__)

# A GC finalizer only queues a lease release; an idle serve must still run it,
# or another process stays refused the session until serve's next lease call.
IDLE_DRAIN_INTERVAL = 1.0


async def _drain_released_leases() -> None:
    while True:
        await asyncio.sleep(IDLE_DRAIN_INTERVAL)
        if not pending_releases():
            continue
        try:
            # a release may wait on a registry lock another thread holds
            await asyncio.to_thread(drain_releases)
        except Exception:  # a failed drain must not stop serve
            logger.exception("draining released leases failed")


class _UnixSocketListener:
    def __init__(
        self,
        path: Path,
        server: asyncio.AbstractServer,
        lock_fd: int,
        identity: tuple[int, int],
    ) -> None:
        self.path = path
        self.server = server
        self._lock_fd = lock_fd
        self._identity = identity

    @classmethod
    async def start(
        cls,
        path: Path,
        accept: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
    ) -> _UnixSocketListener:
        cls._prepare_parent(path)
        lock_fd = os.open(
            f"{path}.lock",
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            try:
                with socket_lock():
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"a server is already listening on {path}") from exc
            cls._prepare_path(path)
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            identity: tuple[int, int] | None = None
            try:
                unix_socket.bind(str(path))
                socket_stat = path.lstat()
                identity = (socket_stat.st_dev, socket_stat.st_ino)
                server = await asyncio.start_unix_server(
                    accept, sock=unix_socket, limit=MAX_FRAME_BYTES + 1
                )
                os.chmod(path, 0o600)
                return cls(path, server, lock_fd, identity)
            except BaseException:
                unix_socket.close()
                if identity is not None:
                    cls._unlink_if_owned(path, identity)
                raise
        except BaseException:
            os.close(lock_fd)
            raise

    async def close(self) -> None:
        try:
            self.server.close()
            await self.server.wait_closed()
            self._unlink_if_owned(self.path, self._identity)
        finally:
            os.close(self._lock_fd)

    @staticmethod
    def _prepare_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            parent_stat = path.parent.stat()
            if parent_stat.st_uid == os.getuid():
                os.chmod(path.parent, 0o700)
        except OSError:
            pass

    @classmethod
    def _prepare_path(cls, path: Path) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_stat.st_mode):
            raise RuntimeError(f"refusing to replace non-socket path: {path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise RuntimeError(f"could not inspect socket: {path}") from exc
        else:
            raise RuntimeError(f"a server is already listening on {path}")
        finally:
            probe.close()
        cls._unlink_if_owned(path, (path_stat.st_dev, path_stat.st_ino))

    @staticmethod
    def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return
        if (path_stat.st_dev, path_stat.st_ino) == identity:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


class ZetaServer:
    """Serve one controller and negotiated observers over local transports."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        cwd: str | Path | None = None,
        socket_path: str | Path | None = None,
        port: int | None = None,
        ws_port: int | None = None,
        ws_discovery: str | Path | None = None,
        ws_origins: tuple[str, ...] = (),
        provider: str | None = None,
        model: str | None = None,
        tools: str | None = None,
        disallowed_tools: str | None = None,
        require_tools: bool = False,
        allow_hooks: bool | None = None,
        auto_memory: bool | None = None,
        cli_yolo: bool | None = None,
        max_turns: int | None = None,
        approval_reconnect_grace: int = 90,
        backend_factory: BackendFactory | None = None,
        runtime: ServerRuntime | None = None,
        foreground_event_publisher: ForegroundEventPublisher | None = None,
        max_role: str = "controller",
        prebound_socket: socket.socket | None = None,
    ) -> None:
        if socket_path is not None and port is not None:
            raise ValueError("choose socket_path or port, not both")
        if port is not None and not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if ws_port is None and ws_origins:
            raise ValueError("ws_origins requires ws_port")
        if ws_port is None and ws_discovery is not None:
            raise ValueError("ws_discovery requires ws_port")
        if ws_discovery is not None:
            ws_discovery = Path(ws_discovery).expanduser()
            if not ws_discovery.is_absolute():
                raise ValueError("ws_discovery must be an absolute path")
        self.home = Path(home).expanduser().resolve() if home is not None else _home()
        self.socket_path = (
            Path(socket_path).expanduser().resolve()
            if socket_path is not None
            else self.home / "run" / "serve.sock"
        )
        self.port = port
        self.approval_reconnect_grace = approval_reconnect_grace
        # deadlines for a controller connection lost mid-request (tests override).
        # drain this long once another controller contends, then cancel the holder.
        self.lost_request_contended_drain = 5.0
        # after the cancel, log the stuck-holder task dump at this interval.
        self.lost_request_cancel_bound = 1.0
        self._borrowed = runtime is not None
        runtime = runtime or ServerRuntime(
            self.home,
            cwd=cwd,
            provider=provider,
            model=model,
            tools=tools,
            disallowed_tools=disallowed_tools,
            require_tools=require_tools,
            allow_hooks=allow_hooks,
            auto_memory=auto_memory,
            cli_yolo=cli_yolo,
            max_turns=max_turns,
            backend_factory=backend_factory,
        )
        self._max_role = max_role
        self._foreground_event_publisher = foreground_event_publisher
        self._foreground_subscription = None
        self._foreground_task: asyncio.Task[None] | None = None
        self._server: asyncio.AbstractServer | None = None
        discovery_path = (
            Path(ws_discovery)
            if ws_discovery is not None
            else self.home / "run" / "serve.ws.json"
        )
        discovery_lock_path = (
            discovery_path.with_name(discovery_path.name + ".lock")
            if ws_discovery is not None
            else self.home / "run" / "serve.ws.lock"
        )
        self._websocket = (
            WebSocketServer(
                port=ws_port,
                origins=ws_origins,
                path=discovery_path,
                lock_path=discovery_lock_path,
                secure_parent=ws_discovery is None,
                accept=self._accept,
            )
            if ws_port is not None
            else None
        )
        self._client_active = False
        self._client: _Client | None = None
        self.connections: ConnectionRegistry | None = None
        self._retention_owner = (
            make_retention_owner(
                self.home,
                runtime.manager._storage_lease,
                lambda: (
                    0
                    if self.controllers.attached().runtime.opened is None
                    else load_settings(home=self.home).settings.retention_days
                ),
                session_available=False,
            )
            if (
                not self._borrowed
                and feature_enabled(
                    runtime.manager._storage_lease.observed_epoch,
                    StorageFeature.SESSION_RETENTION_ACTIVATION,
                )
            )
            else None
        )
        self._account_warnings: tuple[str, ...] = ()
        self._unix_listener: _UnixSocketListener | None = None
        self._prebound_socket = prebound_socket
        self._frame_writes = FrameWriteOwner()
        self._idle_drain: asyncio.Task[None] | None = None
        self.notification_presentations = NotificationPresentation()
        self.turn_contexts = PendingTurnContexts()
        self.foreground_event_publisher = foreground_event_publisher or ForegroundEventPublisher()
        control = ServeControl(self, runtime, self.foreground_event_publisher)
        self.controllers = SessionControllers(control)
        self._bind_pair(control)
        self.parking = ParkedPairs(lambda: self.connections, self.controllers)

    @property
    def _control_for_test(self) -> ServeControl:
        return self.controllers.attached().control

    @property
    def _runtime_for_test(self) -> ServerRuntime:
        return self.controllers.attached().runtime

    def controls(self) -> Iterator[ServeControl]:
        """The attached pair, then every parked pair."""
        yield self.controllers.attached().control
        yield from self.parking.pairs()

    def new_pair(self) -> SessionController:
        """An empty pair beside the attached one; it shares the session manager."""
        self.parking.require_room()
        control = ServeControl(
            self, self.controllers.attached().runtime.sibling(), self.foreground_event_publisher
        )
        self._bind_pair(control)
        return self.controllers.register(control)

    def attach(self, incoming: SessionController) -> SessionController:
        """Make ``incoming`` the attached pair in one step (no await): park the
        outgoing pair while it holds work, else start its close."""
        outgoing = self.controllers.attached().control
        if not self.parking.unpark(incoming.control):
            # a new pair takes over the controller slot state of the pair it replaces
            incoming.control._controller_attached = outgoing._controller_attached
        self.controllers.attach(incoming)
        assert self.connections is not None
        self.connections.replay_session_replaced()
        if outgoing.runtime.opened is not None:
            self.parking.leave(outgoing)
        return incoming

    async def require_all_idle(self) -> None:
        for control in self.controls():
            await control.require_idle()

    def _bind_pair(self, control: ServeControl) -> None:
        """Point the runtime's sinks at its control; a borrowed runtime keeps its owner's."""
        if not self._borrowed:
            runtime = control.runtime
            runtime.set_approval_control_binder(control.bind_policy)
            runtime.set_background_event_sink(control._publish_background_event)
            runtime.set_memory_notice_sink(control._publish_memory_notice)
            runtime.set_run_work_event_sink(control._publish_run_work_event)
            runtime.set_background_wake_sink(control._schedule_background_wake)
            runtime.set_work_changed_sink(
                lambda: self.parking.changed_from_any_thread(control)
            )

    async def _publish_foreground_events(self) -> None:
        assert self._foreground_subscription is not None
        async for event in self._foreground_subscription:
            await self.controllers.attached().control._event(
                event, session_id=self.controllers.attached().runtime.session_id, turn_id=None
            )

    async def _close_foreground_events(self) -> None:
        # a borrowed publisher belongs to the TUI and outlives this server
        if self._foreground_event_publisher is None:
            self.foreground_event_publisher.close()
            return
        subscription, self._foreground_subscription = self._foreground_subscription, None
        if subscription is not None:
            self._foreground_event_publisher.unsubscribe(subscription)
        task, self._foreground_task = self._foreground_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def publish_update(self, old_revision: str, new_revision: str) -> None:
        if self.connections is not None:
            await self.connections.publish_feature_event(
                "zeta_updated",
                EventDescriptor.ZETA_UPDATED,
                self.controllers.attached().runtime.session_id if self.controllers.attached().runtime.opened is not None else None,
                data={"old_revision": old_revision, "new_revision": new_revision},
            )

    @property
    def address(self) -> str:
        if self.port is not None:
            return f"127.0.0.1:{self.port}"
        return str(self.socket_path)

    async def start(self) -> None:
        if not self._borrowed:
            await self.controllers.attached().runtime.initialize_project_registry()
            self._account_warnings = await accounts.migrate(self.home)
        self.connections = ConnectionRegistry(
            asyncio.get_running_loop(), lambda: self.controllers.attached().control.in_flight
        )
        if self.port is None:
            if self._prebound_socket is not None:
                self._server = await asyncio.start_unix_server(
                    self._accept, sock=self._prebound_socket, limit=MAX_FRAME_BYTES + 1
                )
            else:
                self._unix_listener = await _UnixSocketListener.start(
                    self.socket_path, self._accept
                )
                self._server = self._unix_listener.server
        else:
            self._server = await asyncio.start_server(
                self._accept, "127.0.0.1", self.port, limit=MAX_FRAME_BYTES + 1
            )
            bound = self._server.sockets
            if bound:
                self.port = int(bound[0].getsockname()[1])
        if self._websocket is not None:
            try:
                await self._websocket.start()
            except BaseException:
                await self.close()
                raise
        self._idle_drain = asyncio.create_task(_drain_released_leases())
        if self._foreground_event_publisher is not None:
            self._foreground_subscription = self._foreground_event_publisher.subscribe()
            self._foreground_task = asyncio.create_task(self._publish_foreground_events())
        if self._retention_owner is not None:
            async def publish_retention(projection: dict[str, object]) -> None:
                if projection.get("mode") == "none" and not projection.get("errors"):
                    return
                if self.connections is None or self.controllers.attached().runtime.opened is None:
                    return
                await self.connections.publish_feature_event(
                    "session_retention", EventDescriptor.SESSION_RETENTION,
                    self.controllers.attached().runtime.opened.metadata.session_id, data=projection,
                )

            self.controllers.attached().runtime.retention_event_sink = publish_retention
            self.controllers.attached().runtime.retention_wake = self._retention_owner.wake
            self._retention_owner.set_publish(publish_retention)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self, *, approval_settlement_timeout: float | None = None) -> None:
        if self._retention_owner is not None:
            await self._retention_owner.stop()
        if self._idle_drain is not None:
            self._idle_drain.cancel()
            await asyncio.wait([self._idle_drain])
            self._idle_drain = None
        try:
            if self._client is not None:
                await self._client.close(graceful=True)
        finally:
            self._client = None
            self._client_active = False
            self.parking.close_all(approval_settlement_timeout)
            try:
                try:
                    await self.controllers.attached().control.close()
                finally:
                    await self._close_foreground_events()
                try:
                    if not self._borrowed:
                        await self.controllers.attached().runtime.close()
                finally:
                    try:
                        await self.controllers.attached().control.wait_for_approval_settlement(
                            approval_settlement_timeout
                        )
                    finally:
                        await asyncio.gather(
                            *self.parking.closing_tasks(), return_exceptions=True
                        )
            finally:
                if self.connections is not None:
                    await self.connections.aclose()
                    self.connections = None
                await self._frame_writes.aclose()
                try:
                    if self._websocket is not None:
                        await self._websocket.close()
                    if self._unix_listener is not None:
                        await self._unix_listener.close()
                    elif self._server is not None:
                        self._server.close()
                        await self._server.wait_closed()
                finally:
                    self._server = None
                    self._unix_listener = None

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        registry = self.connections
        assert registry is not None
        client = _Client(self, reader, writer)
        try:
            registry.register(client)
        except RuntimeError:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        try:
            await client.run()
        finally:
            if self._client is client:
                self._client = None
                self._client_active = False
            registry.release(client)


class _Client:
    def __init__(
        self,
        server: ZetaServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.server = server
        self.reader = reader
        self.writer = writer
        if server.connections is None:
            server.connections = ConnectionRegistry(
                asyncio.get_running_loop(), lambda: server.controllers.attached().control.in_flight
            )
            server.connections.register(self)
        self.connections = server.connections
        self.logins = login.Logins(server.home)
        self.handshaken = False
        self.role: str | None = None
        self.max_role = getattr(reader, "max_role", server._max_role)
        self.protocol_version = "1.0"
        self.features: frozenset[str] = frozenset()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._local_close = False
        self._controller_release: asyncio.Future[object] | None = None
        self._transport_failed = False
        self._read_buffer = bytearray()
        # the request this connection runs now (method, id, monotonic start), its
        # handler task, and the one pending socket read; used to detect loss
        # while a handler runs and to dump a stuck holder. A contended-slot
        # refusal log also reads these plus the peer facts captured once at
        # accept and the connect/last-frame clocks to name and age both the
        # holder and the refused connection (no behavior change).
        self._request: tuple[str, str | int, float] | None = None
        self._run_task: asyncio.Task[object] | None = None
        self._chunk_read: asyncio.Task[bytes] | None = None
        self._peer = capture_peer_identity(reader, writer)
        self._connected_at = time.monotonic()
        self._last_frame_at = self._connected_at
        self._client_version: str | None = None
        self.codec = FrameCodec()
        self.projects = ProjectRequests.for_runtime(
            home=server.home, runtime=server.controllers.attached().runtime, codec=self.codec
        )

    async def run(self) -> None:
        graceful = True
        try:
            while True:
                line = await self._read_frame()
                if not line:
                    self._release_dead_connection()
                    graceful = False
                    break
                self._last_frame_at = time.monotonic()
                try:
                    request = self.codec.parse_request(line)
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id, exc.descriptor, exc.message, exc.data
                        )
                    )
                    if not self.handshaken:
                        break
                    continue
                handling = asyncio.create_task(self._handle(request))
                self._run_task = handling
                if await self._run_request(handling):
                    await self._handle_loss(handling)
                    graceful = False
                    break
                self._run_task = None
                if not self.handshaken:
                    break
        finally:
            # a cancel of the connection task (server teardown) must also stop an
            # in-flight handler; the EOF-loss path already awaited it to done.
            handler = self._run_task
            if handler is not None and not handler.done():
                handler.cancel()
                with contextlib.suppress(BaseException):
                    await handler
            self._cancel_pending_read()
            await self.close(graceful=graceful and not self._transport_failed)

    async def _run_request(self, handling: asyncio.Task[None]) -> bool:
        """Run one request to completion while reading the socket ahead.

        Dispatch stays serial: the next request is not parsed until this one
        ends. Only the socket read runs concurrently so a peer that closes mid
        request is seen at once. Return ``True`` if the peer closed (EOF).
        """
        while not handling.done():
            if len(self._read_buffer) > MAX_FRAME_BYTES:
                # read-ahead bound reached (one max frame); wait for the handler.
                await asyncio.wait({handling})
                break
            chunk = self._ensure_chunk_read()
            await asyncio.wait({handling, chunk}, return_when=asyncio.FIRST_COMPLETED)
            if chunk.done():
                self._chunk_read = None
                data = chunk.result()
                if not data:
                    return True
                self._read_buffer.extend(data)
        await handling
        return False

    async def _handle_loss(self, handling: asyncio.Task[None]) -> None:
        """Peer EOF while a request runs: start no new request, hold the slot
        until the handler ends, and cancel it only once another controller
        contends for the slot."""
        if self._local_close:
            # a local server close cancels the handler at once and releases via
            # close(); no remote hello waits on it.
            handling.cancel()
            with contextlib.suppress(BaseException):
                await handling
            return
        contended = self.connections.mark_lost(self)
        if contended is None:
            with contextlib.suppress(BaseException):
                await handling
            return
        contended_wait = asyncio.create_task(contended.wait())
        try:
            await asyncio.wait(
                {handling, contended_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if not handling.done():
                # a controller hello is waiting: drain briefly so a handler that
                # is about to finish keeps its side effects, then cancel.
                await asyncio.wait(
                    {handling}, timeout=self.server.lost_request_contended_drain
                )
                if not handling.done():
                    handling.cancel()
                    await self._await_handler_held(handling)
        finally:
            contended_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await contended_wait
        with contextlib.suppress(BaseException):
            await handling
        self._release_dead_connection()

    async def _await_handler_held(self, handling: asyncio.Task[None]) -> None:
        """Hold the slot until the cancelled handler ends; dump the stuck holder
        at each cancel-bound interval so a swallowed cancel stays visible."""
        while True:
            done, _ = await asyncio.wait(
                {handling}, timeout=self.server.lost_request_cancel_bound
            )
            if done:
                return
            log_stuck_holder(
                self,
                self.connections.output_state(self),
                reason="controller slot held; lost holder's request did not end after cancel",
            )

    async def _handle(self, request: dict[str, Any]) -> None:
        request_id = request["id"]
        self._request = (request["method"], request_id, time.monotonic())
        await self.connections.wait_until_settled(self)
        try:
            result = await self._dispatch(
                request_id, request["method"], request["params"]
            )
        except ProtocolError as exc:
            await self._write(
                self.codec.error_response(
                    exc.request_id
                    if exc.request_id is not None
                    else request_id,
                    exc.descriptor,
                    exc.message,
                    exc.data,
                )
            )
            if not self.handshaken:
                self.connections.release(self)
        except PROJECT_REQUEST_EXCEPTIONS as exc:
            descriptor, message, data = project_request_error(exc)
            await self._write(
                self.codec.error_response(request_id, descriptor, message, data)
            )
        except ProjectRegistryBusyError as exc:
            await self._write(
                self.codec.error_response(
                    request_id,
                    ErrorDescriptor.PROJECT_REGISTRY_BUSY,
                    str(exc),
                    {"retryable": exc.retryable},
                )
            )
        except TranscriptRepairRequiresUpgrade as exc:
            await self._write(
                self.codec.error_response(
                    request_id,
                    ErrorDescriptor.TRANSCRIPT_REPAIR_REQUIRES_UPGRADE,
                    str(exc),
                    {
                        "home_epoch": exc.home_epoch,
                        "required_epoch": exc.required_epoch,
                    },
                )
            )
        except TranscriptLoadError as exc:
            await self._write(
                self.codec.error_response(
                    request_id,
                    ErrorDescriptor.CORRUPT_TRANSCRIPT,
                    str(exc),
                    exc.protocol_data(),
                )
            )
        except ConversationIntegrityError as exc:
            await self._write(
                self.codec.error_response(
                    request_id, ErrorDescriptor.INTERNAL_ERROR, str(exc)
                )
            )
        except (SessionError, ValueError) as exc:
            await self._write(
                self.codec.error_response(
                    request_id, ErrorDescriptor.INVALID_PARAMS, str(exc)
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep the socket alive
            await self._write(
                self.codec.error_response(
                    request_id, ErrorDescriptor.INTERNAL_ERROR, str(exc)
                )
            )
        else:
            if request["method"] in {"session_history_and_subscribe", "child_run_events"}:
                baseline_builder = (
                    self.connections.child_runs.with_baseline
                    if request["method"] in {"session_history_and_subscribe", "child_run_events"}
                    else None
                )
                written = await self.connections.admit_join_response(
                    self, request_id, result, baseline_builder
                )
            else:
                written = await self._write(self.codec.response(request_id, result))
            if (
                written
                and request["method"] == "hello"
                and self.role == "controller"
                and "approval_reconnect" in self.features
            ):
                for control in self.server.controls():
                    await control.controller_ready(reconnect=True)
        self._request = None

    async def _read_frame(self) -> bytes | None:
        """Read one line while retaining a bounded prefix of oversized input."""
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                if end <= MAX_FRAME_BYTES:
                    line = bytes(self._read_buffer[:end])
                    del self._read_buffer[:end]
                    return line
                prefix = bytes(self._read_buffer[:MAX_FRAME_BYTES])
                del self._read_buffer[:end]
                return prefix + b"\n"
            if len(self._read_buffer) > MAX_FRAME_BYTES:
                prefix = bytes(self._read_buffer[:MAX_FRAME_BYTES])
                self._read_buffer.clear()
                return await self._discard_oversized_frame(prefix)
            chunk = await self._read_chunk()
            if not chunk:
                if not self._read_buffer:
                    return None
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            self._read_buffer.extend(chunk)

    def _ensure_chunk_read(self) -> asyncio.Task[bytes]:
        """One pending socket read, shared by the run loop and read-ahead so a
        single reader never has two concurrent reads."""
        if self._chunk_read is None:
            self._chunk_read = asyncio.create_task(self._read_raw())
        return self._chunk_read

    async def _read_raw(self) -> bytes:
        try:
            return await self.reader.read(65_536)
        except ConnectionError:
            # A failed write to a gone peer stores its error in the reader; that is EOF.
            return b""

    async def _read_chunk(self) -> bytes:
        chunk = self._ensure_chunk_read()
        data = await chunk
        if self._chunk_read is chunk:
            self._chunk_read = None
        return data

    def _cancel_pending_read(self) -> None:
        if self._chunk_read is not None:
            self._chunk_read.cancel()
            self._chunk_read = None

    def peer_closed(self) -> bool:
        """Best-effort: has this connection's peer gone away (EOF / transport lost)?

        A heuristic for diagnostics; a buffered unread frame can hide EOF on a
        unix socket, which is acceptable here.
        """
        reader_peer_closed = getattr(self.reader, "peer_closed", None)
        if reader_peer_closed is not None:  # WebSocketConnection adapter
            return bool(reader_peer_closed())
        return self.reader.at_eof() or self.writer.is_closing()

    async def _discard_oversized_frame(self, prefix: bytes) -> bytes:
        """Discard an oversized line and preserve any complete later frames."""
        while True:
            chunk = await self._read_chunk()
            if not chunk:
                return prefix + b"\n"
            newline = chunk.find(b"\n")
            if newline < 0:
                continue
            self._read_buffer.extend(chunk[newline + 1 :])
            return prefix + b"\n"

    def _begin_controller_release(self) -> asyncio.Future[object] | None:
        if self._controller_release is not None:
            return self._controller_release
        if self.role == "controller" and self.connections.is_controller(self):
            self._controller_release = asyncio.gather(*(
                control.begin_controller_release(
                    reconnect="approval_reconnect" in self.features,
                    keep_turn="turn_reconnect" in self.features,
                )
                for control in self.server.controls()
            ))
        return self._controller_release

    def _release_dead_connection(self) -> None:
        if self._local_close:
            return
        self._begin_controller_release()
        self.connections.release(self)

    async def close(self, *, graceful: bool = False) -> None:
        async with self._close_lock:
            if self._closed:
                return
            release = self._begin_controller_release()
            if release is not None:
                await asyncio.shield(release)
            await accounts.cancel_owned_switch(self)
            await self.logins.close()
            if graceful:
                await self.connections.flush(self)
            self._local_close = True
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self.connections.release(self)
            self._closed = True

    async def _route_request(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> tuple[bool, object]:
        if method == "hello":
            # a controller hello waits up to one shared handoff deadline for a
            # lost holder to release the slot, then passes the time that is left
            # to controller_claimed so the total wait never exceeds one deadline.
            deadline = control_module.CONTROLLER_HANDOFF_DEADLINE_SECONDS
            if params.get("role", "controller") == "controller":
                start = time.monotonic()
                await self.connections.wait_for_lost_controller(deadline)
                deadline = max(0.0, deadline - (time.monotonic() - start))
            result = self._hello(params)
            reconnect_bindings = None
            if "mcp_bindings" in params:
                if self.role != "controller":
                    raise ProtocolError(
                        ErrorDescriptor.INVALID_PARAMS,
                        "mcp_bindings require a controller connection",
                    )
                from .handlers.lifecycle import reconnect_mcp_bindings

                reconnect_bindings = await reconnect_mcp_bindings(self, params)
            if self.role == "controller":
                reconnect = "approval_reconnect" in self.features
                require_host_binding = "session_host_commands" in self.features

                async def claim() -> bool:
                    return await self.server.controllers.attached().control.controller_claimed(
                        reconnect=reconnect,
                        require_host_binding=require_host_binding,
                        deadline=deadline,
                    )

                # shield the claim so a cancel of the hello does not leave the
                # approvals lifecycle half-moved; it waits for the bounded claim.
                claim_task = asyncio.ensure_future(claim())
                try:
                    claimed = await asyncio.shield(claim_task)
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await claim_task
                    raise
                if not claimed:
                    self.handshaken = False
                    raise ProtocolError(
                        ErrorDescriptor.CONNECTION_CAPACITY,
                        "another client is already connected",
                    )
                await self.server.parking.claim(
                    "session_park" in self.features,
                    reconnect=reconnect,
                    require_host_binding=require_host_binding,
                )
                if reconnect_bindings is not None:
                    await self.server.controllers.attached().control.reattach_mcp_bindings(
                        *reconnect_bindings
                    )
                self.server._client_active = True
                self.server._client = self
                self.connections.activate_controller(self)
                if not reconnect:
                    for control in self.server.controls():
                        await control.controller_ready()
            return True, result
        if not self.handshaken:
            raise ProtocolError(
                ErrorDescriptor.PROTOCOL_MISMATCH, "hello must be the first request"
            )
        return False, None

    _dispatch = _dispatch_request

    def _hello(self, params: dict[str, Any]) -> dict[str, object]:
        if self.handshaken:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST, "hello may only be sent once"
            )
        version = params.get("protocol_version")
        client_version = params.get("client_version")
        if isinstance(client_version, str):
            self._client_version = client_version
        self.protocol_version = negotiate_protocol_version(version, client_version)
        requested = params.get("features")
        if requested is not None and (
            not isinstance(requested, list)
            or not all(isinstance(item, str) for item in requested)
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "features must be an array of strings"
            )
        extended = self.protocol_version != "1.0"
        accepted = (
            [item for item in FEATURES if item in (requested or ())] if extended else []
        )
        accepted = self.server.controllers.attached().runtime.available_project_features(accepted)
        if "projects-memory-v2" in accepted and "projects" not in accepted:
            accepted.remove("projects-memory-v2")
        if "turn_reconnect" in accepted and "approval_reconnect" not in accepted:
            accepted.remove("turn_reconnect")
        if "in_flight_replay" in accepted and "session_replay_seq" not in accepted:
            accepted.remove("in_flight_replay")
        # a borrowed runtime cannot make a second pair
        if "session_park" in accepted and (
            "resume_attach" not in accepted or self.server._borrowed
        ):
            accepted.remove("session_park")
        if (
            "child_run_events" in accepted
            and not CHILD_RUN_EVENTS_PREREQUISITES <= set(accepted)
        ):
            accepted.remove("child_run_events")
        self.features = frozenset(accepted)
        assert self.server.connections is not None
        role = self.server.connections.negotiate_role(
            self,
            params.get("role", "controller"),
            self.protocol_version,
            self.features,
            max_role=self.max_role,
        )
        self.handshaken = True
        requests = MethodDescriptor.advertised(
            self.protocol_version, self.features, role
        )
        if extended:
            self.server.turn_contexts.add_request(requests, self.features)
        capabilities: dict[str, object] = {
            "requests": requests,
            "notifications": ["event"],
        }
        if extended and requested is not None:
            capabilities["features"] = accepted
        result = hello_result(
            self.protocol_version,
            capabilities,
            self.server.controllers.attached().runtime.project_registry_warning,
        )
        if self.server._account_warnings:
            notices = result.setdefault("notices", [])
            assert isinstance(notices, list)
            notices.extend(self.server._account_warnings)
        return result

    async def _write(self, payload: _Frame) -> bool:
        try:
            written = await self.connections.send(self, payload)
        except OSError:
            written = False
        if not written:
            self._transport_failed = True
        return written

    async def _write_physical(self, payload: _Frame) -> bool:
        return await self.server._frame_writes.write(
            self.codec, self.writer, payload, self._write_lock
        )


def _home() -> Path:
    from ..core.session import env_home

    return env_home().expanduser().resolve()


__all__ = ["ZetaServer"]
