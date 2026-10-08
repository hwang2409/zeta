"""Newline-delimited JSON-RPC server for native zeta frontends."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import signal
import socket
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.approval import ApprovalDecision
from ..core.session import SessionError, SessionNotFoundError
from ..protocol.types import (
    Message,
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from ..runtime.compaction_mode import switch_compaction
from . import abort_scope, ergonomics, login, model_selection, slash_commands
from .approval_lifecycle import ApprovalKey, ApprovalLifecycle
from .delivery import DeliveryRequests, TurnOutcome
from .model_inputs import PendingModelInputs
from .project_requests import (
    PROJECT_REQUEST_EXCEPTIONS,
    PROJECT_REQUESTS,
    ProjectRequests,
    project_request_error,
)
from .protocol import (
    FEATURES,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    FrameCodec,
    ProtocolError,
    bounded,
)
from .runtime import BackendFactory, ServerRuntime
from .turn_context import PendingTurnContexts

# Echoed user text stays well inside the 1 MiB frame after JSON escaping.
USER_MESSAGE_MAX_BYTES = 262_144


class ZetaServer:
    """Serve one local client over a Unix socket or localhost TCP."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        cwd: str | Path | None = None,
        socket_path: str | Path | None = None,
        port: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        compaction: str | None = None,
        tools: str | None = None,
        disallowed_tools: str | None = None,
        require_tools: bool = False,
        allow_hooks: bool | None = None,
        auto_memory: bool | None = None,
        cli_yolo: bool | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if socket_path is not None and port is not None:
            raise ValueError("choose socket_path or port, not both")
        if port is not None and not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.home = Path(home).expanduser().resolve() if home is not None else _home()
        self.socket_path = (
            Path(socket_path).expanduser().resolve()
            if socket_path is not None
            else self.home / "run" / "serve.sock"
        )
        self.port = port
        self.runtime = ServerRuntime(
            self.home,
            cwd=cwd,
            provider=provider,
            model=model,
            compaction=compaction,
            tools=tools,
            disallowed_tools=disallowed_tools,
            require_tools=require_tools,
            allow_hooks=allow_hooks,
            auto_memory=auto_memory,
            cli_yolo=cli_yolo,
            backend_factory=backend_factory,
        )
        self._server: asyncio.AbstractServer | None = None
        self._client_active = False
        self._client: _Client | None = None
        self._socket_created = False
        self.turn_contexts = PendingTurnContexts()

    @property
    def address(self) -> str:
        if self.port is not None:
            return f"127.0.0.1:{self.port}"
        return str(self.socket_path)

    async def start(self) -> None:
        if self.port is None:
            self._prepare_socket_path()
            self._server = await asyncio.start_unix_server(
                self._accept, path=str(self.socket_path), limit=MAX_FRAME_BYTES + 1
            )
            os.chmod(self.socket_path, 0o600)
            self._socket_created = True
        else:
            self._server = await asyncio.start_server(
                self._accept, "127.0.0.1", self.port, limit=MAX_FRAME_BYTES + 1
            )
            bound = self._server.sockets
            if bound:
                self.port = int(bound[0].getsockname()[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        try:
            if self._client is not None:
                await self._client.close()
        finally:
            self._client = None
            self._client_active = False
            try:
                if self._server is not None:
                    self._server.close()
                    await self._server.wait_closed()
            finally:
                self._server = None
                try:
                    await self.runtime.close()
                finally:
                    if self.port is None and self._socket_created:
                        with contextlib.suppress(FileNotFoundError):
                            if stat.S_ISSOCK(self.socket_path.stat().st_mode):
                                self.socket_path.unlink()
                        self._socket_created = False

    def _prepare_socket_path(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            parent_stat = self.socket_path.parent.stat()
            if parent_stat.st_uid == os.getuid():
                os.chmod(self.socket_path.parent, 0o700)
        except OSError:
            pass
        try:
            path_mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_mode):
            raise RuntimeError(
                f"refusing to replace non-socket path: {self.socket_path}"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise RuntimeError(
                    f"could not inspect socket: {self.socket_path}"
                ) from exc
        else:
            raise RuntimeError(f"a server is already listening on {self.socket_path}")
        finally:
            probe.close()
        self.socket_path.unlink()

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._client_active:
            codec = FrameCodec()
            await codec.write(
                writer,
                codec.error_response(
                    None, -32001, "another client is already connected"
                ),
            )
            writer.close()
            await writer.wait_closed()
            return
        self._client_active = True
        self._client = _Client(self, reader, writer)
        self.runtime.set_background_event_sink(self._client._publish_background_event)
        self.runtime.set_memory_notice_sink(self._client._publish_memory_notice)
        self.runtime.set_background_wake_sink(self._client._schedule_background_wake)
        try:
            await self._client.run()
        finally:
            self.runtime.set_background_event_sink(None)
            self.runtime.set_memory_notice_sink(None)
            self.runtime.set_background_wake_sink(None)
            self._client = None
            self._client_active = False


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
        self.logins = login.Logins(server.home)
        self.handshaken = False
        self.protocol_version = "1.0"
        self.features: frozenset[str] = frozenset()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_cancel_outcome: TurnOutcome | None = None
        self._read_buffer = bytearray()
        self._model_inputs = PendingModelInputs()
        self.codec = FrameCodec()
        self.projects = ProjectRequests(home=server.home, runtime=server.runtime, codec=self.codec)
        self.deliveries = DeliveryRequests(self)
        self._approvals = ApprovalLifecycle()

    async def run(self) -> None:
        try:
            while True:
                line = await self._read_frame()
                if not line:
                    break
                try:
                    request = self.codec.parse_request(line)
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id, exc.code, exc.message, exc.data
                        )
                    )
                    if not self.handshaken:
                        break
                    continue
                request_id = request["id"]
                try:
                    result = await self._dispatch(request_id, request["method"], request["params"])
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id
                            if exc.request_id is not None
                            else request_id,
                            exc.code,
                            exc.message,
                            exc.data,
                        )
                    )
                except PROJECT_REQUEST_EXCEPTIONS as exc:
                    code, message, data = project_request_error(exc)
                    await self._write(self.codec.error_response(request_id, code, message, data))
                except (SessionError, ValueError) as exc:
                    await self._write(
                        self.codec.error_response(request_id, -32602, str(exc))
                    )
                except Exception as exc:  # noqa: BLE001 - keep the socket alive
                    await self._write(
                        self.codec.error_response(request_id, -32000, str(exc))
                    )
                else:
                    await self._write(self.codec.response(request_id, result))
                if not self.handshaken:
                    break
        finally:
            await self.close()

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
            chunk = await self.reader.read(65_536)
            if not chunk:
                if not self._read_buffer:
                    return None
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            self._read_buffer.extend(chunk)

    async def _discard_oversized_frame(self, prefix: bytes) -> bytes:
        """Discard an oversized line and preserve any complete later frames."""

        while True:
            chunk = await self.reader.read(65_536)
            if not chunk:
                return prefix + b"\n"
            newline = chunk.find(b"\n")
            if newline < 0:
                continue
            self._read_buffer.extend(chunk[newline + 1 :])
            return prefix + b"\n"

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._terminate_pending_approvals(suppress_write_errors=True)
            if self._turn_task is not None and not self._turn_task.done():
                loop = self.server.runtime.loop
                if loop is not None:
                    loop.abort(steering_drop_reason=None)
                self._turn_cancel_outcome = TurnOutcome.DISCONNECTED
                self._turn_task.cancel()
                await asyncio.gather(self._turn_task, return_exceptions=True)
            self._turn_task = None
            self._approvals.clear()
            await self.logins.close()
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self._closed = True

    async def _dispatch(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> object:
        if method == "hello":
            result = self._hello(params)
            await self._attach_pending_notifications()
            return result
        if not self.handshaken:
            raise ProtocolError(-32002, "hello must be the first request")
        if method in login.REQUESTS:
            if self.protocol_version != "1.1" or self.server.runtime.fake_catalog:
                raise ProtocolError(-32601, "login is unavailable on this server")
            return await self.logins.request(method, params)
        if method in ergonomics.EXTENSION_REQUESTS:
            if self.protocol_version != "1.1":
                raise ProtocolError(-32601, "request requires protocol 1.1")
            return await self._ergonomics(method, params)
        if method == "ping":
            if "ping" not in self.features:
                raise ProtocolError(-32601, "ping requires the negotiated ping feature")
            return {"pong": True}
        if method == "set_turn_context":
            contexts = self.server.turn_contexts
            return contexts.set_request(self.features, self.server.runtime, params)
        if method in PROJECT_REQUESTS:
            if "projects" not in self.features:
                raise ProtocolError(
                    -32601, f"{method} requires the negotiated projects feature"
                )
            return self.projects.dispatch(request_id, method, params)
        if method == "list_sessions":
            return self._list_sessions(request_id, params)
        if method == "new_session":
            await self._require_idle()
            cwd = None
            if "cwd" in params:
                self._require_feature("session_cwd", "cwd")
                cwd = _directory(params, "cwd")
            await self._terminate_pending_approvals()
            await self.server.runtime.create_session(
                provider=_optional_string(params, "provider"),
                model=_optional_string(params, "model"),
                cwd=cwd,
            )
            self._approvals.clear()
            await self._attach_pending_notifications()
            return {"session": self._session_snapshot()}
        if method == "resume":
            await self._require_idle()
            session_id = _required_string(params, "session_id")
            await self._terminate_pending_approvals()
            try:
                await self.server.runtime.resume_session(session_id)
            except SessionNotFoundError as exc:
                raise ProtocolError(
                    -32602,
                    str(exc),
                    {"code": "session_not_found", "session_id": session_id},
                ) from exc
            self._approvals.clear()
            await self._attach_pending_notifications()
            return {"session": self._session_snapshot()}
        if method == "send":
            return await self.deliveries.send(params)
        if method == "steer":
            return await self.deliveries.steer(params)
        if method == "delivery_status":
            return self.deliveries.status(params)
        if method == "clear_steering":
            return abort_scope.clear_pending_steering(self.features, self.server.runtime.loop)
        if method in {"approve", "deny"}:
            scope = params.get("scope", "once")
            if not isinstance(scope, str) or scope not in {"once", "always_tool"}:
                raise ProtocolError(-32602, "scope must be 'once' or 'always_tool'")
            if scope == "always_tool" and method != "approve":
                raise ProtocolError(-32602, "scope 'always_tool' requires approve")
            return await self._approval(method, _required_string(params, "request_id"), scope)
        if method == "abort":
            return await self._abort(abort_scope.parse_abort_scope(params, self.features))
        if method == "status":
            return self._status()
        raise ProtocolError(-32601, f"method not found: {method}")

    def _hello(self, params: dict[str, Any]) -> dict[str, object]:
        if self.handshaken:
            raise ProtocolError(-32600, "hello may only be sent once")
        version = params.get("protocol_version")
        if version not in ("1.0", PROTOCOL_VERSION):
            raise ProtocolError(
                -32002,
                "unsupported protocol version",
                {"requested": version, "supported": [PROTOCOL_VERSION]},
            )
        requested = params.get("features")
        if requested is not None and (
            not isinstance(requested, list)
            or not all(isinstance(item, str) for item in requested)
        ):
            raise ProtocolError(-32602, "features must be an array of strings")
        self.protocol_version = (
            PROTOCOL_VERSION if version == PROTOCOL_VERSION or params.get("client_version") == PROTOCOL_VERSION else "1.0"
        )
        extended = self.protocol_version == PROTOCOL_VERSION
        accepted = [item for item in FEATURES if item in (requested or ())] if extended else []
        self.features = frozenset(accepted)
        self.server.runtime.set_post_stream_provider_retry(
            "assistant_reset" in self.features
        )
        self.handshaken = True
        requests = [
            "list_sessions",
            "new_session",
            "resume",
            "send",
            "steer",
            "approve",
            "deny",
            "abort",
            "status",
        ]
        if extended:
            requests += ergonomics.EXTENSION_REQUESTS
            if not self.server.runtime.fake_catalog:
                requests += login.REQUESTS
            if "ping" in self.features:
                requests.append("ping")
            self.server.turn_contexts.add_request(requests, self.features)
            if "projects" in self.features:
                requests += PROJECT_REQUESTS
            if "abort_scope" in self.features:
                requests.append("clear_steering")
            if "delivery_id" in self.features:
                requests.append("delivery_status")
        capabilities: dict[str, object] = {
            "requests": requests,
            "notifications": ["event"],
        }
        if extended and requested is not None:
            capabilities["features"] = accepted
        return {
            "protocol_version": self.protocol_version,
            "server": "zeta",
            "capabilities": capabilities,
        }

    def _require_feature(self, feature: str, param: str) -> None:
        if feature not in self.features:
            raise ProtocolError(
                -32602, f"{param} requires the negotiated {feature} feature"
            )

    async def _user_message(
        self, text: str, mode: str, attachments: list[dict[str, object]] | None = None
    ) -> None:
        """Echo an accepted user message so observers can rebuild the transcript."""

        if "user_message_event" in self.features:
            await self._notify(
                "user_message",
                self.server.runtime.session_id,
                text=bounded(text, USER_MESSAGE_MAX_BYTES),
                mode=mode,
                attachments=attachments or [],
            )

    async def _ergonomics(self, method: str, params: dict[str, Any]) -> object:
        runtime = self.server.runtime
        if method in {"rename_session", "delete_session"}:
            session_id = runtime.manager.resolve_id(_required_string(params, "session_id"))
            if method == "delete_session":
                await self._require_idle()
                if runtime.opened is not None and session_id == runtime.session_id:
                    raise ProtocolError(-32005, "select another session before deleting this session", {"code": "active_session"})
                runtime.manager.delete(session_id)
                return {"session_id": session_id}
            name = params.get("name")
            if not isinstance(name, str):
                raise ProtocolError(-32602, "name must be a string")
            metadata = runtime.manager.rename(session_id, name)
            if runtime.opened is not None and session_id == runtime.session_id:
                runtime.manager._copy_metadata(runtime.metadata, metadata)
            return {"session": metadata.to_dict()}
        store = ergonomics.active(runtime, params)
        if method == "session_tree":
            return ergonomics.tree(runtime)
        if method == "session_history":
            return ergonomics.history(runtime, params)
        if method == "model_catalog":
            return ergonomics.catalog(runtime)
        if method == "session_settings":
            return ergonomics.settings(runtime)
        if method == "slash_list":
            return slash_commands.list_commands(runtime)
        await self._require_idle()
        if method == "slash_run":
            register = self._model_inputs.registrar(runtime.session_id, enabled="model_input_ids" in self.features)
            return await slash_commands.run_command(
                runtime, _required_string(params, "text"), register_model_input=register
            )
        ergonomics.require_mutable(runtime)
        if method == "send_images":
            message = ergonomics.image_message(runtime, params)
            await self._user_message(
                params.get("text", ""), "send", ergonomics.attachment_summary(message)
            )
            self._turn_task = asyncio.create_task(self._run_turn(params.get("text", ""), message))
            return {"accepted": True, "session_id": runtime.session_id}
        if method == "set_settings":
            model = _required_string(params, "model")
            mode = _required_string(params, "approval_mode")
            if model not in ergonomics.catalog(runtime)["models"] or mode not in {"ask", "allow", "deny"}:
                raise ProtocolError(-32602, "invalid model or approval mode")
            model_selection.apply(runtime, model, mode)
            return ergonomics.settings(runtime)
        if method == "set_compaction":
            return switch_compaction(
                runtime.loop, _required_string(params, "mode")
            ).to_dict()
        if method == "fork_message":
            store.append_message_fork(_required_string(params, "message_id"))
        elif method == "switch_branch":
            head = _required_string(params, "head_id")
            current = store.replay()
            if not current or current[-1].id != head:
                store.switch_to_branch(head)
        return ergonomics.tree(runtime)
    async def _approval(
        self, method: str, request_id: str, scope: str = "once"
    ) -> dict[str, object]:
        policy = self.server.runtime.policy
        loop = self.server.runtime.loop
        if policy is None or loop is None:
            raise ProtocolError(-32003, "no active session")
        core_key = self._approvals.core_key(request_id)
        pending = next(
            (item for item in policy.pending_requests() if item.key == core_key),
            None,
        )
        if (
            scope == "always_tool"
            and pending is not None
            and pending.child_instance_id is not None
        ):
            raise ProtocolError(
                -32602, "scope 'always_tool' is unavailable for delegated approvals"
            )
        if scope == "always_tool" and pending is not None:
            try:
                policy.remember_allow(pending)
            except ValueError as exc:
                raise ProtocolError(
                    -32602, f"cannot always allow approval request: {exc}"
                ) from exc
        resolved = policy.resolve(
            core_key,
            ApprovalDecision.ALLOW if method == "approve" else ApprovalDecision.DENY,
        )
        if not resolved:
            raise ProtocolError(
                -32006, f"approval request not found or already resolved: {request_id}"
            )
        if pending is not None:
            self._approvals.observe(pending)
        await self._end_approval(core_key, self.server.runtime.session_id)
        active = self._turn_busy()
        if (
            not active
            and isinstance(core_key, str)
            and loop.prepare_resume_pending_tool(core_key)
        ):
            if self.server.runtime.state is not None:
                self.server.runtime.state.tool_started()
            task = asyncio.create_task(self._resume_tool(core_key))
            self._turn_task = task
        result: dict[str, object] = {
            "accepted": True,
            "request_id": request_id,
            "decision": method,
        }
        if scope != "once":
            result["scope"] = scope
        return result

    async def _abort(self, scope: abort_scope.AbortScope = "session") -> dict[str, object]:
        session_id = self.server.runtime.session_id
        outcome = (
            TurnOutcome.FOREGROUND_CANCELED
            if scope == "foreground"
            else TurnOutcome.SESSION_CANCELED
        )
        turn_task = self._turn_task

        def mark_canceled() -> None:
            if turn_task is not None and not turn_task.done():
                self._turn_cancel_outcome = outcome

        aborted = await abort_scope.abort_active_turn(
            turn_task,
            scope=scope,
            loop=self.server.runtime.loop,
            terminate_approvals=self._terminate_pending_approvals,
            before_cancel=mark_canceled,
        )
        if aborted:
            await self._notify("turn_aborted", session_id)
        return {"aborted": aborted}

    def _session_snapshot(self) -> dict[str, object] | None:
        runtime = self.server.runtime
        if runtime.opened is None:
            return None
        snapshot = runtime.metadata.to_dict()
        if runtime.policy is not None:
            snapshot["approval_mode"] = runtime.policy.default.value
        return snapshot

    def _status(self) -> dict[str, object]:
        runtime = self.server.runtime
        session = self._session_snapshot()
        pending = []
        if runtime.policy is not None:
            pending = [
                {
                    **self._approvals.observe(item),
                    **_approval_display_fields(item),
                }
                for item in runtime.policy.pending_requests()
            ]
        return {
            "session": session,
            "state": runtime.state.status if runtime.state is not None else "idle",
            "pending_approvals": pending,
            "usage": dict(runtime.usage),
            "compaction_markers": (
                runtime.opened.store.compaction_marker_count()
                if runtime.opened is not None
                else 0
            ),
        }

    async def _run_turn(
        self,
        text: str,
        user_message: Message | None = None,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        persist_user_message: bool = True,
    ) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        state.turn_started()
        outcome = TurnOutcome.COMPLETED
        agent_end: StreamEvent | None = None
        try:
            async for event in loop.run_turn(
                text,
                origin=origin,
                user_message=user_message,
                persist_user_message=persist_user_message,
            ):
                if event.type is StreamEventType.ERROR:
                    outcome = TurnOutcome.FAILED
                if event.type is StreamEventType.AGENT_END:
                    agent_end = event
                else:
                    await self._event(event, session_id=session_id)
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            await self._notify(
                "error",
                session_id,
                error={"code": "server_error", "message": str(exc)},
                data={},
            )
        finally:
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    def _schedule_background_wake(self, session_id: str) -> None:
        if self._closed:
            return
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        if not loop.schedule_notification_turn():
            return
        self._turn_task = asyncio.create_task(
            self._run_notification_turn(session_id)
        )

    async def _run_notification_turn(self, session_id: str) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        outcome = TurnOutcome.FAILED
        started = False
        agent_end: StreamEvent | None = None
        try:
            await asyncio.sleep(0.01)
            state.turn_started()
            started = True
            success, agent_end = await self.server.turn_contexts.run_notification_turn(
                session_id, loop, self._event
            )
            outcome = TurnOutcome.COMPLETED if success else TurnOutcome.FAILED
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            await self._notify(
                "error",
                session_id,
                error={"code": "server_error", "message": str(exc)},
                data={},
            )
        finally:
            if not started and loop.notification_turn_state == "scheduled":
                await loop.finish_notification_turn(success=False)
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    async def _attach_pending_notifications(self) -> None:
        runtime = self.server.runtime
        loop = runtime.loop
        if loop is None or runtime.state is None:
            return
        if loop.notification_system_message() is not None:
            self._schedule_background_wake(runtime.session_id)

    async def _resume_tool(self, request_id: str) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        event_tasks: list[asyncio.Task[None]] = []
        agent_end: StreamEvent | None = None

        def emit(event: StreamEvent) -> None:
            nonlocal agent_end
            if event.type is StreamEventType.AGENT_END:
                agent_end = event
            else:
                event_tasks.append(
                    asyncio.create_task(self._event(event, session_id=session_id))
                )

        outcome = TurnOutcome.COMPLETED
        try:
            await loop.resume_pending_tool(
                request_id,
                prepared=True,
                event_sink=emit,
            )
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except BaseException:
            outcome = TurnOutcome.FAILED
            raise
        finally:
            if event_tasks:
                await asyncio.gather(*event_tasks, return_exceptions=True)
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            self._schedule_background_wake(session_id)

    async def _event(
        self,
        event: StreamEvent,
        *,
        session_id: str | None = None,
        background: bool = False,
    ) -> None:
        state = self.server.runtime.state
        if session_id is None and not background and state is not None:
            session_id = state.session_id
        foreground = (
            not background
            and state is not None
            and session_id is not None
            and state.session_id == session_id
        )
        kind = event.type
        if kind not in {
            StreamEventType.TOOL_APPROVAL_START,
            StreamEventType.TOOL_APPROVAL_END,
        }:
            await self._end_stale_approvals(session_id)
        if kind is StreamEventType.MESSAGE_UPDATE:
            delta = event.delta
            stream_kind = "assistant"
            if isinstance(event.content, TextContent):
                delta = event.content.text
            elif event.content is not None:
                delta = getattr(event.content, "text", None)
                stream_kind = str(getattr(event.content, "type", "content"))
            if delta:
                await self._notify(
                    "assistant_delta",
                    session_id,
                    delta=bounded(delta),
                    kind=stream_kind,
                )
            return
        if kind is StreamEventType.MESSAGE_END:
            if foreground and not event.data.get("truncated"):
                model_selection.confirm(self.server.runtime)
            usage = event.data.get("usage")
            if foreground and isinstance(usage, Mapping) and usage:
                state.usage.update(usage)
                await self._notify("usage", session_id, usage=dict(usage))
            if event.message is not None:
                wire_message = event.message.to_dict()
                # Codex keeps encrypted replay items in assistant metadata.
                # They are server-only and may contain raw reasoning text.
                wire_message.pop("metadata", None)
                await self._notify(
                    "assistant_message", session_id, message=wire_message
                )
            return
        if kind is StreamEventType.TOOL_EXECUTION_UPDATE:
            await self._notify(
                "tool_output",
                session_id,
                tool_call=_tool_call(event),
                output=bounded(event.delta or _data_text(event.data)),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_START:
            if foreground:
                state.tool_started()
            await self._notify(
                "tool_start",
                session_id,
                tool_call=_tool_call(event),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_END:
            if foreground:
                state.tool_finished()
            await self._notify(
                "tool_end",
                session_id,
                tool_call=_tool_call(event),
                tool_result=(
                    event.tool_result.to_dict() if event.tool_result else None
                ),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_END:
            await self._end_approval(
                _approval_key(event), session_id, data=dict(event.data)
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_START:
            if foreground:
                state.tool_started()
            raw_request_id = event.tool_call.id if event.tool_call else ""
            child_id = event.data.get("agent_instance_id")
            core_key: str | tuple[str, str] = (
                (child_id, raw_request_id)
                if isinstance(child_id, str) and raw_request_id
                else raw_request_id
            )
            policy = self.server.runtime.policy
            request = (
                next(
                    (
                        item
                        for item in policy.pending_requests()
                        if item.key == core_key
                    ),
                    None,
                )
                if policy is not None
                else None
            )
            if request is None:
                if policy is not None:
                    policy.deny(core_key)
                # A live approval without its exact pending request has lost the
                # harness-owned execution facts.  Never offer an approval based
                # only on the provider-controlled ToolCall.
                await self._notify(
                    "error",
                    session_id,
                    error={
                        "code": "approval_context_missing",
                        "message": "approval request is unavailable; denying live approval",
                    },
                    data={"request_id": raw_request_id},
                )
                return
            await self._notify(
                "approval_request",
                session_id,
                **self._approvals.observe(request),
                **_approval_display_fields(request),
            )
            return
        if kind is StreamEventType.AGENT_NOTIFICATION:
            notification = dict(event.data)
            event_name = (
                "task_exit_notification"
                if notification.get("kind", "agent_completion") == "task_exited"
                else "sub_agent_receipt"
            )
            await self._notify(event_name, session_id, data=notification)
            return
        if kind is StreamEventType.ERROR:
            error = (
                {"code": event.error.code, "message": event.error.message}
                if event.error else {"code": "unknown", "message": "unknown error"}
            )
            login_provider = None
            if (foreground and event.error is not None and event.error.provider_error
                    and (event.error.code == "auth_error" or event.error.status_code == 401)
                    and self.protocol_version == "1.1" and not self.server.runtime.fake_catalog):
                login_provider = self.server.runtime.metadata.provider
            if foreground and event.error is not None:
                recovered = model_selection.recover(self.server.runtime, event.error)
                if self.protocol_version == "1.1":
                    error = recovered
            await self._notify(
                "error", session_id, error=error, data={**event.data, **({"login_provider": login_provider} if login_provider else {})},
            )
            return
        if kind is StreamEventType.COMPACTION_START:
            await self._notify("compaction_start", session_id, data=dict(event.data))
            return
        if kind is StreamEventType.COMPACTION_END:
            await self._notify("compaction_end", session_id, data=dict(event.data))
            return
        if foreground and kind is StreamEventType.AGENT_END:
            state.turn_finished()
        await self._notify(kind.value, session_id, data=dict(event.data))

    def _publish_background_event(self, session_id: str, event: StreamEvent) -> None:
        """Schedule child events from the loop's synchronous sink."""

        if not self._closed:
            asyncio.create_task(
                self._event(event, session_id=session_id, background=True)
            )

    def _publish_memory_notice(self, session_id: str, message: str) -> None:
        """Publish updates only when the client negotiated the optional feature."""
        if not self._closed and "memory_updated" in self.features:
            asyncio.create_task(
                self._notify("memory_updated", session_id, message=message)
            )

    async def _notify(
        self, event: str, session_id: str | None = None, **fields: object
    ) -> None:
        if session_id is None and self.server.runtime.opened is not None:
            session_id = self.server.runtime.session_id
        await self._write(self.codec.notification(event, session_id, **fields))

    def _list_sessions(
        self, request_id: str | int, params: dict[str, Any]
    ) -> dict[str, object]:
        paging = "list_sessions_paging" in self.features
        for name in ("offset", "limit"):
            if name in params:
                self._require_feature("list_sessions_paging", name)
        offset = _integer(params, "offset", 0, minimum=0)
        limit = _integer(params, "limit", None, minimum=1)
        if "project_id" in params:
            self._require_feature("projects", "project_id")
            available = self.projects.project_sessions(params["project_id"])
        else:
            available = self.server.runtime.list_sessions()
        metadata = available[offset : None if limit is None else offset + limit]
        previews = {
            item.session_id: item.preview
            for item in self.server.runtime.manager.list_session_previews(
                limit=len(metadata), sessions=metadata
            )
        }
        sessions = [
            {**item.to_dict(), "first_message_preview": previews.get(item.session_id, "")}
            for item in metadata
        ]
        if self.protocol_version != "1.1":
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
                }
            page = candidate
        if not paging:
            return {"sessions": page}
        end = offset + len(page)
        return {"sessions": page, "next_offset": end if end < len(available) else None}

    async def _end_approval(
        self,
        key: ApprovalKey,
        session_id: str | None,
        *,
        data: dict[str, object] | None = None,
        suppress_write_errors: bool = False,
    ) -> None:
        payload = self._approvals.end(key, data=data)
        if payload is None:
            return
        try:
            await self._notify("approval_end", session_id, **payload)
        except (ConnectionError, OSError):
            if not suppress_write_errors:
                raise

    async def _end_stale_approvals(self, session_id: str | None) -> None:
        policy = self.server.runtime.policy
        pending_keys = (
            {request.key for request in policy.pending_requests()}
            if policy is not None
            else set()
        )
        for key in self._approvals.active_keys():
            if key not in pending_keys:
                await self._end_approval(key, session_id)

    async def _terminate_pending_approvals(
        self,
        *,
        foreground_only: bool = False,
        suppress_write_errors: bool = False,
    ) -> None:
        policy = self.server.runtime.policy
        if policy is not None:
            for request in policy.pending_requests():
                if foreground_only and request.child_instance_id is not None:
                    continue
                with contextlib.suppress(ValueError, RuntimeError):
                    policy.abort(request.key)
        for key in self._approvals.active_keys():
            if foreground_only and not isinstance(key, str):
                continue
            await self._end_approval(
                key,
                self.server.runtime.session_id if self.server.runtime.opened else None,
                suppress_write_errors=suppress_write_errors,
            )

    async def _write(self, payload: bytes) -> None:
        async with self._write_lock:
            await self.codec.write(self.writer, payload)

    def _turn_busy(self) -> bool:
        loop = self.server.runtime.loop
        return (
            self._turn_task is not None and not self._turn_task.done()
        ) or (loop is not None and loop.notification_turn_state != "idle")

    async def _require_idle(self) -> None:
        if self._turn_busy():
            raise ProtocolError(-32004, "a turn is already running")


def _home() -> Path:
    from ..core.session import env_home

    return env_home().expanduser().resolve()


def _required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _integer(
    params: dict[str, Any], name: str, default: int | None, *, minimum: int
) -> int | None:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProtocolError(-32602, f"{name} must be an integer of at least {minimum}")
    return value


def _directory(params: dict[str, Any], name: str) -> Path:
    value = params.get(name)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ProtocolError(-32602, f"{name} must be an absolute directory path")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ProtocolError(-32602, f"{name} is not an existing directory: {value}")
    return path


def _optional_string(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is not None and (not isinstance(value, str) or not value):
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _tool_call(event: StreamEvent) -> dict[str, object] | None:
    return event.tool_call.to_dict() if event.tool_call is not None else None


def _approval_key(event: StreamEvent) -> ApprovalKey:
    raw_request_id = event.tool_call.id if event.tool_call is not None else ""
    child_id = event.data.get("agent_instance_id")
    if isinstance(child_id, str) and raw_request_id:
        return child_id, raw_request_id
    return raw_request_id


def _approval_display_fields(request: Any) -> dict[str, object]:
    """The one immutable approval-display object shared with the frontend client.

    Returns an ``approval_display`` wire field only when the harness resolved
    trusted project or execution facts; otherwise nothing is added and the client
    keeps its backward-compatible behavior.
    """
    fields: dict[str, object] = {}
    if request.action is not None:
        fields["approval_action"] = request.action
    display = request.audit_display()
    if display:
        fields["approval_display"] = display
    return fields


def _data_text(data: Mapping[str, object]) -> str:
    value = data.get("output", data.get("text", ""))
    return value if isinstance(value, str) else ""


async def run_server(server: ZetaServer) -> None:
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    await server.start()
    serving = asyncio.create_task(server.serve_forever())
    try:
        await stopped.wait()
    finally:
        await server.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        for signum in installed:
            loop.remove_signal_handler(signum)


__all__ = ["ZetaServer", "run_server"]
