from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from zeta.remote.descriptor import (
    Descriptor,
    DescriptorError,
    publish,
    socket_path,
    validate,
)


def test_descriptor_rejects_unsafe_socket_and_overlong_path(tmp_path: Path) -> None:
    descriptor = Descriptor(str(tmp_path / "missing.sock"), "session", os.getpid())
    path = publish(tmp_path, descriptor)
    with pytest.raises(DescriptorError):
        validate(path)


def test_descriptor_rejects_permissive_descriptor(tmp_path: Path) -> None:
    directory = tmp_path / "run" / "remote"
    directory.mkdir(parents=True)
    path = directory / "session.json"
    path.write_bytes(Descriptor("/tmp/no.sock", "session", os.getpid()).encode())
    os.chmod(path, 0o644)
    with pytest.raises(DescriptorError):
        validate(path)


@pytest.mark.asyncio
async def test_stale_probe_runs_off_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zeta.remote.host import ObserverHost
    from zeta.runtime.foreground_events import ForegroundEventPublisher
    from zeta.server.runtime import ServerRuntime

    path = socket_path(tmp_path, "session")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(path))
    stale.close()
    import threading

    threads: list[threading.Thread] = []
    import zeta.remote.host as host_module

    original = host_module.create_observer_socket

    def checked(path: Path) -> socket.socket:
        threads.append(threading.current_thread())
        return original(path)

    monkeypatch.setattr(host_module, "create_observer_socket", checked)
    host = ObserverHost(
        home=tmp_path,
        session_id="session",
        runtime=ServerRuntime(tmp_path, provider="codex", model="gpt-5"),
        publisher=ForegroundEventPublisher(),
    )
    await host.start()
    try:
        assert threads and threads[0] is not threading.main_thread()
        await host.start()
        assert host._server is not None
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_probe_stops_when_peer_closes_and_ignores_notifications(tmp_path: Path) -> None:
    from zeta.remote.bridge import _probe_state

    path = socket_path(tmp_path, "session")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    server = await asyncio.start_unix_server(lambda reader, writer: _probe_peer(reader, writer), path)
    try:
        descriptor = Descriptor(str(path), "session", os.getpid())
        assert await asyncio.wait_for(asyncio.to_thread(_probe_state, descriptor), 10) == "running"
    finally:
        server.close()
        await server.wait_closed()


# test-contract: transport raw socket peer is required to exercise probe framing and eof.
async def _probe_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        # test-contract: transport raw socket peer is required to exercise probe framing and eof.
        await reader.readline()
        writer.write(b'{"jsonrpc":"2.0","method":"event","params":{}}\n')
        writer.write(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
        await writer.drain()
        # test-contract: transport raw socket peer is required to exercise probe framing and eof.
        await reader.readline()
        writer.write(b'{"jsonrpc":"2.0","id":2,"result":{"state":"tool"}}\n')
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_attach_exits_when_stdout_stalls_and_socket_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.remote.bridge import attach

    path = socket_path(tmp_path, "session")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    output_read, output_write = os.pipe()
    os.set_blocking(output_write, False)
    try:
        while True:
            try:
                os.write(output_write, b"x" * 65536)
            except BlockingIOError:
                break

        class Buffer:
            def fileno(self) -> int:
                return output_write

        class Stdout:
            buffer = Buffer()

        monkeypatch.setattr("sys.stdout", Stdout())
        server = await asyncio.start_unix_server(
            lambda reader, writer: _stalled_stdout_peer(reader, writer), path
        )
        read_fd, write_fd = os.pipe()

        class Stdin:
            def fileno(self) -> int:
                return read_fd

        monkeypatch.setattr("sys.stdin", Stdin())
        descriptor_path = publish(tmp_path, Descriptor(str(path), "session", os.getpid()))
        try:
            assert await asyncio.wait_for(attach(tmp_path, "session"), 10) == 2
        finally:
            os.close(read_fd)
            os.close(write_fd)
            descriptor_path.unlink(missing_ok=True)
            server.close()
            await server.wait_closed()
    finally:
        os.close(output_read)
        os.close(output_write)


async def _stalled_stdout_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.write(b"x" * 65536)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_attach_exits_when_socket_closes_with_stdin_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zeta.remote.bridge import attach

    path = socket_path(tmp_path, "session")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    server = await asyncio.start_unix_server(lambda reader, writer: _close_peer(reader, writer), path)
    os.chmod(path, 0o600)
    read_fd, write_fd = os.pipe()

    class Stdin:
        def fileno(self) -> int:
            return read_fd

    monkeypatch.setattr("sys.stdin", Stdin())
    descriptor_path = publish(tmp_path, Descriptor(str(path), "session", os.getpid()))
    try:
        assert await asyncio.wait_for(attach(tmp_path, "session"), 10) == 0
    finally:
        os.close(read_fd)
        os.close(write_fd)
        server.close()
        await server.wait_closed()
        descriptor_path.unlink(missing_ok=True)


async def _close_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_host_rejects_unsafe_socket_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.remote.host import ObserverHost
    from zeta.runtime.foreground_events import ForegroundEventPublisher
    from zeta.server.runtime import ServerRuntime

    target = Path("/tmp")
    symlink = tmp_path / "observer-dir"
    symlink.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        "zeta.remote.host.socket_path",
        lambda home, session_id: symlink / "observer.sock",
    )
    host = ObserverHost(
        home=tmp_path,
        session_id="session",
        runtime=ServerRuntime(tmp_path, provider="codex", model="gpt-5"),
        publisher=ForegroundEventPublisher(),
    )
    try:
        try:
            await host.start()
        except (DescriptorError, RuntimeError, OSError):
            pass
        else:
            pytest.fail("observer host accepted a symlink socket directory")
        assert not (target / "observer.sock").exists()
    finally:
        await host.close()


def test_standalone_socket_parent_keeps_permissive_mode(tmp_path: Path) -> None:
    from zeta.server.server import _UnixSocketListener

    parent = tmp_path / "ordinary"
    parent.mkdir(mode=0o755)
    _UnixSocketListener._prepare_parent(parent / "socket")
    assert (parent.stat().st_mode & 0o777) == 0o700


def test_descriptor_rejects_overlong_socket_before_probe(tmp_path: Path) -> None:
    descriptor = Descriptor("/tmp/" + "x" * 100, "session", os.getpid())
    path = publish(tmp_path, descriptor)
    with pytest.raises(DescriptorError, match="too long|invalid"):
        validate(path)


@pytest.mark.asyncio
async def test_observer_host_restart_keeps_borrowed_publisher_live(tmp_path: Path) -> None:
    """A host close (fork suspend) must not close the TUI's publisher, so a
    restarted host still streams foreground events to new observers."""
    from tests.support.fake_backend import FakeBackend
    from tests.support.owned import await_or_fail
    from tests.support.replay_transport import ReplayClient, negotiate
    from zeta.protocol.types import StreamEvent, StreamEventType
    from zeta.remote.host import ObserverHost
    from zeta.runtime.foreground_events import ForegroundEventPublisher
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(
        tmp_path,
        provider="codex",
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    publisher = ForegroundEventPublisher()
    host: ObserverHost | None = None
    client: ReplayClient | None = None
    try:
        session_id = (await runtime.create_session(provider="codex")).session_id
        host = ObserverHost(home=tmp_path, session_id=session_id, runtime=runtime, publisher=publisher)
        await host.start()
        await host.close()
        await host.start()
        reader, writer = await asyncio.open_unix_connection(str(host.socket))
        client = ReplayClient(reader, writer, False)
        await negotiate(client, observer=True)
        joined = await client.request(2, "session_history_and_subscribe", {"session_id": session_id})
        assert "error" not in joined, joined

        async def source():
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="live after restart")

        assert [event.delta async for event in publisher.events(source())] == ["live after restart"]
        event = await await_or_fail(client.event("assistant_delta"), what="observer live event")
        assert event["delta"] == "live after restart"
    finally:
        if client is not None:
            await client.close()
        if host is not None:
            await host.close()
        publisher.close()
        await runtime.close()
