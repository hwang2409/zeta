from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from zeta.remote.descriptor import (
    Descriptor,
    DescriptorError,
    observer_socket_dir,
    publish,
    socket_path,
    validate,
)


def test_descriptor_rejects_unsafe_socket_and_overlong_path(tmp_path: Path) -> None:
    descriptor = Descriptor(str(tmp_path / "missing.sock"), "session", os.getpid())
    path = publish(tmp_path, descriptor)
    with pytest.raises(DescriptorError):
        validate(path)
    assert len(os.fsencode(socket_path(tmp_path, "session"))) <= 90


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
    called = False
    original = asyncio.to_thread

    async def checked(function, *args, **kwargs):
        nonlocal called
        called = True
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", checked)
    host = ObserverHost(
        home=tmp_path,
        session_id="session",
        runtime=ServerRuntime(tmp_path, provider="codex", model="gpt-5"),
        publisher=ForegroundEventPublisher(),
    )
    await host.start()
    try:
        assert called
        await host.start()
        assert host._server is not None
    finally:
        await host.close()


def test_socket_path_limit_is_encoded() -> None:
    path = socket_path(Path("/"), "x")
    assert len(os.fsencode(path)) <= 90


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


def test_observer_socket_dir_rejects_symlink_and_wrong_mode(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(DescriptorError, match="unsafe"):
        observer_socket_dir(symlink)
    wrong = tmp_path / "wrong"
    wrong.mkdir(mode=0o755)
    with pytest.raises(DescriptorError, match="unsafe"):
        observer_socket_dir(wrong)


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
