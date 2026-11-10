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
    finally:
        await host.close()


def test_socket_path_limit_is_encoded() -> None:
    path = socket_path(Path("/"), "x")
    assert len(os.fsencode(path)) <= 90


def test_descriptor_rejects_overlong_socket_before_probe(tmp_path: Path) -> None:
    descriptor = Descriptor("/tmp/" + "x" * 100, "session", os.getpid())
    path = publish(tmp_path, descriptor)
    with pytest.raises(DescriptorError, match="too long|invalid"):
        validate(path)
