from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest

from tests.support.server_client import event, request


def _entrypoint() -> Path:
    path = Path(sys.executable).with_name("zeta")
    assert path.is_file(), (
        f"installed zeta entrypoint not found beside {sys.executable}"
    )
    return path


async def _start_server(home: Path, socket_path: Path, cwd: Path):
    environment = os.environ.copy()
    environment["ZETA_HOME"] = str(home)
    environment["ZETA_TEST_SCRIPTED_PROVIDER"] = "1"
    return await asyncio.create_subprocess_exec(
        _entrypoint(),
        "serve",
        "--socket",
        str(socket_path),
        "--provider",
        "codex",
        "--cwd",
        str(cwd),
        "--no-auto-memory",
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _connect_ready(
    process: asyncio.subprocess.Process, socket_path: Path, default_socket: Path
):
    while True:
        assert not default_socket.exists(), "zeta serve ignored the --socket path"
        assert process.returncode is None, await _process_output(process)
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
        except (FileNotFoundError, ConnectionRefusedError):
            await asyncio.sleep(0)
            continue
        break
    hello = (
        await request(reader, writer, "hello", "hello", {"protocol_version": "1.2"})
    )[-1]
    assert hello["result"]["server"] == "zeta"
    return reader, writer


async def _process_output(process: asyncio.subprocess.Process) -> str:
    stdout, stderr = await process.communicate()
    return (stdout + stderr).decode(errors="replace")


async def _send(reader, writer, request_id: str, text: str) -> None:
    sent = (await request(reader, writer, request_id, "send", {"text": text}))[-1]
    assert sent["result"]["accepted"] is True
    reply = await event(reader, "assistant_message")
    assert reply["message"]["content"] == [
        {"type": "text", "text": f"you said: {text}"}
    ]
    await event(reader, "agent_end")


async def _stop(process: asyncio.subprocess.Process, socket_path: Path) -> None:
    process.send_signal(signal.SIGTERM)
    output = await _process_output(process)
    assert process.returncode == 0, output
    assert not socket_path.exists(), "SIGTERM left the serve socket behind"


def _assert_runtime_lease_released(home: Path, session_id: str) -> None:
    lease_path = home / "sessions" / session_id / "runtime.lease"
    with lease_path.open("r+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.fail("SIGTERM left the session runtime lease held")
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def _message_texts(messages: list[dict[str, object]], role: str) -> list[str]:
    return [
        block["text"]
        for message in messages
        if message["role"] == role
        for block in message["content"]
        if block["type"] == "text"
    ]


@pytest.mark.asyncio
async def test_installed_serve_process_survives_reconnect_and_restart(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    home = Path("/tmp") / f"zeta-process-{token}-home"
    socket_path = Path("/tmp") / f"zeta-process-{token}.sock"
    default_socket = home / "run" / "serve.sock"
    processes: list[asyncio.subprocess.Process] = []
    writers: list[asyncio.StreamWriter] = []

    try:
        async with asyncio.timeout(90):
            first = await _start_server(home, socket_path, tmp_path)
            processes.append(first)
            reader, writer = await _connect_ready(first, socket_path, default_socket)
            writers.append(writer)

            created = (await request(reader, writer, "new", "new_session", {}))[-1]
            session_id = created["result"]["session"]["session_id"]
            await _send(reader, writer, "first", "first process turn")

            writer.close()
            await writer.wait_closed()
            writers.remove(writer)

            reader, writer = await _connect_ready(first, socket_path, default_socket)
            writers.append(writer)
            resumed = (
                await request(
                    reader, writer, "resume", "resume", {"session_id": session_id}
                )
            )[-1]
            assert resumed["result"]["session"]["session_id"] == session_id
            await _send(reader, writer, "second", "reconnected turn")

            await _stop(first, socket_path)
            _assert_runtime_lease_released(home, session_id)
            writer.close()
            await writer.wait_closed()
            writers.remove(writer)

            second = await _start_server(home, socket_path, tmp_path)
            processes.append(second)
            reader, writer = await _connect_ready(second, socket_path, default_socket)
            writers.append(writer)
            resumed = (
                await request(
                    reader, writer, "resume-again", "resume", {"session_id": session_id}
                )
            )[-1]
            assert resumed["result"]["session"]["session_id"] == session_id

            history = (
                await request(
                    reader,
                    writer,
                    "history",
                    "session_history",
                    {"session_id": session_id},
                )
            )[-1]["result"]["messages"]
            assert _message_texts(history, "user") == [
                "first process turn",
                "reconnected turn",
            ]
            assert _message_texts(history, "assistant") == [
                "you said: first process turn",
                "you said: reconnected turn",
            ]

            await _stop(second, socket_path)
            _assert_runtime_lease_released(home, session_id)
    finally:
        for writer in writers:
            writer.close()
            await writer.wait_closed()
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        socket_path.unlink(missing_ok=True)
        Path(f"{socket_path}.lock").unlink(missing_ok=True)
        shutil.rmtree(home, ignore_errors=True)
