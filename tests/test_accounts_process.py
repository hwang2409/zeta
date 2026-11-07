from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.account_homes import build_account_home
from tests.support.server_client import request
from zeta.accounts import AccountId

_PROCESS_TIMEOUT = 30


def _entrypoint() -> Path:
    path = Path(sys.executable).with_name("zeta")
    assert path.is_file(), (
        f"installed zeta entrypoint not found beside {sys.executable}"
    )
    return path


def _environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["ZETA_HOME"] = str(home)
    environment.pop("ZETA_ACCOUNT_HOME", None)
    for name in tuple(environment):
        if name.startswith("ZETA_TEST_"):
            environment.pop(name)
    return environment


async def _wait_until_ready(
    process: asyncio.subprocess.Process, output: bytearray
) -> None:
    assert process.stderr is not None
    while True:
        line = await process.stderr.readline()
        output.extend(line)
        assert line, output.decode(errors="replace")
        if b"ready" in line:
            return


async def _stop_process(
    process: asyncio.subprocess.Process, output: bytearray, step: str
) -> None:
    if process.returncode is None:
        process.send_signal(signal.SIGTERM)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), _PROCESS_TIMEOUT)
    except TimeoutError as exc:
        process.kill()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), _PROCESS_TIMEOUT
            )
        except TimeoutError as kill_exc:
            raise AssertionError(
                f"timed out waiting for {step} after killing the process"
            ) from kill_exc
        raise AssertionError(
            f"timed out waiting for {step}: "
            f"{(output + stdout + stderr).decode(errors='replace')}"
        ) from exc
    assert process.returncode == 0, (output + stdout + stderr).decode(errors="replace")


@pytest.mark.parametrize(
    "arguments",
    [
        ("--provider", "codex"),
        ("--provider", "codex", "-p", "/status"),
    ],
    ids=("tui", "headless"),
)
def test_console_surfaces_refuse_only_the_active_missing_account(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    account = AccountId("codex", "work")
    build_account_home(tmp_path, (account,), defaults=(account,))

    result = subprocess.run(
        (_entrypoint(), *arguments),
        cwd=Path.cwd(),
        env=_environment(tmp_path),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROCESS_TIMEOUT,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "zeta: account codex:work is unavailable: credentials missing; "
        "run 'zeta accounts login codex work --replace'\n"
    )


def test_console_lists_non_active_missing_accounts(
    tmp_path: Path,
) -> None:
    current = AccountId("anthropic", "current")
    missing = AccountId("codex", "missing")
    build_account_home(
        tmp_path,
        (current, missing),
        available=(current,),
        defaults=(current,),
    )
    environment = _environment(tmp_path)

    listed = subprocess.run(
        (_entrypoint(), "accounts"),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROCESS_TIMEOUT,
    )
    assert listed.returncode == 0, listed.stderr
    assert "anthropic:current" in listed.stdout
    assert "codex:missing  unavailable: credentials missing" in listed.stdout


@pytest.mark.asyncio
async def test_console_serve_refuses_a_session_with_missing_active_account(
    tmp_path: Path,
) -> None:
    account = AccountId("codex", "work")
    await asyncio.to_thread(
        build_account_home, tmp_path, (account,), defaults=(account,)
    )
    socket_path = Path("/tmp") / f"zeta-accounts-missing-{os.getpid()}.sock"
    process = await asyncio.create_subprocess_exec(
        _entrypoint(),
        "serve",
        "--provider",
        "codex",
        "--socket",
        socket_path,
        "--ws",
        "0",
        "--no-auto-memory",
        cwd=Path.cwd(),
        env=_environment(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output = bytearray()
    writer: asyncio.StreamWriter | None = None
    try:
        try:
            await asyncio.wait_for(_wait_until_ready(process, output), _PROCESS_TIMEOUT)
        except TimeoutError as exc:
            raise AssertionError(
                "timed out waiting for serve readiness: "
                f"{output.decode(errors='replace')}"
            ) from exc
        reader, writer = await asyncio.open_unix_connection(socket_path)
        await request(reader, writer, "hello", "hello", {"protocol_version": "1.1"})
        response = (
            await request(
                reader,
                writer,
                "new",
                "new_session",
                {"provider": "codex"},
            )
        )[-1]
        assert response["error"]["message"] == (
            "account codex:work is unavailable: credentials missing; "
            "run 'zeta accounts login codex work --replace'"
        )
    finally:
        try:
            if writer is not None:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), _PROCESS_TIMEOUT)
        finally:
            await _stop_process(process, output, "serve shutdown")
            Path(f"{socket_path}.lock").unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_console_serve_starts_when_only_non_active_accounts_are_missing(
    tmp_path: Path,
) -> None:
    current = AccountId("anthropic", "current")
    missing = AccountId("codex", "missing")
    await asyncio.to_thread(
        build_account_home,
        tmp_path,
        (current, missing),
        available=(current,),
        defaults=(current,),
    )
    socket_path = Path("/tmp") / f"zeta-accounts-{os.getpid()}.sock"
    process = await asyncio.create_subprocess_exec(
        _entrypoint(),
        "serve",
        "--provider",
        "claude",
        "--socket",
        socket_path,
        "--ws",
        "0",
        "--no-auto-memory",
        cwd=Path.cwd(),
        env=_environment(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output = bytearray()
    try:
        try:
            await asyncio.wait_for(_wait_until_ready(process, output), _PROCESS_TIMEOUT)
        except TimeoutError as exc:
            raise AssertionError(
                "timed out waiting for serve readiness: "
                f"{output.decode(errors='replace')}"
            ) from exc
        assert process.returncode is None
    finally:
        await _stop_process(process, output, "serve shutdown")
        Path(f"{socket_path}.lock").unlink(missing_ok=True)
