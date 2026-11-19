"""Production-path voice tests: real processes, real files, real ffmpeg when installed."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import wave
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from tests.support.owned import await_or_fail
from zeta.voice import audio, whisper_cpp
from zeta.voice.audio import (
    MAX_PCM_BYTES,
    MicrophoneSource,
    WavWriter,
    sweep_stale,
    validate_wav,
)
from zeta.voice.process import KILL_SECONDS, TERMINATE_SECONDS, OwnedProcess


def _reaped(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def _private_dir(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(mode=0o700, parents=True)
    (directory / audio.LOCK_NAME).touch(mode=0o600)
    return directory


def _start_time(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return result.stdout.strip()


def _record(directory: Path, pid: int, start: str) -> None:
    (directory / audio.PID_NAME).write_text(json.dumps({"pid": pid, "start": start}))


def _spawn_shell(script: str) -> tuple[subprocess.Popen[bytes], threading.Thread]:
    """A real `sh` group leader, reaped by a thread so it vanishes once signalled."""
    child = subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True)
    reaper = threading.Thread(target=child.wait, daemon=True)
    reaper.start()
    return child, reaper


def _end(child: subprocess.Popen[bytes], reaper: threading.Thread) -> None:
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    reaper.join(10)


@pytest.mark.asyncio
async def test_record_exists_before_ffmpeg_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a fake ffmpeg checks the record at its first instruction: a parent killed
    # at that moment must already have left an identity record behind
    bin_dir, directory, probe = tmp_path / "bin", tmp_path / "run", tmp_path / "probe"
    bin_dir.mkdir()
    directory.mkdir(mode=0o700)
    started = socket.create_server(("127.0.0.1", 0))  # fake ffmpeg connects once it has checked
    started.settimeout(10)
    fake = bin_dir / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, socket, sys\n"
        "try:\n"
        f"    ok = json.load(open({str(directory / audio.PID_NAME)!r}))['pid'] == os.getpid()\n"
        "except OSError:\n"
        "    ok = False\n"
        f"open({str(probe)!r}, 'w').write('recorded' if ok else 'missing')\n"
        f"socket.create_connection(('127.0.0.1', {started.getsockname()[1]})).close()\n"
        "sys.stdout.buffer.write(bytes(3200))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    identity = audio.process_identity

    def after_ffmpeg_started(pid: int) -> str | None:
        # forced order: any parent-side record step runs only after ffmpeg started
        started.accept()[0].close()
        return identity(pid)

    monkeypatch.setattr(audio, "process_identity", after_ffmpeg_started)
    source = MicrophoneSource()
    try:
        await await_or_fail(source.start(directory / audio.CAPTURE_NAME), what="fake ffmpeg capture")
    finally:
        await await_or_fail(source.close(), what="close fake capture")
        started.close()
    assert probe.read_text() == "recorded", "ffmpeg ran before its pid record existed"


def test_owned_process_reaps_child_that_exits_before_group_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    killpg = os.killpg
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    ) as child:
        stdin = child.stdin
        assert stdin is not None

        def kill_after_exit(pgid: int, sig: int) -> None:
            assert pgid == child.pid
            # Let the child exit after poll(), but keep it unreaped for the real killpg().
            stdin.close()
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            killpg(pgid, sig)

        monkeypatch.setattr(os, "killpg", kill_after_exit)
        OwnedProcess(child, 0, 0).kill_blocking()
        assert stdin.closed and child.returncode == 0 and _reaped(child.pid)


def test_owned_process_waits_after_eperm_while_child_is_not_waitable(monkeypatch: pytest.MonkeyPatch) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    ) as child:
        stdin = child.stdin
        assert stdin is not None
        wait = child.wait

        def deny_signal(pgid: int, sig: int) -> None:
            assert pgid == child.pid and sig == signal.SIGTERM
            assert child.poll() is None
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

        def finish_exit(timeout: float | None = None) -> int:
            if timeout is not None:
                # Release the child only once cleanup waits instead of polling again.
                stdin.write(b"x")
                stdin.flush()
            return wait(timeout=timeout)

        monkeypatch.setattr(os, "killpg", deny_signal)
        monkeypatch.setattr(child, "wait", finish_exit)
        error = None
        try:
            OwnedProcess(child, 0, 0).kill_blocking()
        except PermissionError as exc:
            error = exc
        assert error is None, "cleanup must wait after EPERM even when poll() still returns None"
        assert stdin.closed and child.returncode == 0 and _reaped(child.pid)


def test_owned_process_reports_child_alive_after_sigkill_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    child = Mock(spec=subprocess.Popen, pid=123)
    child.poll.return_value = None
    child.stdin = Mock()
    child.wait.side_effect = subprocess.TimeoutExpired("voice child", 2)
    deny_signal = Mock(side_effect=PermissionError(errno.EPERM, os.strerror(errno.EPERM)))
    monkeypatch.setattr(os, "killpg", deny_signal)

    with pytest.raises(RuntimeError, match="voice child pid 123 did not exit by the SIGKILL deadline"):
        OwnedProcess(child, 0, 0).kill_blocking()

    assert deny_signal.call_args_list == [call(123, signal.SIGTERM), call(123, signal.SIGKILL)]
    assert child.wait.call_args_list == [call(timeout=TERMINATE_SECONDS), call(timeout=KILL_SECONDS)]
    child.stdin.close.assert_called_once_with()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.asyncio
async def test_microphone_source_captures_fixture_into_private_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.wav"
    with wave.open(str(fixture), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(bytes(16_000 * 2 * 5))  # 5 s of silence; stop ends it early
    monkeypatch.setenv("ZETA_VOICE_INPUT_FILE", str(fixture))
    directory = tmp_path / "run"
    directory.mkdir(mode=0o700)
    capture = directory / audio.CAPTURE_NAME
    source = MicrophoneSource()
    try:
        await await_or_fail(source.start(capture), what="ffmpeg first frame")
        record = json.loads((directory / audio.PID_NAME).read_text())
        assert audio.is_recorded_process(record["pid"], record["start"])  # real ffmpeg, pid kept across exec
        await await_or_fail(source.stop(), what="ffmpeg stop")
    finally:
        await await_or_fail(source.close(), what="close capture")
    assert _reaped(record["pid"])
    assert not (directory / audio.PID_NAME).exists()
    assert capture.stat().st_mode & 0o777 == 0o600
    validate_wav(capture)
    assert capture.stat().st_size < 44 + 16_000 * 2 * 5  # stopped before the fixture ended


def test_wav_writer_stops_storing_at_the_cap(tmp_path: Path) -> None:
    writer = WavWriter(tmp_path / "capture.wav", lambda: None)
    try:
        for _ in range(MAX_PCM_BYTES // 640_000 + 2):
            writer.write(bytes(640_000))
    finally:
        writer.close()
    assert writer.path.stat().st_size == 44 + MAX_PCM_BYTES
    validate_wav(writer.path)


@pytest.mark.parametrize("script", ["sleep 60; :", "trap '' TERM; while :; do sleep 1; done"], ids=["term", "kill"])
def test_stale_sweep_ends_the_recorded_process(tmp_path: Path, script: str) -> None:
    child, reaper = _spawn_shell(script)
    try:
        directory = _private_dir(tmp_path, "stale")
        _record(directory, child.pid, _start_time(child.pid))
        sweep_stale(tmp_path)
        reaper.join(10)
        assert child.returncode == (-signal.SIGTERM if script.startswith("sleep") else -signal.SIGKILL)
        assert not directory.exists()
    finally:
        _end(child, reaper)


def test_stale_sweep_spares_mismatched_and_locked_runtimes(tmp_path: Path) -> None:
    child, reaper = _spawn_shell("sleep 60; :")
    lock_fd = None
    try:
        mismatched = _private_dir(tmp_path, "mismatch")
        _record(mismatched, child.pid, "Thu Jan  1 00:00:00 1970")
        locked = _private_dir(tmp_path, "locked")
        _record(locked, child.pid, _start_time(child.pid))
        lock_fd = os.open(locked / audio.LOCK_NAME, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sweep_stale(tmp_path)
        assert child.poll() is None, "sweep signalled a process it does not own"
        assert not mismatched.exists()  # only the record and its directory were removed
        assert (locked / audio.PID_NAME).exists()
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        _end(child, reaper)


def _serve(tmp_path: Path, body: bytes, monkeypatch: pytest.MonkeyPatch, *, digest: str | None = None) -> str:
    source = tmp_path / "remote.bin"
    source.write_bytes(body)
    monkeypatch.setattr(whisper_cpp, "MODEL_SHA256", digest or hashlib.sha256(body).hexdigest())
    return source.as_uri()


def test_model_download_publishes_only_a_verified_private_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = _serve(tmp_path, b"model weights", monkeypatch)
    target = tmp_path / "cache" / "ggml-small.en.bin"
    whisper_cpp.download_model(target, threading.Event(), url)
    assert target.read_bytes() == b"model weights"
    assert target.stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]


@pytest.mark.parametrize("fault", ["digest", "size", "connection"])
def test_model_download_rejects_bad_payload_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    url = _serve(tmp_path, b"model weights", monkeypatch, digest="0" * 64 if fault == "digest" else None)
    if fault == "size":
        monkeypatch.setattr(whisper_cpp, "MAX_MODEL_BYTES", 4)
    if fault == "connection":
        opened: list[int] = []
        real_open = whisper_cpp.os.open

        def capture_open(*args: object, **kwargs: object) -> int:
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        monkeypatch.setattr(whisper_cpp.os, "open", capture_open)
        monkeypatch.setattr(
            whisper_cpp.urllib.request,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("connection failed")),
        )
    target = tmp_path / "cache" / "ggml-small.en.bin"
    with pytest.raises((RuntimeError, OSError), match="sha256|exceeds|connection"):
        whisper_cpp.download_model(target, threading.Event(), url)
    assert list(target.parent.iterdir()) == []
    if fault == "connection":
        assert opened
        with pytest.raises(OSError):
            os.fstat(opened[0])


@pytest.mark.parametrize("cached", [b"model weights", b"tampered"], ids=["verified", "tampered"])
@pytest.mark.asyncio
async def test_prepare_reuses_cached_model_only_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bytes
) -> None:
    monkeypatch.setattr(whisper_cpp, "MODEL_SHA256", hashlib.sha256(b"model weights").hexdigest())
    downloads: list[Path] = []
    monkeypatch.setattr(whisper_cpp, "download_model", lambda path, _stop: downloads.append(path))
    transcriber = whisper_cpp.WhisperCpp(zeta_home=tmp_path)
    model = tmp_path / "cache" / "voice" / "whisper.cpp" / "ggml-small.en.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(cached)
    await await_or_fail(transcriber.prepare(), what="prepare")
    assert downloads == ([] if cached == b"model weights" else [model])
