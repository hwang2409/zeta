"""Client for the native macOS computer-use helper.

Spawns the signed Swift helper (``helpers/macos-computer``) and speaks its JSON-lines stdio
protocol: one request object per line, one response object per line. Requests are serialized;
each carries an incrementing ``id`` that the response echoes. Helper error responses are raised
as :class:`HelperError` with the protocol ``code`` (``locked``, ``secure_input``,
``not_permitted``, ``killed``, ``bad_request``, ``internal``). A slow or crashed helper raises
:class:`HelperTimeout` or :class:`HelperProtocolError` instead of hanging.

The real ``MacBackend`` (added once slice 1 lands) wraps this client; the client itself knows
nothing about the ``ComputerBackend`` protocol so it stays small and directly testable against a
fake helper process.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
from typing import Self

DEFAULT_TIMEOUT = 10.0
_READ_CHUNK = 65536


class HelperError(Exception):
    """The helper returned ``ok: false``. ``code`` is the protocol error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class HelperTimeout(Exception):
    """The helper did not answer a request within the timeout."""


class HelperProtocolError(Exception):
    """The helper closed early or emitted output that is not a valid response line."""


class HelperClient:
    """Owns one helper subprocess. Not safe to share across threads concurrently; requests are
    serialized with a lock so overlapping callers queue rather than interleave on the pipe."""

    def __init__(self, argv: list[str] | str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._argv = [argv] if isinstance(argv, str) else list(argv)
        self._timeout = timeout
        self._lock = threading.Lock()
        self._next_id = 1
        self._proc: subprocess.Popen[bytes] | None = None
        self._sel: selectors.BaseSelector | None = None
        self._buf = b""

    # lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("helper already started")
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._proc.stdout, selectors.EVENT_READ)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        # closing stdin ends the helper's readLine loop; terminate/kill if it lingers.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._sel is not None:
            self._sel.close()
            self._sel = None
        self._proc = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # requests ------------------------------------------------------------------

    def request(self, op: str, *, timeout: float | None = None, **args: object) -> dict:
        """Send one op and return its response payload (without ``id``/``ok``).

        Raises :class:`HelperError` on an ``ok: false`` response, :class:`HelperTimeout` when the
        helper is silent past the deadline, and :class:`HelperProtocolError` when it dies or
        replies with something unparseable.
        """
        if self._proc is None or self._sel is None:
            raise RuntimeError("helper not started")
        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            payload = {"id": request_id, "op": op, **args}
            self._write(payload)
            while True:
                line = self._read_line(deadline)
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HelperProtocolError(f"invalid response line: {line!r}") from exc
                if not isinstance(response, dict):
                    raise HelperProtocolError(f"response not an object: {line!r}")
                if response.get("id") != request_id:
                    # a stale line for an earlier request; skip it and keep reading.
                    continue
                if not response.get("ok"):
                    error = response.get("error") or {}
                    raise HelperError(
                        str(error.get("code", "internal")),
                        str(error.get("message", "")),
                    )
                return {k: v for k, v in response.items() if k not in ("id", "ok")}

    # framing -------------------------------------------------------------------

    def _write(self, payload: dict) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        data = json.dumps(payload).encode("utf-8") + b"\n"
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise HelperProtocolError("helper stdin closed") from exc

    def _read_line(self, deadline: float) -> bytes:
        proc = self._proc
        sel = self._sel
        assert proc is not None and proc.stdout is not None and sel is not None
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HelperTimeout(f"helper did not respond within {self._timeout}s")
            if not sel.select(remaining):
                raise HelperTimeout(f"helper did not respond within {self._timeout}s")
            chunk = os.read(proc.stdout.fileno(), _READ_CHUNK)
            if not chunk:
                raise HelperProtocolError("helper closed stdout")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line
