"""The computer MCP server: line-delimited JSON-RPC over stdio.

Run as ``python -m zeta.computer.server``. Zeta starts it for a computer
session and passes its configuration in the environment:

``ZETA_COMPUTER_SESSION``        session ID (required; labels the desktop)
``ZETA_COMPUTER_BACKEND``        backend name, default ``local``
``ZETA_COMPUTER_TTL_SECONDS``    desktop lifetime, default 3600
``ZETA_COMPUTER_RECORDING_DIR``  recording directory; absent disables recording
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from .actions import MODEL_HEIGHT, MODEL_WIDTH, validate_action, validate_batch
from .backend import DEFAULT_BACKEND, DesktopBackend, DesktopOptions, create_backend
from .observe import format_observation
from .recording import SessionRecorder
from .tools import SINGLE_ACTIONS, TOOLS

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TTL_SECONDS = 3600
Result = dict[str, object]
# Failures a tool reports to the model instead of ending the server.
ACTION_ERRORS = (KeyError, ValueError, RuntimeError, OSError, subprocess.SubprocessError)


class ComputerServer:
    """Dispatch tool calls to one backend and shape their MCP results."""

    def __init__(self, backend: DesktopBackend, recorder: SessionRecorder | None = None) -> None:
        self.backend = backend
        self.recorder = recorder
        self._previous_hash: str | None = None

    def call(self, name: str, arguments: object) -> Result:
        if type(arguments) is not dict:
            result = self.error("arguments must be an object")
            arguments = {}
        else:
            result = self._call(name, arguments)
        if self.recorder is not None:
            self.recorder.record_tool(name, arguments, result)
        return result

    def _call(self, name: str, arguments: dict[str, object]) -> Result:
        try:
            if name == "batch":
                actions = validate_batch(arguments)
                self.backend.start()
                return self._batch(actions, screenshot=arguments.get("screenshot", True) is True)
            if name == "screenshot":
                if arguments:
                    raise ValueError("screenshot takes no arguments")
                self.backend.start()
                return self._screenshot([])
            if name not in SINGLE_ACTIONS:
                raise ValueError(f"unknown tool: {name}")
            validated = validate_action(name, arguments)
            self.backend.start()
            self.backend.input(name, validated)
            return self._screenshot([self._settle()])
        except ACTION_ERRORS as exc:
            return self.error(str(exc))

    def _settle(self) -> str:
        return f"Screen settled in {self.backend.settle():.3f} seconds."

    def _batch(self, actions: list[tuple[str, dict[str, object]]], *, screenshot: bool) -> Result:
        results: list[dict[str, object]] = []
        failed = False
        for index, (action, action_arguments) in enumerate(actions):
            try:
                self.backend.input(action, action_arguments)
            except ACTION_ERRORS as exc:
                results.append({"index": index, "type": action, "status": "error", "error": str(exc)})
                failed = True
                break
            results.append({"index": index, "type": action, "status": "ok"})
        messages = ["Batch results: " + json.dumps(results, separators=(",", ":")), self._settle()]
        if screenshot:
            return self._screenshot(messages, is_error=failed)
        return {
            "content": [{"type": "text", "text": message} for message in messages],
            "isError": failed,
        }

    def _screenshot(self, messages: list[str], *, is_error: bool = False) -> Result:
        frame = self.backend.screenshot()
        if self.recorder is not None:
            self.recorder.record_frame(frame.data)
        observation, self._previous_hash = format_observation(
            self.backend.observe(), frame.data, self._previous_hash
        )
        text = [
            f"Fresh {MODEL_WIDTH}x{MODEL_HEIGHT} screenshot ({len(frame.data)} bytes).",
            *messages,
            "Observation: " + observation,
        ]
        return {
            "content": [
                *({"type": "text", "text": item} for item in text),
                {
                    "type": "image",
                    "data": base64.b64encode(frame.data).decode("ascii"),
                    "mimeType": frame.media_type,
                },
            ],
            "isError": is_error,
        }

    @staticmethod
    def error(message: str) -> Result:
        return {"content": [{"type": "text", "text": message}], "isError": True}


def _respond(sink: TextIO, payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")), file=sink, flush=True)


def handle_request(server: ComputerServer, request: Mapping[str, object]) -> object:
    """Return the JSON-RPC result for one request; raise ``ValueError`` on bad input."""

    method = request.get("method")
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "zeta-computer", "version": "1"},
        }
    if method == "tools/list":
        return {"tools": [dict(tool) for tool in TOOLS]}
    if method == "tools/call":
        params = request.get("params")
        if type(params) is not dict or type(params.get("name")) is not str:
            raise ValueError("tools/call requires a tool name")
        return server.call(params["name"], params.get("arguments", {}))
    if method in {"shutdown", "ping"}:
        return {}
    raise ValueError(f"unsupported method: {method}")


def serve(server: ComputerServer, source: TextIO, sink: TextIO) -> None:
    """Answer requests until EOF. The caller owns backend cleanup."""

    for line in source:
        request_id: object = None
        try:
            request = json.loads(line)
            if type(request) is not dict:
                continue
            request_id = request.get("id")
            if request.get("method") == "notifications/exit":
                return
            if request_id is None:
                continue
            _respond(sink, {"jsonrpc": "2.0", "id": request_id, "result": handle_request(server, request)})
        except (ValueError, json.JSONDecodeError) as exc:
            if request_id is not None:
                _respond(
                    sink,
                    {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(exc)}},
                )


def _terminate(_signum: int, _frame: object) -> None:
    raise SystemExit(0)


def main() -> int:
    session_id = os.environ.get("ZETA_COMPUTER_SESSION", "")
    if not session_id:
        print("zeta computer server: ZETA_COMPUTER_SESSION is required", file=sys.stderr)
        return 2
    options = DesktopOptions(
        session_id=session_id,
        ttl_seconds=int(os.environ.get("ZETA_COMPUTER_TTL_SECONDS", DEFAULT_TTL_SECONDS)),
    )
    backend = create_backend(os.environ.get("ZETA_COMPUTER_BACKEND", DEFAULT_BACKEND), options)
    recording = os.environ.get("ZETA_COMPUTER_RECORDING_DIR")
    recorder = SessionRecorder(Path(recording)) if recording else None
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    try:
        serve(ComputerServer(backend, recorder), sys.stdin, sys.stdout)
    finally:
        # Zeta may kill this process soon after SIGTERM, so finish the cheap
        # local write first. The session also removes the desktop host-side.
        try:
            if recorder is not None:
                recorder.close()
        finally:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
