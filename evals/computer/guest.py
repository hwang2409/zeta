"""One MCP command or browser tool inside a disposable computer container."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any


def _reply(
    request_id: object,
    result: dict[str, Any] | None = None,
    *,
    error: str | None = None,
) -> None:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        message["result"] = result
    else:
        message["error"] = {"code": -32601, "message": error}
    print(json.dumps(message, separators=(",", ":")), flush=True)


def _command(arguments: object) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    command = arguments.get("command")
    timeout = arguments.get("timeout", 30)
    if type(command) is not str or not command or len(command) > 8000:
        raise ValueError("command must be 1 to 8000 characters")
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("timeout must be an integer from 1 to 300 seconds")
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            ["/bin/sh", "-lc", command],
            cwd="/workspace",
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/workspace"},
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        stdout.seek(0)
        stderr.seek(0)
        output = (
            f"exit_code={process.returncode}\n"
            f"stdout:\n{stdout.read(10_000).decode(errors='replace')}\n"
            f"stderr:\n{stderr.read(10_000).decode(errors='replace')}"
        )
        if timed_out:
            output += f"\n[timed out after {timeout}s]"
    return {
        "content": [{"type": "text", "text": output}],
        "isError": timed_out or process.returncode != 0,
        "structuredContent": {"exit_code": process.returncode, "timed_out": timed_out},
    }


def main() -> None:
    browser = None
    if sys.argv[1:] == ["--browser"] or (
        len(sys.argv) == 3 and sys.argv[1] == "--browser-public"
    ):
        from browser_guest import TOOL, BrowserGuest

        public_host = sys.argv[2] if len(sys.argv) == 3 else None
        browser = BrowserGuest(public_host)
        browser_tool = (
            TOOL
            | {
                "description": f"Control sandboxed Chromium on https://{public_host}/ through a GET-only broker. Open, inspect, find, or act by role/name. No shell or direct network is available."
            }
            if public_host
            else TOOL
        )
    elif sys.argv[1:]:
        raise SystemExit("unsupported guest mode")
    try:
        for line in sys.stdin:
            request_id: object = None
            try:
                request = json.loads(line)
                if type(request) is not dict:
                    continue
                request_id = request.get("id")
                method = request.get("method")
                if method == "notifications/initialized":
                    continue
                if method == "initialize":
                    _reply(
                        request_id,
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "zeta-computer-eval",
                                "version": "0.1",
                            },
                        },
                    )
                elif method == "tools/list":
                    _reply(
                        request_id,
                        {
                            "tools": [browser_tool]
                            if browser is not None
                            else [
                                {
                                    "name": "bash",
                                    "description": "Run one command in the isolated /workspace computer.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "command": {
                                                "type": "string",
                                                "minLength": 1,
                                            },
                                            "timeout": {
                                                "type": "integer",
                                                "minimum": 1,
                                                "maximum": 300,
                                            },
                                        },
                                        "required": ["command"],
                                        "additionalProperties": False,
                                    },
                                }
                            ]
                        },
                    )
                elif method == "tools/call":
                    params = request.get("params")
                    if type(params) is not dict or params.get("name") != (
                        "browser" if browser else "bash"
                    ):
                        raise ValueError("unknown tool")
                    try:
                        result = (
                            browser.call(params.get("arguments"))
                            if browser is not None
                            else _command(params.get("arguments"))
                        )
                    except Exception as exc:  # noqa: BLE001 - report tool failures over MCP
                        result = {
                            "content": [{"type": "text", "text": str(exc)}],
                            "isError": True,
                        }
                    _reply(request_id, result)
                elif request_id is not None:
                    _reply(request_id, error=f"unsupported method: {method}")
            except (TypeError, ValueError) as exc:
                if request_id is not None:
                    _reply(request_id, error=f"invalid request: {exc}")
    finally:
        if browser is not None:
            browser.close()


if __name__ == "__main__":
    main()
