"""Local-only spectating: a recording viewer and a live VNC bridge.

Both listeners bind to 127.0.0.1 only. The viewer requires a random URL token
and loads no external assets. The VNC bridge reaches the guest through
``docker exec`` standard I/O, so the desktop gets no port, mount, or network
interface. The VNC password reaches the guest through standard input, never
through a command argument.
"""

from __future__ import annotations

import json
import re
import secrets
import signal
import socket
import string
import subprocess
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .actions import MODEL_HEIGHT, MODEL_WIDTH
from .docker import DockerClient
from .x11 import DISPLAY

LIVE_WINDOW_SECONDS = 300.0
VNC_PORT = 5900
FRAME_PATH = re.compile(r"^/frames/(\d{6}\.jpg)$")
CONTAINER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def safe_json(value: object) -> bytes:
    """Encode JSON so hostile strings cannot form HTML if a client sniffs it."""

    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").encode()
    )


def is_live(directory: Path, metadata: dict[str, object], now: float) -> bool:
    """A recording is live while it is open and still receives events."""

    if not metadata.get("active"):
        return False
    for name in ("events.jsonl", "metadata.json"):
        try:
            return now - (directory / name).stat().st_mtime < LIVE_WINDOW_SECONDS
        except OSError:
            continue
    return False


def load_recording(directory: Path) -> dict[str, object]:
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if type(metadata) is not dict:
            raise ValueError("metadata must be an object")
    except (OSError, ValueError):
        metadata = {"active": False}
    metadata["active"] = is_live(directory, metadata, time.time())
    events: list[object] = []
    try:
        lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if type(event) is dict:
            events.append(event)
    return {
        "metadata": metadata,
        "model_frame": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
        "events": events,
    }


def _viewer() -> bytes:
    return (resources.files("zeta.computer") / "assets" / "viewer.html").read_bytes()


def make_web_server(directory: Path, token: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Create a token-protected viewer bound to 127.0.0.1 on a free port."""

    directory = directory.resolve()
    access_token = token or secrets.token_urlsafe(24)
    page = _viewer()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            supplied = parse_qs(parsed.query).get("token", [])
            if len(supplied) != 1 or not secrets.compare_digest(supplied[0], access_token):
                self.send_error(403)
                return
            if parsed.path == "/":
                self._send("text/html; charset=utf-8", page)
                return
            if parsed.path == "/api/state":
                self._send("application/json", safe_json(load_recording(directory)))
                return
            match = FRAME_PATH.fullmatch(parsed.path)
            if match is None:
                self.send_error(404)
                return
            try:
                data = (directory / "frames" / match.group(1)).read_bytes()
            except OSError:
                self.send_error(404)
                return
            self._send("image/jpeg", data)

        def _send(self, content_type: str, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler), access_token


class Spectator:
    """A viewer that serves one recording from a daemon thread."""

    def __init__(self, directory: Path) -> None:
        self._server, token = make_web_server(directory)
        port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{port}/?token={quote(token)}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="zeta-computer-spectator", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def tunnel_command(docker: DockerClient, container: str) -> list[str]:
    """Return the stdio bridge argv to the guest's loopback VNC port."""

    if not CONTAINER_ID.fullmatch(container):
        raise ValueError("invalid container id")
    return docker.command("exec", "-i", container, "socat", "-", f"TCP:127.0.0.1:{VNC_PORT}")


def vnc_command(container: str, password_file: str) -> list[str]:
    """Return the always-view-only x11vnc command."""

    if not CONTAINER_ID.fullmatch(container):
        raise ValueError("invalid container id")
    return [
        "exec", "-d", "-e", f"DISPLAY={DISPLAY}", container, "x11vnc", "-display", DISPLAY,
        "-localhost", "-rfbport", str(VNC_PORT), "-forever", "-shared", "-passwdfile",
        password_file, "-viewonly",
    ]


def _vnc_listening(docker: DockerClient, container: str) -> bool:
    probe = docker.run(
        "exec", container, "socat", "-T", "1", "-u", "/dev/null", f"TCP:127.0.0.1:{VNC_PORT}",
        timeout=30, check=False,
    )
    return probe.returncode == 0


def _stop_vnc(docker: DockerClient, container: str, password_file: str) -> None:
    docker.run(
        "exec", "-e", f"DISPLAY={DISPLAY}", container, "x11vnc", "-display", DISPLAY,
        "-remote", "stop", timeout=30, check=False,
    )
    docker.run("exec", container, "rm", "-f", password_file, timeout=30, check=False)


def _interrupt(signum: int, _frame: object) -> None:
    signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt


def run_live(
    docker: DockerClient,
    container: str,
    *,
    announce: Callable[[str], None] = print,
) -> None:
    """Bridge a random 127.0.0.1 port to a password VNC server in the guest.

    Blocks until Ctrl-C or until the container is gone, then removes the VNC
    server and its password file. A second bridge to the same container fails.
    """

    if not CONTAINER_ID.fullmatch(container):
        raise ValueError("invalid container id")
    if _vnc_listening(docker, container):
        raise RuntimeError("a live view is already open for this desktop")
    alphabet = string.ascii_letters + string.digits
    password = "".join(secrets.choice(alphabet) for _ in range(8))
    password_file = f"/tmp/zeta-vnc-{secrets.token_hex(8)}"
    docker.run("exec", "-i", container, "sh", "-c", 'umask 077 && cat > "$1"', "sh", password_file,
               stdin=password.encode(), timeout=30)
    command = vnc_command(container, password_file)
    listener = socket.socket()
    clients: list[subprocess.Popen[bytes]] = []
    try:
        docker.run(*command, timeout=30)
        for _ in range(50):
            if _vnc_listening(docker, container):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("x11vnc did not become ready")
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(1)
        announce(f"live view: vnc://127.0.0.1:{listener.getsockname()[1]}")
        announce(f"password: {password}")
        announce("view only")
        previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while docker.run("inspect", container, timeout=30, check=False).returncode == 0:
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                process = subprocess.Popen(
                    tunnel_command(docker, container),
                    stdin=connection,
                    stdout=connection,
                    stderr=subprocess.DEVNULL,
                    env=docker.env,
                )
                connection.close()
                clients[:] = [item for item in clients if item.poll() is None] + [process]
        except KeyboardInterrupt:
            pass
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    finally:
        listener.close()
        for process in clients:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        _stop_vnc(docker, container, password_file)


__all__ = [
    "Spectator",
    "is_live",
    "load_recording",
    "make_web_server",
    "run_live",
    "safe_json",
    "tunnel_command",
    "vnc_command",
]
