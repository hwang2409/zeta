#!/usr/bin/env python3
"""Local-only live VNC tunnel and recording viewer."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import socket
import string
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from backend import DEFAULT_DOCKER_HOST, MODEL_HEIGHT, MODEL_WIDTH

CONTAINER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LIVE_WINDOW_SECONDS = 15.0
FRAME = re.compile(r"^/frames/(\d{6}\.jpg)$")

VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zeta computer spectator</title><style>
:root{color-scheme:dark;background:#111;color:#eee;font:14px system-ui,sans-serif}body{margin:0;display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);height:100vh}main{display:flex;flex-direction:column;min-width:0;padding:14px;gap:10px}.screen{background:#050505;min-height:0;flex:1;display:grid;place-items:center}canvas{max-width:100%;max-height:100%;box-shadow:0 0 0 1px #444}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:180px}aside{border-left:1px solid #333;padding:14px;overflow:auto}h1,h2{font-size:16px;margin:0 0 10px}.stats{color:#9cc;margin-bottom:12px}.event{border-top:1px solid #333;padding:8px 0;white-space:pre-wrap;overflow-wrap:anywhere}.active{color:#7ee787}.warning{color:#ffb86c}.done{text-decoration:line-through;color:#888}ul{padding-left:22px}@media(max-width:760px){body{grid-template-columns:1fr;grid-template-rows:minmax(55vh,2fr) 1fr}aside{border-left:0;border-top:1px solid #333}}
</style></head><body><main><div class="bar"><strong id="status">Loading</strong><span id="stats"></span></div><div class="screen"><canvas id="screen" width="1024" height="640"></canvas></div><div class="bar"><button id="prev">◀</button><button id="play">Play</button><button id="next">▶</button><input id="scrub" type="range" min="0" max="0" value="0"><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div></main><aside><h1>Checklist</h1><ul id="checklist"></ul><h2>Action timeline</h2><div id="timeline"></div></aside><script>
'use strict';const token=new URLSearchParams(location.search).get('token');const canvas=document.getElementById('screen'),ctx=canvas.getContext('2d'),img=new Image();let state={events:[],model_frame:{width:1024,height:640}},index=0,timer=null;
function modelFrame(){const f=state.model_frame;return f&&f.width&&f.height?f:{width:1024,height:640}}
function text(el,value){el.textContent=value==null?'':String(value)}
function points(event){const c=event&&event.args&&event.args._coordinates;if(!c)return[];return['start','end','point'].filter(k=>c[k]&&c[k].model).map(k=>({kind:k,x:c[k].model.x,y:c[k].model.y}))}
function draw(){const event=state.events[index];ctx.clearRect(0,0,canvas.width,canvas.height);let frameEvent=null;for(let i=index;i>=0;i--){if(state.events[i].frame){frameEvent=state.events[i];break}}if(!frameEvent)return;img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);const frame=modelFrame();const marks=points(event).map(p=>({...p,x:p.x*canvas.width/frame.width,y:p.y*canvas.height/frame.height}));ctx.strokeStyle='#ff3155';ctx.lineWidth=4;if(marks.length===2&&marks[0].kind==='start'){ctx.beginPath();ctx.moveTo(marks[0].x,marks[0].y);ctx.lineTo(marks[1].x,marks[1].y);ctx.stroke()}for(const p of marks){ctx.beginPath();ctx.arc(p.x,p.y,12,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(p.x-18,p.y);ctx.lineTo(p.x+18,p.y);ctx.moveTo(p.x,p.y-18);ctx.lineTo(p.x,p.y+18);ctx.stroke()}};img.src=frameEvent.frame+'?token='+encodeURIComponent(token)+'&v='+encodeURIComponent(frameEvent.ts)}
function render(){index=Math.max(0,Math.min(index,state.events.length-1));const meta=state.metadata||{};text(document.getElementById('status'),meta.active?'LIVE':'REPLAY');document.getElementById('status').className=meta.active?'active':'';const frames=state.events.filter(e=>e.frame).length,elapsed=state.events.length?state.events[state.events.length-1].elapsed:0;text(document.getElementById('stats'),state.events.length+' steps · '+frames+' screenshots · '+Number(elapsed).toFixed(1)+'s');const scrub=document.getElementById('scrub');scrub.max=Math.max(0,state.events.length-1);scrub.value=index;const checklist=document.getElementById('checklist');checklist.replaceChildren();const current=state.events[index]||{},list=current.checklist||[];for(const item of list){const li=document.createElement('li');text(li,(item.done?'✓ ':'○ ')+item.step+(item.note?' — '+item.note:''));if(item.done)li.className='done';checklist.append(li)}const timeline=document.getElementById('timeline');timeline.replaceChildren();state.events.forEach((event,i)=>{const div=document.createElement('div');div.className='event'+(event.verify_warnings&&event.verify_warnings.length?' warning':'');text(div,(i+1)+'. '+event.tool+'\n'+JSON.stringify(event.args));div.onclick=()=>{index=i;render()};timeline.append(div)});draw()}
async function refresh(){const response=await fetch('/api/state?token='+encodeURIComponent(token),{cache:'no-store'});if(!response.ok){text(document.getElementById('status'),'Access denied');return}const follow=index>=state.events.length-1;state=await response.json();if(follow)index=Math.max(0,state.events.length-1);render()}
document.getElementById('prev').onclick=()=>{index--;render()};document.getElementById('next').onclick=()=>{index++;render()};document.getElementById('scrub').oninput=e=>{index=Number(e.target.value);render()};document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;text(e.target,'Play');return}text(e.target,'Pause');timer=setInterval(()=>{if(index>=state.events.length-1){clearInterval(timer);timer=null;text(e.target,'Play')}else{index++;render()}},1000/Number(document.getElementById('speed').value))};refresh();setInterval(refresh,1000);
</script></body></html>"""


def safe_json(value: object) -> bytes:
    """Encode JSON so hostile strings cannot form HTML tags if mis-sniffed."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .encode()
    )


def overlay_position(
    x: float, y: float, width: int, height: int
) -> tuple[float, float]:
    """Scale a model-frame marker to the displayed frame."""
    return x * width / MODEL_WIDTH, y * height / MODEL_HEIGHT


def is_live(directory: Path, metadata: dict[str, object], now: float) -> bool:
    """A run counts as live while its open recording still receives events."""
    if not metadata.get("active"):
        return False
    try:
        modified = (directory / "events.jsonl").stat().st_mtime
    except OSError:
        return False
    return now - modified < LIVE_WINDOW_SECONDS


def _load_recording(directory: Path) -> dict[str, object]:
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
    except (OSError, ValueError):
        metadata = {"active": False, "model_frame": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT}}
    metadata["active"] = is_live(directory, metadata, time.time())
    events = []
    try:
        lines = (directory / "events.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if type(event) is dict:
            events.append(event)
    return {"metadata": metadata, "model_frame": metadata.get("model_frame", {}), "events": events}


def make_web_server(
    directory: Path, token: str | None = None
) -> tuple[ThreadingHTTPServer, str]:
    """Create a token-protected viewer bound only to IPv4 loopback."""
    directory = directory.resolve()
    access_token = token or secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parse_qs(parsed.query).get("token") != [access_token]:
                self.send_error(403)
                return
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", VIEWER.encode())
                return
            if parsed.path == "/api/state":
                self._send(200, "application/json", safe_json(_load_recording(directory)))
                return
            match = FRAME.fullmatch(parsed.path)
            if match:
                frame = directory / "frames" / match.group(1)
                try:
                    data = frame.read_bytes()
                except OSError:
                    self.send_error(404)
                    return
                self._send(200, "image/jpeg", data)
                return
            self.send_error(404)

        def _send(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler), access_token


def _docker_config() -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    configured = os.environ.get("ZETA_COMPUTER_DOCKER_CONFIG") or os.environ.get("DOCKER_CONFIG")
    if configured:
        path = Path(configured)
        return path, None
    temporary = tempfile.TemporaryDirectory(prefix="zeta-spectate-docker-")
    path = Path(temporary.name)
    (path / "config.json").write_text("{}\n")
    return path, temporary


def docker_command(config: Path, docker_host: str, *arguments: str) -> list[str]:
    return ["docker", "--host", docker_host, "--config", str(config), *arguments]


def tunnel_command(config: Path, docker_host: str, container: str) -> list[str]:
    """Construct the stdio bridge without interpolation or a shell."""
    if not CONTAINER.fullmatch(container):
        raise ValueError("invalid container id or name")
    return docker_command(config, docker_host, "exec", "-i", container, "socat", "-", "TCP:127.0.0.1:5900")


def _run_docker(config: Path, host: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(docker_command(config, host, *args), capture_output=True, check=check, timeout=30)


def _write_guest_file(
    config: Path, host: str, container: str, path: str, content: str
) -> None:
    """Send a secret into the guest through stdin, never through argv."""
    subprocess.run(
        docker_command(config, host, "exec", "-i", container, "tee", path),
        input=content.encode(),
        capture_output=True,
        check=True,
        timeout=30,
    )
    _run_docker(config, host, "exec", container, "chmod", "600", path)


def _select_container(config: Path, host: str, requested: str | None) -> str:
    if requested:
        if not CONTAINER.fullmatch(requested):
            raise SystemExit("invalid container id or name")
        return requested
    result = _run_docker(config, host, "ps", "--filter", "label=zeta.computer-mcp=true", "--format", "{{.ID}}")
    containers = result.stdout.decode().split()
    if len(containers) != 1:
        raise SystemExit(f"expected one running computer sandbox, found {len(containers)}; pass a container id")
    return containers[0]


def close_tunnel(listener: socket.socket, clients: list[subprocess.Popen[bytes]]) -> None:
    """Close the listener and all active stdio bridge processes."""
    listener.close()
    for process in clients:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def _stop_vnc(config: Path, host: str, container: str, password_file: str) -> None:
    _run_docker(
        config,
        host,
        "exec",
        "-e",
        "DISPLAY=:99",
        container,
        "x11vnc",
        "-display",
        ":99",
        "-remote",
        "stop",
        check=False,
    )
    _run_docker(config, host, "exec", container, "rm", "-f", password_file, check=False)


def _interrupt(signum: int, frame: object) -> None:
    """Treat the first stop request like Ctrl-C, then protect the cleanup."""
    signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt


def _vnc_listening(config: Path, host: str, container: str) -> bool:
    """Report whether the guest already serves RFB on its own loopback port."""
    probe = _run_docker(
        config,
        host,
        "exec",
        container,
        "socat",
        "-T",
        "1",
        "-u",
        "/dev/null",
        "TCP:127.0.0.1:5900",
        check=False,
    )
    return probe.returncode == 0


def live(container: str | None, *, control: bool) -> int:
    config, temporary = _docker_config()
    host = os.environ.get("ZETA_COMPUTER_DOCKER_HOST", os.environ.get("DOCKER_HOST", DEFAULT_DOCKER_HOST))
    selected = _select_container(config, host, container)
    if _vnc_listening(config, host, selected):
        raise SystemExit(
            "a VNC server already listens inside this container; "
            "stop the other spectate session before starting a new one"
        )
    password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    password_file = f"/tmp/zeta-vnc-{secrets.token_hex(8)}"
    _write_guest_file(config, host, selected, password_file, password)
    command = ["exec", "-d", "-e", "DISPLAY=:99", selected, "x11vnc", "-display", ":99", "-localhost", "-rfbport", "5900", "-forever", "-shared", "-passwdfile", password_file]
    if not control:
        command.append("-viewonly")
    _run_docker(config, host, *command)
    for _ in range(50):
        if _vnc_listening(config, host, selected):
            break
        time.sleep(0.1)
    else:
        _stop_vnc(config, host, selected, password_file)
        if temporary is not None:
            temporary.cleanup()
        raise RuntimeError("x11vnc did not become ready")
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(1)
    port = listener.getsockname()[1]
    print(f"vnc://127.0.0.1:{port}", flush=True)
    print(f"password: {password}", flush=True)
    if control:
        print("WARNING: control is enabled; the VNC client can send keyboard and pointer input.", flush=True)
    clients: list[subprocess.Popen[bytes]] = []
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        while True:
            alive = _run_docker(config, host, "inspect", selected, check=False)
            if alive.returncode:
                break
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            process = subprocess.Popen(tunnel_command(config, host, selected), stdin=connection, stdout=connection, stderr=subprocess.DEVNULL)
            connection.close()
            clients.append(process)
            clients[:] = [item for item in clients if item.poll() is None]
    except KeyboardInterrupt:
        pass
    finally:
        close_tunnel(listener, clients)
        _stop_vnc(config, host, selected, password_file)
        if temporary is not None:
            temporary.cleanup()
    return 0


def _latest_recording() -> Path:
    roots = [Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta")) / "recordings", Path("/tmp/computer-bench"), Path("/tmp/computer-demo")]
    candidates = [path.parent for root in roots if root.exists() for path in root.rglob("metadata.json")]
    if not candidates:
        raise SystemExit("no recording found; pass a recording directory")
    return max(candidates, key=lambda path: (path / "metadata.json").stat().st_mtime)


def web(directory: Path | None) -> int:
    selected = (directory or _latest_recording()).resolve()
    if not selected.is_dir():
        raise SystemExit(f"recording directory does not exist: {selected}")
    server, token = make_web_server(selected)
    port = server.server_address[1]
    print(f"http://127.0.0.1:{port}/?token={quote(token)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live_parser = commands.add_parser("live", help="open a local VNC bridge")
    live_parser.add_argument("container", nargs="?")
    live_parser.add_argument("--control", action="store_true")
    web_parser = commands.add_parser("web", help="serve a local recording viewer")
    web_parser.add_argument("recording_dir", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "live":
        return live(args.container, control=args.control)
    return web(args.recording_dir)


if __name__ == "__main__":
    raise SystemExit(main())
