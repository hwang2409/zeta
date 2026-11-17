"""Local, deterministic click-target fixture server."""
from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WIDTH, HEIGHT = 1280, 800


def trial_state(state_dir: Path, trial: int) -> dict:
    seed = hashlib.sha256(f"computer-use-{trial}".encode()).digest()
    x = 240 + seed[0] % 760
    y = 180 + seed[1] % 420
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"trial-{trial}.json"
    if path.exists():
        return json.loads(path.read_text())
    state = {"trial": trial, "nonce": seed.hex(), "target": {"x": x, "y": y, "width": 44, "height": 44}, "status": "READY", "clicks": []}
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def html(state: dict) -> bytes:
    target = state["target"]
    payload = json.dumps(state["nonce"])
    text = f"""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
<title>zeta computer fixture</title><style>body{{margin:0;background:#fff}}#target{{position:absolute;left:{target['x']}px;top:{target['y']}px;width:44px;height:44px;background:#1677ff;border:0;border-radius:4px}}#status{{position:fixed;left:12px;top:12px;font:16px monospace}}</style>
<div id=status>READY</div><button id=target aria-label='blue target'></button><script>
const nonce={payload}; const target=document.querySelector('#target');
target.onclick=e=>{{fetch('/event',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{nonce, x:e.clientX, y:e.clientY, viewport:[innerWidth,innerHeight], device_pixel_ratio:devicePixelRatio}})}}); document.querySelector('#status').textContent='PASS';}};
</script>"""
    return text.encode()


class Handler(BaseHTTPRequestHandler):
    server: FixtureServer
    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        state = trial_state(self.server.state_dir, self.server.trial)
        data = html(state)
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/event":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        state_path = self.server.state_dir / f"trial-{self.server.trial}.json"
        state = json.loads(state_path.read_text())
        state["clicks"].append(data)
        state["status"] = "PASS"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


class FixtureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state_dir: Path, trial: int) -> None:
        super().__init__(address, Handler)
        self.state_dir = state_dir
        self.trial = trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    server = FixtureServer(("127.0.0.1", args.port), args.state_dir, args.trial)
    trial_state(args.state_dir, args.trial)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
