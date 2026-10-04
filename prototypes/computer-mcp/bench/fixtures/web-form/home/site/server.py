import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        data = Path("/home/zeta/site/form.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        q = parse_qs(self.rfile.read(n).decode())
        out = {k: v[0] for k, v in q.items()}
        Path("/home/zeta/form-submission.json").write_text(
            json.dumps(out, sort_keys=True)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Request submitted</h1>")

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
