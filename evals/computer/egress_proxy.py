"""Read-only, single-host HTTPS fetch broker for the disposable browser eval."""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import os
import socket
import socketserver
import ssl
import sys
from urllib.parse import urlsplit

SOCKET = "/proxy/egress.sock"
MAX_BODY = 2_000_000
MAX_MESSAGE = 4_000_000


def _destination(url: str, host: str) -> tuple[str, str]:
    if not url or any(ord(char) < 33 or ord(char) == 127 for char in url):
        raise ValueError("invalid URL")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc.casefold() not in {host, f"{host}:443"}:
        raise ValueError("destination is not the approved HTTPS host")
    target = parts.path or "/"
    if parts.query:
        target += f"?{parts.query}"
    return host, target


def _public_address(host: str) -> str:
    addresses = {
        entry[4][0]
        for entry in socket.getaddrinfo(
            host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM
        )
    }
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("destination did not resolve exclusively to public IPs")
    return min(addresses)


def fetch(url: str, host: str) -> dict[str, object]:
    host, target = _destination(url, host)
    address = _public_address(host)
    with (
        socket.create_connection((address, 443), timeout=10) as raw,
        ssl.create_default_context().wrap_socket(raw, server_hostname=host) as tls,
    ):
        tls.settimeout(10)
        request = (
            f"GET {target} HTTP/1.1\r\nHost: {host}\r\n"
            "Accept-Encoding: identity\r\nUser-Agent: ZetaBrowserEval/1\r\n"
            "Connection: close\r\n\r\n"
        )
        tls.sendall(request.encode("ascii"))
        response = http.client.HTTPResponse(tls)
        response.begin()
        if response.status not in range(200, 400):
            raise ValueError(f"upstream returned HTTP {response.status}")
        body = response.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise ValueError("upstream response too large")
        headers = {
            key: value
            for key in ("content-type", "content-encoding", "location")
            if (value := response.getheader(key)) is not None
        }
        if "location" in headers:
            from urllib.parse import urljoin

            _destination(urljoin(url, headers["location"]), host)
        return {
            "status": response.status,
            "headers": headers,
            "body": base64.b64encode(body).decode("ascii"),
        }


class Handler(socketserver.StreamRequestHandler):
    host = ""

    def handle(self) -> None:
        try:
            line = self.rfile.readline(8193)
            if len(line) > 8192:
                raise ValueError("request too large")
            request = json.loads(line)
            if type(request) is not dict or request.get("method") != "GET":
                raise ValueError("only GET is allowed")
            url = request.get("url")
            if type(url) is not str:
                raise ValueError("URL must be a string")
            reply = fetch(url, self.host)
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            reply = {"error": str(exc)}
        self.wfile.write(json.dumps(reply, separators=(",", ":")).encode() + b"\n")


def request(url: str) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(15)
        connection.connect(SOCKET)
        connection.sendall(json.dumps({"method": "GET", "url": url}).encode() + b"\n")
        with connection.makefile("rb") as response:
            line = response.readline(MAX_MESSAGE + 1)
    if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ValueError("invalid egress reply")
    result = json.loads(line)
    if "error" in result:
        raise ValueError(str(result["error"]))
    return result


def main() -> None:
    host = sys.argv[1]
    if not host or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for char in host
    ):
        raise SystemExit("invalid approved host")
    Handler.host = host
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    with socketserver.ThreadingUnixStreamServer(SOCKET, Handler) as server:
        os.chmod(SOCKET, 0o666)
        server.serve_forever()


if __name__ == "__main__":
    main()
