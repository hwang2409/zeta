"""Real-transport tests for the browser WebSocket serve adapter."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import stat
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from zeta.cli.main import build_parser
from zeta.server import ZetaServer
from zeta.server.protocol import MAX_FRAME_BYTES
from zeta.server.transport import _parse_subprotocol_auth

TIMEOUT = 3


async def _read_http_headers(
    reader: asyncio.StreamReader,
) -> tuple[int, dict[str, str]]:
    raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), TIMEOUT)
    lines = raw.decode("ascii").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {
        name.lower(): value.strip()
        for line in lines[1:]
        if line
        for name, value in [line.split(":", 1)]
    }
    return status, headers


async def _connect(
    server: ZetaServer,
    *,
    token: str | None,
    origin: str | None = None,
    protocol_token: str | None = None,
    protocol_fields: list[str] | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, int, dict[str, str]]:
    assert server._websocket is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server._websocket.port)
    query = "" if token is None else "?" + urlencode({"token": token})
    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    lines = [
        f"GET /{query} HTTP/1.1",
        f"Host: 127.0.0.1:{server._websocket.port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if origin is not None:
        lines.append(f"Origin: {origin}")
    if protocol_fields is not None:
        lines.extend(f"Sec-WebSocket-Protocol: {field}" for field in protocol_fields)
    elif protocol_token is not None:
        lines.append(f"Sec-WebSocket-Protocol: zeta.v1, zeta.bearer.{protocol_token}")
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    await writer.drain()
    status, headers = await _read_http_headers(reader)
    if status == 101:
        expected = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode("ascii")
        assert headers["sec-websocket-accept"] == expected
    return reader, writer, status, headers


def test_parse_subprotocol_auth() -> None:
    assert _parse_subprotocol_auth([]) == ("none", None)
    assert _parse_subprotocol_auth(["zeta.v1"]) == ("none", None)
    assert _parse_subprotocol_auth(["zeta.v1, zeta.bearer.secret"]) == (
        "bearer",
        "secret",
    )
    assert _parse_subprotocol_auth(["zeta.bearer.secret"]) == ("invalid", None)
    assert _parse_subprotocol_auth(["zeta.v1, zeta.bearer."]) == ("invalid", None)
    assert _parse_subprotocol_auth(
        ["zeta.v1, zeta.bearer.one", "zeta.bearer.two"]
    ) == ("invalid", None)
    assert _parse_subprotocol_auth(["zeta.v1, zeta.bearer.one, zeta.bearer.two"]) == (
        "invalid",
        None,
    )


def _masked_frame(
    payload: bytes,
    *,
    opcode: int = 1,
    final: bool = True,
) -> bytes:
    first = (0x80 if final else 0) | opcode
    size = len(payload)
    if size < 126:
        header = bytes((first, 0x80 | size))
    elif size <= 0xFFFF:
        header = bytes((first, 0x80 | 126)) + size.to_bytes(2, "big")
    else:
        header = bytes((first, 0x80 | 127)) + size.to_bytes(8, "big")
    mask = b"test"
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return header + mask + masked


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    first, second = await asyncio.wait_for(reader.readexactly(2), TIMEOUT)
    assert second & 0x80 == 0
    size = second & 0x7F
    if size == 126:
        size = int.from_bytes(await reader.readexactly(2), "big")
    elif size == 127:
        size = int.from_bytes(await reader.readexactly(8), "big")
    payload = await reader.readexactly(size)
    return first & 0x0F, payload


async def _send_json(writer: asyncio.StreamWriter, value: object) -> None:
    writer.write(_masked_frame(json.dumps(value, separators=(",", ":")).encode()))
    await writer.drain()


async def _read_json(reader: asyncio.StreamReader) -> dict[str, object]:
    opcode, payload = await _read_frame(reader)
    assert opcode == 1
    return json.loads(payload)


async def _close(
    server: ZetaServer, writer: asyncio.StreamWriter | None = None
) -> None:
    if writer is not None:
        writer.close()
    await asyncio.wait_for(server.close(), TIMEOUT)


@pytest.mark.asyncio
async def test_websocket_round_trip_uses_real_serve_runtime(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    reader, writer, status, headers = await _connect(
        server, token=None, protocol_token=server._websocket.token
    )
    try:
        assert status == 101
        assert headers["sec-websocket-protocol"] == "zeta.v1"
        await _send_json(
            writer,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {"protocol_version": "1.0"},
            },
        )
        assert (await _read_json(reader))["result"]["server"] == "zeta"
        await _send_json(
            writer,
            {"jsonrpc": "2.0", "id": 2, "method": "list_sessions", "params": {}},
        )
        assert await _read_json(reader) == {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"sessions": []},
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_websocket_upgrade_requires_token_and_allowed_origin(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        home=tmp_path,
        port=0,
        ws_port=0,
        ws_origins=("http://127.0.0.1:5173",),
        provider="codex",
    )
    await server.start()
    try:
        for token, origin, expected in (
            (None, None, 401),
            ("wrong", None, 401),
            (server._websocket.token, "http://localhost:5173", 403),
            (server._websocket.token, "http://127.0.0.1:5173", 101),
            (server._websocket.token, None, 101),
        ):
            _reader, writer, status, _headers = await _connect(
                server, token=token, origin=origin
            )
            assert status == expected
            writer.close()
            await writer.wait_closed()
            if status == 101:
                await asyncio.sleep(0)
    finally:
        await _close(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_token", "protocol_token", "expected"),
    [
        (None, "wrong", 401),
        (None, None, 401),
        ("__server_token__", "wrong", 401),
        ("__server_token__", None, 101),
    ],
)
async def test_websocket_subprotocol_auth_rejects_bad_tokens(
    tmp_path: Path,
    query_token: str | None,
    protocol_token: str | None,
    expected: int,
) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    assert server._websocket is not None
    if query_token == "__server_token__":
        query_token = server._websocket.token
    try:
        _reader, writer, status, _headers = await _connect(
            server, token=query_token, protocol_token=protocol_token
        )
        assert status == expected
        writer.close()
        await writer.wait_closed()
    finally:
        await _close(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "query", "expected"),
    [
        (["zeta.bearer.{token}"], "{token}", 401),
        (["zeta.v1, zeta.bearer.{token}", "zeta.bearer.wrong"], "{token}", 401),
        (["zeta.v1, zeta.bearer.{token}, zeta.bearer.wrong"], "{token}", 401),
        (["zeta.v1"], "{token}", 101),
        (["zeta.v1"], None, 401),
    ],
)
async def test_websocket_subprotocol_auth_shape(
    tmp_path: Path,
    fields: list[str],
    query: str | None,
    expected: int,
) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    assert server._websocket is not None
    token = server._websocket.token
    assert token is not None
    try:
        fields = [field.format(token=token) for field in fields]
        query = None if query is None else query.format(token=token)
        _reader, writer, status, _headers = await _connect(
            server, token=query, protocol_fields=fields
        )
        assert status == expected
        writer.close()
        await writer.wait_closed()
    finally:
        await _close(server)


@pytest.mark.asyncio
async def test_websocket_valid_bearer_echoes_only_v1(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    assert server._websocket is not None
    try:
        _reader, writer, status, headers = await _connect(
            server,
            token=None,
            protocol_fields=[f"zeta.bearer.{server._websocket.token}, zeta.v1"],
        )
        assert status == 101
        assert headers["sec-websocket-protocol"] == "zeta.v1"
        writer.close()
        await writer.wait_closed()
    finally:
        await _close(server)


async def test_websocket_and_unix_socket_share_single_client_limit(
    tmp_path: Path,
) -> None:
    socket_path = Path("/tmp") / f"zeta-ws-{secrets.token_hex(8)}.sock"
    server = ZetaServer(
        home=tmp_path, socket_path=socket_path, ws_port=0, provider="codex"
    )
    await server.start()
    ws_reader, ws_writer, status, _headers = await _connect(
        server, token=server._websocket.token
    )
    try:
        assert status == 101
        await _send_json(
            ws_writer,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {"protocol_version": "1.0"},
            },
        )
        await _read_json(ws_reader)
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        error = json.loads(await asyncio.wait_for(reader.readline(), TIMEOUT))
        assert error["error"]["code"] == -32001
        writer.close()
        await writer.wait_closed()
    finally:
        await _close(server, ws_writer)


@pytest.mark.asyncio
async def test_websocket_reassembles_fragmented_text(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    reader, writer, status, _headers = await _connect(
        server, token=server._websocket.token
    )
    try:
        assert status == 101
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {"protocol_version": "1.0"},
            },
            separators=(",", ":"),
        ).encode()
        split = len(payload) // 2
        writer.write(_masked_frame(payload[:split], final=False))
        writer.write(_masked_frame(payload[split:], opcode=0))
        await writer.drain()
        assert (await _read_json(reader))["result"]["server"] == "zeta"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "close_code"),
    [
        (
            b"\x81\xff" + (MAX_FRAME_BYTES + 1).to_bytes(8, "big") + b"test",
            1009,
        ),
        (_masked_frame(b"binary", opcode=2), 1003),
    ],
    ids=("oversize", "binary"),
)
async def test_websocket_rejects_invalid_message_types(
    tmp_path: Path, frame: bytes, close_code: int
) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    reader, writer, status, _headers = await _connect(
        server, token=server._websocket.token
    )
    try:
        assert status == 101
        writer.write(frame)
        await writer.drain()
        opcode, payload = await _read_frame(reader)
        assert opcode == 8
        assert int.from_bytes(payload[:2], "big") == close_code
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_websocket_token_file_lifecycle_and_loopback_bind(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path,
        port=0,
        ws_port=0,
        ws_origins=("http://127.0.0.1:5173",),
        provider="codex",
    )
    await server.start()
    token_path = tmp_path / "run" / "serve.ws.json"
    try:
        value = json.loads(token_path.read_text())
        assert value == {
            "port": server._websocket.port,
            "token": server._websocket.token,
            "pid": os.getpid(),
            "origins": ["http://127.0.0.1:5173"],
            "ws_auth": ["subprotocol", "query"],
        }
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        assert server._websocket.server is not None
        assert {sock.getsockname()[0] for sock in server._websocket.server.sockets} == {
            "127.0.0.1"
        }
    finally:
        await _close(server)
    assert not token_path.exists()


@pytest.mark.asyncio
async def test_websocket_prints_connect_url_once_without_logging_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    with caplog.at_level(logging.DEBUG):
        await server.start()
        assert server._websocket is not None
        token = server._websocket.token
        port = server._websocket.port
        await _close(server)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert token is not None
    expected = f"zeta serve: websocket ready: ws://127.0.0.1:{port}/?token={token}\n"
    assert captured.err == expected
    startup_url = captured.err.removeprefix("zeta serve: websocket ready: ").removesuffix("\n")
    parsed = urlsplit(startup_url)
    assert parsed.scheme == "ws"
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port == port
    assert parsed.path == "/"
    assert parse_qs(parsed.query) == {"token": [token]}
    assert all(token not in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_websocket_discovery_file_has_one_owner(tmp_path: Path) -> None:
    token_path = tmp_path / "run" / "serve.ws.json"
    first = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    second = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await first.start()
    first_value = token_path.read_bytes()
    assert stat.S_IMODE((tmp_path / "run" / "serve.ws.lock").stat().st_mode) == 0o600
    try:
        with pytest.raises(
            ValueError,
            match=rf"WebSocket server.*pid {os.getpid()}.*stop it",
        ):
            await second.start()
        assert second._websocket.server is None
        assert token_path.read_bytes() == first_value
    finally:
        await _close(second)
        await _close(first)
    assert not token_path.exists()

    replacement = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await replacement.start()
    try:
        assert token_path.exists()
        assert token_path.read_bytes() != first_value
    finally:
        await _close(replacement)


@pytest.mark.asyncio
async def test_websocket_replaces_stale_discovery_file(tmp_path: Path) -> None:
    token_path = tmp_path / "run" / "serve.ws.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text('{"stale":true}')
    (token_path.parent / "serve.ws.lock").write_text("999999")
    server = ZetaServer(home=tmp_path, port=0, ws_port=0, provider="codex")
    await server.start()
    try:
        value = json.loads(token_path.read_text())
        assert value["token"] == server._websocket.token
        assert value["pid"] == os.getpid()
        assert (token_path.parent / "serve.ws.lock").read_text() == str(os.getpid())
    finally:
        await _close(server)


def test_serve_websocket_cli_flags() -> None:
    args = build_parser().parse_args(
        [
            "serve",
            "--ws",
            "8765",
            "--ws-origin",
            "http://127.0.0.1:5173",
            "--ws-origin",
            "https://app.example:443",
        ]
    )
    assert args.ws_port == 8765
    assert args.ws_origins == ["http://127.0.0.1:5173", "https://app.example:443"]
