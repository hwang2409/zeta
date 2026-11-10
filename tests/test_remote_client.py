from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.remote.client import RemoteSessionClient
from zeta.remote.discovery import choose_session, split_target, validate_host
from zeta.remote.transport import TransportItem, remote_bridge_command
from zeta.remote.wire import RemoteWireError, decode_lines


def test_remote_target_and_shell_validation() -> None:
    assert split_target("home:abc") == ("home", "abc")
    with pytest.raises(ValueError):
        validate_host("bad host")
    with pytest.raises(ValueError):
        split_target("home:a/b")
    command = remote_bridge_command("attach", "abc")
    assert 'command -v zeta' in command
    assert '"$HOME/.local/bin/zeta"' in command
    with pytest.raises(ValueError):
        remote_bridge_command("attach", "a;b")


def test_remote_wire_rejects_unbounded_or_unterminated_frames() -> None:
    with pytest.raises(RemoteWireError):
        tuple(decode_lines([b'{"method":"event"}']))
    with pytest.raises(RemoteWireError):
        tuple(decode_lines([b"x" * (256 * 1024 + 1)]))


def test_session_prefix_must_be_unique() -> None:
    from zeta.remote.discovery import RemoteSession

    rows = (RemoteSession("home", "abc1", "first", "idle"), RemoteSession("home", "abc2", "second", "idle"))
    with pytest.raises(ValueError):
        choose_session(rows, "abc")


@pytest.mark.asyncio
async def test_attach_ignores_hello_result_before_history() -> None:
    class FakeTransport:
        def __init__(self, host: str, *, callback, **_: object) -> None:
            self.callback = callback

        def start(self, command: str) -> None:
            self.callback(TransportItem("stdout", b'{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"1.1"}}\n'))
            self.callback(TransportItem("stdout", b'{"jsonrpc":"2.0","method":"event","params":{"event":"usage","usage":{"input_tokens":1}}}\n'))
            self.callback(TransportItem("stdout", b'{"jsonrpc":"2.0","id":2,"result":{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}}\n'))

        def send(self, data: bytes) -> None:
            return None

        async def close_async(self) -> None:
            return None

    client = RemoteSessionClient(Path("/tmp/remote-test"), transport_factory=FakeTransport)
    await client.attach("home", "abc123")
    try:
        event = await asyncio.wait_for(client.next_event(), 10)
        assert event.type.value == "message_end"
    finally:
        await client.detach()
