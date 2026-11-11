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


@pytest.mark.asyncio
async def test_second_remote_attach_stops_the_first_view_consumer(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rich.console import Console

    from zeta.tui.slash_handlers.remote import RemoteViewMixin

    rows = b"".join(
        b'{"kind":"session","session_id":"%s","name":"%s","state":"idle"}\n' % (sid, sid)
        for sid in (b"abc123", b"def456")
    )

    class IdleAttachTransport:
        def __init__(self, host: str, *, callback, **_: object) -> None:
            self.callback = callback

        def start(self, command: str) -> None:
            if "attach" not in command:
                self.callback(TransportItem("stdout", rows))
                self.callback(TransportItem("exit", b"0"))

        def send(self, data: bytes) -> None:
            return None

        async def close_async(self) -> None:
            return None

    class App(RemoteViewMixin):
        def __init__(self) -> None:
            self.console = Console()
            self._zeta_home = tmp_path
            self._active_session = None
            self._agent_navigation = SimpleNamespace(show_external_view=lambda *_: None, clear_external_view=lambda: None)
            self._init_remote_view()
            self._remote_session_client = RemoteSessionClient(tmp_path, transport_factory=IdleAttachTransport)

        def _invalidate_prompt(self) -> None:
            return None

    app = App()
    first = None
    try:
        assert await app.slash_remote("home:abc123") == ""
        first = app._remote_event_task
        assert await app.slash_remote("home:def456") == ""
        assert first is not None and first.done(), "the first view's event consumer kept waiting after a second attach"
    finally:
        if first is not None and not first.done():
            first.cancel()
            await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 10)
        await app.slash_detach("")


@pytest.mark.parametrize(("remote_home", "expected"), [("~/.zeta", "{home}/.zeta"), ("~", "{home}"), ("/srv/z home", "/srv/z home")])
def test_bridge_command_expands_remote_home_on_the_remote_shell(tmp_path: Path, remote_home: str, expected: str) -> None:
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "zeta"
    fake.write_text('#!/bin/sh\nprintf "%s" "$ZETA_HOME"\n')
    fake.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run(["sh", "-c", remote_bridge_command("list", remote_home=remote_home)], env=env, capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout == expected.format(home=tmp_path)
