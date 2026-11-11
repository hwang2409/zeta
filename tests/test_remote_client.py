from __future__ import annotations

import asyncio
import json
import re
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from tests.support.owned import owned_app
from zeta.core.checkpoints import ConversationEntry
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.remote.client import RemoteSessionClient
from zeta.remote.discovery import split_target, validate_host
from zeta.remote.transport import RemoteTransport, TransportItem, remote_bridge_command
from zeta.remote.wire import BoundedLineDecoder, RemoteWireError, decode_line
from zeta.runtime.loop import AgentLoop
from zeta.server import event_contract
from zeta.server.handlers.history import replay_history
from zeta.server.protocol import EventDescriptor, FrameCodec
from zeta.skills import SkillCatalog
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.slash_handlers.remote import RemoteViewMixin, RemoteViewState


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


def test_remote_line_decoder_bounds_frames_split_across_chunks() -> None:
    decoder = BoundedLineDecoder()
    assert decoder.feed(b'{"a":') == []
    assert decoder.feed(b'1}\n\n{"b":2}\n') == [b'{"a":1}', b'{"b":2}']
    with pytest.raises(RemoteWireError):
        BoundedLineDecoder().feed(b"x" * (256 * 1024 + 1))
    with pytest.raises(RemoteWireError):
        BoundedLineDecoder().feed(b"x" * (256 * 1024 + 1) + b"\n")
    # a deeply nested frame fits the size bound but overflows the json parser's recursion.
    with pytest.raises(RemoteWireError):
        decode_line(b"[" * 100_000 + b"]" * 100_000)


def _scripted_transport(attach_lines: list[bytes], *, exit_after: bool = True):
    rows = b"".join(
        b'{"kind":"session","session_id":"%s","name":"%s","state":"idle"}\n' % (sid, sid)
        for sid in (b"abc123", b"def456")
    )

    class Transport:
        def __init__(self, host: str, *, callback, **_: object) -> None:
            self.callback = callback

        def start(self, command: str) -> None:
            if "attach" not in command:
                self.callback(TransportItem("stdout", rows))
                self.callback(TransportItem("exit", b"0"))
                return
            for line in attach_lines:
                self.callback(TransportItem("stdout", line))
            if exit_after:
                self.callback(TransportItem("exit", b"0"))

        def send(self, data: bytes) -> None:
            return None

        async def close_async(self) -> None:
            return None

    return Transport


def _entry(data: dict[str, object]) -> ConversationEntry:
    return ConversationEntry(1, "row-1", None, "main", "x", data)


def _server_fields(descriptor: EventDescriptor) -> dict[str, object]:
    """Payload of one event as the server builds it, with event_contract's own builders."""
    call = ToolCall("call-1", "bash", {"command": "ls"})
    builders = {
        EventDescriptor.ASSISTANT_DELTA: lambda: {"delta": "hi", "kind": "assistant"},
        EventDescriptor.ASSISTANT_MESSAGE: lambda: event_contract.assistant_message(StreamEvent(
            StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, [TextContent("done")]))),
        EventDescriptor.TOOL_START: lambda: event_contract.tool_start(
            StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call), fidelity=True),
        EventDescriptor.TOOL_OUTPUT: lambda: event_contract.tool_output(
            StreamEvent(StreamEventType.TOOL_EXECUTION_UPDATE, tool_call=call, delta="chunk"), fidelity=True),
        EventDescriptor.TOOL_END: lambda: event_contract.tool_end(StreamEvent(
            StreamEventType.TOOL_EXECUTION_END, tool_call=call, tool_result=ToolResult("call-1", "ok")), fidelity=True),
        EventDescriptor.CONFIG_CHANGED: lambda: event_contract.config_changed_wire_fields(
            StreamEvent(StreamEventType.CONFIG_CHANGED, data={"message": "model changed"})),
        EventDescriptor.DELIVERY_STATUS: lambda: event_contract.delivery_status_row(
            _entry({"status": "accepted", "delivery_id": "d1"})),
        EventDescriptor.SERVED_TURN_END: lambda: event_contract.served_turn_end_row(
            _entry({"turn_id": "t1", "reason": "completed"})),
        EventDescriptor.APPROVAL_REQUEST: lambda: event_contract.approval_request(
            {"approval_id": "a1", "tool_call": call.to_dict(), "summary": "run ls"}),
        EventDescriptor.APPROVAL_END: lambda: event_contract.approval_end({"approval_id": "a1", "decision": "approved"}),
        EventDescriptor.ERROR: lambda: {"error": {"code": "boom", "message": "remote failed"}},
    }
    for name, wire in event_contract.NOTIFICATION_WIRE_EVENTS.items():
        builders[wire] = lambda name=name: {"data": event_contract.notification_payload({"kind": name, "text": "x"})}
    # events built inline in control.py: a payload with keys this client has never seen.
    fields = builders.get(descriptor, lambda: {"data": {"future": True}, "added_in_a_newer_zeta": [1, 2]})()
    # like delivery.py, the server drops a row's own "event" key when it publishes the row.
    return {**{key: value for key, value in fields.items() if key != "event"}, "future_key": {"nested": "value"}}


def _history_frame() -> bytes:
    """A history reply with the server's projected system rows next to a plain user row."""
    rows = [
        {"id": "u1", "role": "user", "content": [{"type": "text", "text": "hi"}], "added_in_a_newer_zeta": 1},
        event_contract.delivery_status_row(_entry({"status": "accepted", "delivery_id": "d1"})),
        event_contract.served_turn_end_row(_entry({"turn_id": "t1", "reason": "completed"})),
    ]
    return json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"messages": rows}}).encode() + b"\n"


@pytest.mark.asyncio
async def test_every_server_event_frame_decodes_or_is_ignored_never_fatal() -> None:
    codec = FrameCodec()
    frames = [
        bytes(codec.notification(descriptor, "abc123", **_server_fields(descriptor)))
        for descriptor in EventDescriptor
    ] + [_history_frame(), b'{"jsonrpc":"2.0","method":"event","params":{"event":"invented_after_this_client"}}\n']
    client = RemoteSessionClient(Path("/tmp/remote-test"), transport_factory=_scripted_transport(frames))
    await client.attach("home", "abc123")
    events: list[StreamEvent] = []
    try:
        while not isinstance(event := await asyncio.wait_for(client.next_event(), 10), str):
            events.append(event)
    finally:
        await client.detach()
    assert event == "disconnected", f"a real server frame ended the remote view: {event}"
    deltas = [item.delta for item in events if item.type is StreamEventType.TOOL_EXECUTION_UPDATE]
    assert deltas == ["chunk"], "tool_output.output must reach the view as the tool delta"
    assert {StreamEventType.TOOL_EXECUTION_START, StreamEventType.TOOL_EXECUTION_END, StreamEventType.MESSAGE_END} <= {
        item.type for item in events
    }


@pytest.mark.asyncio
async def test_attach_ignores_hello_result_before_history() -> None:
    frames = [
        b'{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"1.1"}}\n',
        b'{"jsonrpc":"2.0","method":"event","params":{"event":"usage","usage":{"input_tokens":1}}}\n',
        b'{"jsonrpc":"2.0","id":2,"result":{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}}\n',
    ]
    client = RemoteSessionClient(Path("/tmp/remote-test"), transport_factory=_scripted_transport(frames, exit_after=False))
    await client.attach("home", "abc123")
    try:
        event = await asyncio.wait_for(client.next_event(), 10)
        assert event.type.value == "message_end"
    finally:
        await client.detach()


@pytest.mark.asyncio
async def test_configured_remote_keeps_ssh_user_and_port(tmp_path: Path) -> None:
    (tmp_path / "settings.toml").write_text('[remotes]\nlaptop = "ssh://henry@laptop:2222"\n')
    seen: list[tuple[str, object]] = []
    base = _scripted_transport([])

    def factory(host: str, **kwargs: object):
        seen.append((host, kwargs.get("port")))
        return base(host, **kwargs)

    rows, _ = await RemoteSessionClient(tmp_path, transport_factory=factory).discover()
    assert [row.host for row in rows] == ["laptop", "laptop"]
    assert seen == [("henry@laptop", 2222)], f"ssh target lost the URL's user or port: {seen}"


def test_attach_history_replays_prompt_text_and_named_tool_card(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall("call-1", "bash", {"command": "echo streamed-line-1"})
    store.append_message(with_message_origin(
        Message(MessageRole.USER, [TextContent("hello remote")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("running a command now."), ToolUseContent(call)]))
    store.append_message(Message(MessageRole.TOOL_RESULT, [], tool_result=ToolResult("call-1", "streamed-line-1")))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("after the tool.")]))
    # the rows the host's serve sends for session_history_and_subscribe.
    result, _, _ = replay_history(SimpleNamespace(opened=SimpleNamespace(store=store)), include_delivery_id=False,
                                  fidelity=True, content_read=False, served_turn_end=False)
    frame = json.dumps({"jsonrpc": "2.0", "id": 2, "result": result}).encode()
    state = RemoteViewState.create(SimpleNamespace(console=Console()), "home", "abc123", "")
    for event in decode_line(frame):
        state.render(event)
    rendered = re.sub(r"\x1b\[[0-9;]*m", "", state.transcript.render(120))
    for expected in ("hello remote", "running a command now.", "bash echo streamed-line-1", "streamed-line-1", "after the tool."):
        assert expected in rendered, f"attach history lost {expected!r}:\n{rendered}"


def _remote_app(tmp_path: Path, transport) -> RemoteViewMixin:
    class App(RemoteViewMixin):
        def __init__(self) -> None:
            self.console = Console()
            self._zeta_home = tmp_path
            self._active_session = None
            self._agent_navigation = SimpleNamespace(show_external_view=lambda *_: None, clear_external_view=lambda: None)
            self._init_remote_view()
            self._remote_session_client = RemoteSessionClient(tmp_path, transport_factory=transport)

        def _invalidate_prompt(self) -> None:
            return None

        def _print_system(self, output: str) -> None:
            self.system_lines.append(output)

    app = App()
    app.system_lines = []
    return app


def _fake_ssh(tmp_path: Path) -> str:
    """An ssh that lists one session on host "good" and fails like a missing key on any other host."""
    fake = tmp_path / "ssh"
    fake.write_text(
        '#!/bin/sh\n'
        'while [ "$1" != "--" ]; do shift; done\n'
        'if [ "$2" = good ]; then echo \'{"kind":"session","session_id":"abc123","name":"a","state":"idle"}\'; exit 0; fi\n'
        'echo "debug1: some earlier line" >&2\n'
        'echo "Permission denied (publickey)." >&2\n'
        'exit 255\n'
    )
    fake.chmod(0o755)
    return str(fake)


@pytest.mark.asyncio
async def test_ssh_failure_is_reported_not_shown_as_no_sessions(tmp_path: Path) -> None:
    client = RemoteSessionClient(tmp_path, ssh=_fake_ssh(tmp_path))
    with pytest.raises(RemoteWireError, match=r"^ssh to laptop failed: Permission denied \(publickey\)\.$"):
        await client.discover(("laptop",))
    rows, errors = await client.discover(("good", "laptop"))
    assert [row.session_id for row in rows] == ["abc123"]
    assert errors == ("ssh to laptop failed: Permission denied (publickey).",)


@pytest.mark.asyncio
async def test_failed_remote_command_output_is_repainted_without_a_keypress(tmp_path: Path) -> None:
    # the slash command finishes after its enter key was drawn; the screen must repaint its output.
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="codex", model="offline", zeta_home=tmp_path, console=Console(file=StringIO()),
    )
    async with owned_app(app):
        app._remote_session_client = RemoteSessionClient(tmp_path, ssh=_fake_ssh(tmp_path))
        with create_pipe_input() as pipe:
            app._active_session = FullScreenPromptSession(input=pipe, output=DummyOutput(), multiline=True)
            painted: list[str] = []
            app._invalidate_prompt = lambda: painted.append(app._transcript.render(120))  # type: ignore[method-assign]
            app._submit_input("/remote bad")
            pipeline = app._submissions
            while pipeline._slash_tasks or (pipeline._consumer is not None and not pipeline._consumer.done()):
                pending = [task for task in (pipeline._consumer, *pipeline._slash_tasks.values()) if task is not None]
                await asyncio.wait_for(asyncio.gather(*map(asyncio.shield, pending), return_exceptions=True), 10)
            assert "ssh to bad failed" in app._transcript.render(120), "the command did not report the ssh failure"
            assert painted and "ssh to bad failed" in painted[-1], "the ssh failure was printed but never repainted"


@pytest.mark.asyncio
async def test_attach_ssh_failure_ends_the_view_with_the_ssh_error(tmp_path: Path) -> None:
    app = _remote_app(tmp_path, RemoteTransport)
    app._remote_session_client.ssh = _fake_ssh(tmp_path)
    try:
        await app._remote_client().attach("laptop", "abc123")
        app._remote_view = state = RemoteViewState.create(app, "laptop", "abc123", "")
        await asyncio.wait_for(app._consume_remote_events(state, app._remote_client()), 10)
        assert "ssh to laptop failed: Permission denied (publickey)." in state.transcript.render(120), "attach ended without the ssh error"
        assert state.disconnected
    finally:
        await app.slash_detach("")


@pytest.mark.asyncio
async def test_fatal_link_error_ends_the_remote_view_with_its_reason(tmp_path: Path) -> None:
    app = _remote_app(tmp_path, _scripted_transport([b"not json\n"], exit_after=False))
    try:
        assert await app.slash_remote("home:abc123") == ""
        state, task = app._remote_view, app._remote_event_task
        done, _ = await asyncio.wait({task}, timeout=10)
        assert done, "the remote view kept waiting after a fatal link error"
        assert state.disconnected
        assert "invalid remote frame" in state.transcript.render(120)
    finally:
        await app.slash_detach("")


@pytest.mark.asyncio
async def test_second_remote_attach_stops_the_first_view_consumer(tmp_path: Path) -> None:
    app = _remote_app(tmp_path, _scripted_transport([], exit_after=False))
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
