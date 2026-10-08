from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_agent import (
    RunBackend,
    _agent_call,
    _collect,
    _run_handle_from_receipt,
    _wait_for_notification,
)
from tests.test_automations import (
    DUE,
    RecordingDelivery,
    _arm,
    _empty_mount,
    _job,
)
from tests.test_server import _close, _connect, _request
from tests.test_steering import SteerToolBackend
from zeta.automations.runner import _receipt, run_claimed
from zeta.automations.store import SQLiteStore
from zeta.automations.tick import tick
from zeta.core.session import SessionManager
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.mcp import MCPPrompt, MCPPromptArgument
from zeta.memory.reconciler import Transcript, prepare_request
from zeta.model_input import ModelInputEnvelope
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
)
from zeta.runtime.driver import drive_turn
from zeta.runtime.headless import run_headless
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.empty_turn import build_nudge_message
from zeta.server import ZetaServer
from zeta.skills import SkillCatalog, discover_session_skills
from zeta.tools.agent_send import send_to_run
from zeta.tui._attachments import build_user_message
from zeta.tui.app import TUIApp
from zeta.tui.slash_handlers import SlashHandlerMixin

SESSION_ID = "a" * 32


def _rendered(row: dict[str, object]) -> dict[str, object]:
    request = prepare_request(
        Transcript(SESSION_ID, (row,)), {}, as_of=date(2026, 10, 7)
    )
    rows = json.loads(request.prompt.split("Completed transcript rows:\n", 1)[1])
    return rows[0]


def _label(row: dict[str, object]) -> str:
    return str(_rendered(row)["authorship"])


def _message_row(message: Message) -> dict[str, object]:
    return {"seq": 1, "type": "message", "data": {"message": message.to_dict()}}


def test_typed_tui_message_is_labeled_user(tmp_path: Path) -> None:
    message = build_user_message("typed in TUI", tmp_path)

    assert message.metadata[MESSAGE_ORIGIN_METADATA] == MessageOrigin.USER
    assert _label(_message_row(message)) == "user"


@pytest.mark.asyncio
async def test_serve_user_message_is_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    async for _event in loop.run_turn("sent through serve", origin=MessageOrigin.USER):
        pass

    assert _label(store.entries[0].to_dict()) == "user"
    await loop.close()


@pytest.mark.asyncio
async def test_headless_prompt_is_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    code = await drive_turn(
        loop,
        "sent with -p",
        origin=MessageOrigin.USER,
        format="text",
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    assert _label(store.entries[0].to_dict()) == "user"
    await loop.close()


@pytest.mark.asyncio
async def test_automation_prompt_is_not_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    code = await drive_turn(
        loop,
        "saved automation prompt",
        format="text",
        stdout=StringIO(),
        stderr=StringIO(),
        origin=MessageOrigin.AUTOMATION_PROMPT,
    )

    assert code == 0
    assert _label(store.entries[0].to_dict()) == "automation_prompt"
    await loop.close()


def test_skill_expansion_is_not_labeled_user(tmp_path: Path) -> None:
    skill = tmp_path / ".zeta" / "skills" / "review.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.",
        encoding="utf-8",
    )
    registry = create_slash_registry(
        project_dir=tmp_path,
        skill_catalog=discover_session_skills(project_dir=tmp_path),
    )

    expansion = registry.dispatch(object(), "$review this")

    assert isinstance(expansion, ModelInputEnvelope)
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={
            MESSAGE_ORIGIN_METADATA: expansion.origin.value,
            "zeta.user_display_text": expansion.display_text,
        },
    )
    rendered = _rendered(_message_row(message))
    assert rendered["authorship"] == "skill_expansion"
    assert rendered["user_authored_input"] == {
        "authorship": "user",
        "text": "$review this",
    }


def test_slash_skill_preserves_user_authored_input(tmp_path: Path) -> None:
    skill = tmp_path / ".zeta" / "skills" / "review.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.",
        encoding="utf-8",
    )
    registry = create_slash_registry(
        project_dir=tmp_path,
        skill_catalog=discover_session_skills(project_dir=tmp_path),
    )

    expansion = registry.dispatch(object(), "/review this branch")

    assert isinstance(expansion, ModelInputEnvelope)
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={
            MESSAGE_ORIGIN_METADATA: expansion.origin.value,
            "zeta.user_display_text": expansion.display_text,
        },
    )
    rendered = _rendered(_message_row(message))
    assert rendered["authorship"] == "skill_expansion"
    assert rendered["user_authored_input"] == {
        "authorship": "user",
        "text": "/review this branch",
    }


def _render_envelope(envelope: ModelInputEnvelope) -> dict[str, object]:
    return _rendered(
        _message_row(
            Message(
                MessageRole.USER,
                [TextContent(envelope.text)],
                metadata={
                    MESSAGE_ORIGIN_METADATA: envelope.origin.value,
                    "zeta.user_display_text": envelope.display_text,
                },
            )
        )
    )


def test_plan_preserves_exact_typed_input_as_nested_user_evidence() -> None:
    class Session(SlashHandlerMixin):
        active = False
        pending_approvals: tuple[object, ...] = ()
        plan_mode = False

        def __init__(self) -> None:
            self.loop = self

        def set_plan_mode(self, enabled: bool) -> None:
            self.plan_mode = enabled

        def _invalidate_prompt(self) -> None:
            pass

    typed = "/plan   inspect this exact branch"
    envelope = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
        Session(), typed
    )

    assert isinstance(envelope, ModelInputEnvelope)
    assert envelope.display_text == typed
    assert envelope.origin is MessageOrigin.SLASH_EXPANSION
    assert _render_envelope(envelope)["user_authored_input"] == {
        "authorship": "user",
        "text": typed,
    }


@pytest.mark.asyncio
async def test_mcp_prompt_preserves_exact_typed_input_as_nested_user_evidence() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(self, name: str, arguments: dict[str, str]) -> str:
            return f"resolved {name} {arguments['topic']}"

    typed = "/server:review   exact topic"
    envelope = await registry.dispatch_async(Session(), typed)

    assert isinstance(envelope, ModelInputEnvelope)
    assert envelope.display_text == typed
    assert envelope.origin is MessageOrigin.SLASH_EXPANSION
    assert _render_envelope(envelope)["user_authored_input"] == {
        "authorship": "user",
        "text": typed,
    }


def test_slash_expansion_is_not_labeled_user() -> None:
    expansion = ModelInputEnvelope(
        "expanded slash prompt",
        "/generated",
        MessageOrigin.SLASH_EXPANSION,
    )
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={MESSAGE_ORIGIN_METADATA: expansion.origin.value},
    )

    assert _label(_message_row(message)) == "slash_expansion"


_MODEL_INPUT_AUDIT = (
    ("plain text", MessageOrigin.USER, "plain text"),
    ("$skill", MessageOrigin.SKILL_EXPANSION, "$review this"),
    ("/skill", MessageOrigin.SKILL_EXPANSION, "/review this"),
    ("custom command", MessageOrigin.SLASH_EXPANSION, "/custom this"),
    ("custom inline shell", MessageOrigin.SLASH_EXPANSION, "/custom-shell"),
    ("/plan", MessageOrigin.SLASH_EXPANSION, "/plan inspect"),
    ("/init", MessageOrigin.SLASH_EXPANSION, "/init"),
    ("/implement", MessageOrigin.SLASH_EXPANSION, "/implement"),
    ("MCP prompt", MessageOrigin.SLASH_EXPANSION, "/server:prompt value"),
    (
        "MCP resource",
        MessageOrigin.SLASH_EXPANSION,
        "/mcp resources server mcp://doc/1",
    ),
    ("escaped slash", MessageOrigin.USER, "//status"),
    ("unknown slash", MessageOrigin.USER, "/not-a-command"),
    ("attachments/images", MessageOrigin.USER, "inspect @note.txt"),
    ("paste expansion", MessageOrigin.USER, "inspect [Image #1]"),
    ("steer/queued input", MessageOrigin.USER, "steer now"),
    ("serve send", MessageOrigin.USER, None),
    ("serve slash input", MessageOrigin.SLASH_EXPANSION, "/custom this"),
    ("serve send_images", MessageOrigin.USER, None),
    ("serve steer", MessageOrigin.USER, None),
    ("inbox wake", MessageOrigin.NOTIFICATION, None),
)


class _AuditSlashSession(SlashHandlerMixin):
    active = False
    pending_approvals: tuple[object, ...] = ()

    def __init__(self, *, plan_mode: bool = False) -> None:
        self.loop = self
        self.plan_mode = plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        self.plan_mode = enabled

    def _invalidate_prompt(self) -> None:
        pass

    async def slash_mcp_prompt(self, name: str, arguments: dict[str, str]) -> str:
        return f"resolved {name} {arguments['topic']}"


def _write_audit_skill(tmp_path: Path) -> None:
    skill = tmp_path / ".zeta" / "skills" / "review.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.",
        encoding="utf-8",
    )


def _envelope_row(envelope: ModelInputEnvelope) -> dict[str, object]:
    return _message_row(
        Message(
            MessageRole.USER,
            [TextContent(envelope.text)],
            metadata={
                MESSAGE_ORIGIN_METADATA: envelope.origin.value,
                "zeta.user_display_text": envelope.display_text,
            },
        )
    )


async def _audit_tui_row(path: str, tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    typed = {
        "plain text": "plain text",
        "$skill": "$review this",
        "/skill": "/review this",
        "custom command": "/custom this",
        "custom inline shell": "/custom-shell",
        "/plan": "/plan inspect",
        "/init": "/init",
        "/implement": "/implement",
        "MCP prompt": "/server:prompt value",
        "MCP resource": "/mcp resources server mcp://doc/1",
        "escaped slash": "//status",
        "unknown slash": "/not-a-command",
        "attachments/images": "inspect @note.txt",
        "paste expansion": "inspect [Image #1]",
        "steer/queued input": "steer now",
    }[path]
    if path in {"$skill", "/skill"}:
        _write_audit_skill(project)
    if path in {"custom command", "custom inline shell"}:
        command_dir = project / ".zeta" / "commands"
        command_dir.mkdir(parents=True, exist_ok=True)
        name = "custom" if path == "custom command" else "custom-shell"
        body = (
            "expanded $ARGUMENTS"
            if path == "custom command"
            else "expanded !`printf generated`"
        )
        (command_dir / f"{name}.md").write_text(body, encoding="utf-8")
    if path == "/init":
        await asyncio.to_thread(
            subprocess.run, ["git", "init", "--quiet", str(project)], check=True
        )
    if path == "attachments/images":
        (project / "note.txt").write_text("evidence", encoding="utf-8")

    store = ConversationStore(tmp_path / "tui-session", cwd=project)
    catalog = discover_session_skills(project_dir=project)
    backend: object = FakeBackend([ScriptedTurn([TextContent("done")])])
    tools = None
    if path == "steer/queued input":
        backend = SteerToolBackend(ToolCall("call-1", "noop", {}))

        async def noop_tool(
            _arguments: dict[str, object],
            _abort_signal: object,
            _publisher: object,
        ) -> str:
            return "ok"

        tools = {"noop": noop_tool}
    loop = AgentLoop(backend, store, tools=tools, max_turns=3, skill_catalog=catalog)
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        zeta_home=tmp_path / "home",
        console=Console(file=StringIO(), force_terminal=False),
    )
    if path == "/implement":
        loop.set_plan_mode(True)
    if path == "MCP prompt":
        app._slash_commands.set_mcp_prompts(
            [
                (
                    "server:prompt",
                    "server",
                    MCPPrompt(
                        "prompt", arguments=(MCPPromptArgument("topic", required=True),)
                    ),
                )
            ]
        )

        async def prompt(_name: str, arguments: dict[str, str]) -> str:
            return f"resolved {arguments['topic']}"

        loop.slash_mcp_prompt = prompt  # type: ignore[method-assign]
    if path == "MCP resource":

        async def resource(_args: str) -> ModelInputEnvelope:
            return ModelInputEnvelope(
                "resource contents", "", MessageOrigin.SLASH_EXPANSION
            )

        loop.slash_mcp = resource  # type: ignore[method-assign]

    if path == "steer/queued input":
        first = asyncio.create_task(app._handle_prompt_value("start"))
        assert isinstance(backend, SteerToolBackend)
        await backend.turn1_streaming.wait()
        await app._handle_prompt_value(typed)
        backend.release_turn1.set()
        await first
    else:
        await app._handle_prompt_value(typed)
    assert app._active_task is not None
    await app._active_task
    user_rows = [
        entry.to_dict()
        for entry in store.entries
        if entry.type == "message" and entry.data["message"]["role"] == "user"
    ]
    row = user_rows[-1] if path == "steer/queued input" else user_rows[0]
    await app.close()
    return row


async def _audit_server_row(path: str, tmp_path: Path) -> dict[str, object]:
    import base64

    project = tmp_path / "project"
    command_dir = project / ".zeta" / "commands"
    command_dir.mkdir(parents=True)
    (command_dir / "custom.md").write_text("expanded $ARGUMENTS", encoding="utf-8")
    turns = [ScriptedTurn([TextContent("done")]), ScriptedTurn([TextContent("done")])]
    server = ZetaServer(
        home=tmp_path / "home",
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_args, **_kwargs: (FakeBackend(turns), "offline"),
    )
    reader, writer = await _connect(server)
    try:
        hello = (
            await _request(
                reader,
                writer,
                1,
                "hello",
                {"protocol_version": "1.1", "features": ["model_input_ids"]},
            )
        )[-1]["result"]
        assert "model_input_ids" in hello["capabilities"]["features"]
        sid = (await _request(reader, writer, 2, "new_session"))[-1]["result"][
            "session"
        ]["session_id"]
        input_id = None
        if path == "serve send":
            await _request(reader, writer, 3, "send", {"text": "serve send input"})
        elif path == "serve slash input":
            result = (
                await _request(
                    reader,
                    writer,
                    3,
                    "slash_run",
                    {"session_id": sid, "text": "/custom this"},
                )
            )[-1]["result"]
            assert "text" not in result and "origin" not in result
            input_id = result["input_id"]
            await _request(reader, writer, 4, "send", {"input_id": input_id})
        elif path == "serve send_images":
            png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            await _request(
                reader,
                writer,
                3,
                "send_images",
                {
                    "session_id": sid,
                    "text": "inspect image",
                    "images": [
                        {
                            "name": "shot.png",
                            "mime_type": "image/png",
                            "data": base64.b64encode(png).decode(),
                        }
                    ],
                },
            )
        elif path == "serve steer":
            await _request(reader, writer, 3, "send", {"text": "start"})
            await _request(reader, writer, 4, "steer", {"text": "serve steer input"})
        else:
            raise AssertionError(path)
        assert server._client is not None and server._client._turn_task is not None
        await server._client._turn_task
        if input_id is not None:
            reused = await _request(reader, writer, 5, "send", {"input_id": input_id})
            assert reused[-1]["error"]["code"] == -32602
        messages = [
            entry.to_dict()
            for entry in server.runtime.opened.store.entries
            if entry.type == "message"
            and entry.data["message"]["role"] == MessageRole.USER.value
        ]
        return messages[-1] if path == "serve steer" else messages[0]
    finally:
        await _close(server, writer)


async def _audit_production_row(path: str, tmp_path: Path) -> dict[str, object]:
    tui_paths = {
        "plain text",
        "$skill",
        "/skill",
        "custom command",
        "custom inline shell",
        "/plan",
        "/init",
        "/implement",
        "MCP prompt",
        "MCP resource",
        "escaped slash",
        "unknown slash",
        "attachments/images",
        "paste expansion",
        "steer/queued input",
    }
    if path in tui_paths:
        return await _audit_tui_row(path, tmp_path)
    if path.startswith("serve "):
        return await _audit_server_row(path, tmp_path)
    if path == "inbox wake":
        store = ConversationStore(tmp_path / "inbox")
        loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
        loop.tool_registry.register(
            "inbox",
            lambda _arguments: "unused",
            parameters={"type": "object", "properties": {}},
            requires_approval=False,
        )
        loop.tool_registry.project_registry = object()
        loop.tool_registry.project_id = "project"

        class Inbox:
            @staticmethod
            def claim_wake(_project_id: str, _message_ids: tuple[str, ...]) -> bool:
                return True

        class Scanner:
            inbox = Inbox()

            @staticmethod
            def scan() -> tuple[str, ...]:
                return ("message",)

        loop._inbox_scanner = Scanner()
        await loop._check_project_inbox()
        row = store.entries[-1].to_dict()
        await loop.close()
        return row
    raise AssertionError(f"missing audit production path: {path}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "origin", "display_text"),
    _MODEL_INPUT_AUDIT,
    ids=[case[0] for case in _MODEL_INPUT_AUDIT],
)
async def test_model_input_audit_preserves_authorship_and_nested_user_input(
    tmp_path: Path,
    path: str,
    origin: MessageOrigin,
    display_text: str | None,
) -> None:
    row = await _audit_production_row(path, tmp_path)
    rendered = _rendered(row)
    data = row["data"]
    assert isinstance(data, dict)
    if row["type"] == "message":
        message = data["message"]
        assert isinstance(message, dict)
        actual_origin = message["metadata"][MESSAGE_ORIGIN_METADATA]
        assert message["metadata"].get("zeta.user_display_text") == display_text
    else:
        actual_origin = data["origin"]

    assert actual_origin == origin.value
    expected_authorship = (
        "harness_notification" if origin is MessageOrigin.NOTIFICATION else origin.value
    )
    assert rendered["authorship"] == expected_authorship
    if origin in {MessageOrigin.SKILL_EXPANSION, MessageOrigin.SLASH_EXPANSION}:
        assert rendered["user_authored_input"] == {
            "authorship": MessageOrigin.USER.value,
            "text": display_text,
        }
    else:
        assert "user_authored_input" not in rendered


def test_headless_entry_persists_user_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "headless")
    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn([TextContent("done")])]),
            store,
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    monkeypatch.setattr("zeta.tui.app.create_app", lambda _args: app)
    monkeypatch.setattr(sys, "stdout", StringIO())
    monkeypatch.setattr(sys, "stderr", StringIO())

    code = run_headless(
        SimpleNamespace(format="text", require_tools=False, compaction=None),
        "headless input",
    )

    assert code == 0
    row = store.entries[0].to_dict()
    metadata = row["data"]["message"]["metadata"]
    assert metadata[MESSAGE_ORIGIN_METADATA] == "user"
    assert "zeta.user_display_text" not in metadata
    assert _label(row) == "user"


@pytest.mark.asyncio
async def test_automation_runner_persists_automation_prompt_origin(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path) as automation_store:
        _arm(automation_store, _job(tmp_path))
        occurrence = tick(automation_store, DUE)[0]
        run_id = automation_store.claim(occurrence)
        assert run_id is not None
        await run_claimed(
            automation_store,
            occurrence,
            run_id,
            home=tmp_path,
            backend=FakeBackend([ScriptedTurn([TextContent("brief")])]),
            mount_factory=_empty_mount,
            delivery=RecordingDelivery(),
        )
        session_id = automation_store.runs("brief")[0].session_id
    session = SessionManager(tmp_path).open(session_id)
    row = next(
        entry.to_dict()
        for entry in session.store.entries
        if entry.type == "message"
        and entry.data["message"]["role"] == MessageRole.USER.value
    )
    metadata = row["data"]["message"]["metadata"]
    assert metadata[MESSAGE_ORIGIN_METADATA] == "automation_prompt"
    assert "zeta.user_display_text" not in metadata
    assert _label(row) == "automation_prompt"
    session.store.close()


@pytest.mark.asyncio
async def test_child_spawn_persists_agent_prompt_origin(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    store = ConversationStore(tmp_path / "parent")
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    row = next(
        entry.to_dict()
        for entry in child_store.entries
        if entry.type == "message"
        and entry.data["message"]["role"] == MessageRole.USER.value
    )
    metadata = row["data"]["message"]["metadata"]
    assert metadata[MESSAGE_ORIGIN_METADATA] == "agent_prompt"
    assert "zeta.user_display_text" not in metadata
    assert _label(row) == "agent_prompt"
    child_store.close()
    await loop.close()


@pytest.mark.asyncio
async def test_agent_send_entry_persists_agent_send_origin(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path / "parent")
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)
    child_path = Path(str(store.agent_children()[handle]["child_session_path"]))
    assert send_to_run(store, handle, "also check tests") is None
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    row = next(
        entry.to_dict()
        for entry in child_store.entries
        if entry.type == "message"
        and entry.data["message"]["metadata"].get(MESSAGE_ORIGIN_METADATA)
        == MessageOrigin.AGENT_SEND.value
    )
    metadata = row["data"]["message"]["metadata"]
    assert metadata[MESSAGE_ORIGIN_METADATA] == "agent_send"
    assert "zeta.user_display_text" not in metadata
    assert _label(row) == "agent_send"
    child_store.close()
    await loop.close()


def test_agent_send_pending_prompt_is_not_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    entry = store.pending_prompt_queue.append(
        "agent follow-up", origin=MessageOrigin.AGENT_SEND
    )

    assert _label(entry.to_dict()) == "agent_send"


def test_empty_turn_nudge_is_not_labeled_user() -> None:
    assert _label(_message_row(build_nudge_message())) == "harness_nudge"


def test_agent_completion_notification_is_not_labeled_user(tmp_path: Path) -> None:
    entry = ConversationStore(tmp_path).append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="review",
        status="completed",
        text="done",
    )

    assert entry.data["origin"] == MessageOrigin.NOTIFICATION
    assert _label(entry.to_dict()) == "harness_notification"


def test_automation_receipt_is_labeled_harness(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _receipt(store, "delivered")

    assert _label(store.entries[-1].to_dict()) == "harness"


def _unmarked_user_message() -> Message:
    return Message(MessageRole.USER, [TextContent("unattributed text")])


def test_new_user_message_without_origin_is_rejected_by_store_append_paths(
    tmp_path: Path,
) -> None:
    for method_name in ("append_message", "append_message_with_approval_requests"):
        store = ConversationStore(tmp_path / method_name)
        before = store.path.read_bytes()
        with pytest.raises(ValueError, match="origin"):
            getattr(store, method_name)(_unmarked_user_message())
        assert store.entries == []
        assert store.path.read_bytes() == before


@pytest.mark.parametrize("origin", ["not-an-origin", MessageOrigin.UNKNOWN.value])
def test_new_user_message_with_invalid_origin_is_rejected(
    tmp_path: Path, origin: str
) -> None:
    store = ConversationStore(tmp_path)
    message = Message(
        MessageRole.USER,
        [TextContent("unattributed text")],
        metadata={MESSAGE_ORIGIN_METADATA: origin},
    )

    with pytest.raises(ValueError, match="origin"):
        store.append_message(message)


def test_new_user_message_without_origin_is_rejected_by_steer(tmp_path: Path) -> None:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )

    with pytest.raises(ValueError, match="origin"):
        loop.steer(_unmarked_user_message())

    assert loop.has_pending_steering is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin", "message"),
    [
        (MessageOrigin.USER, _unmarked_user_message()),
        (MessageOrigin.UNKNOWN, None),
    ],
    ids=["missing-metadata", "unknown-origin"],
)
async def test_new_user_message_without_origin_is_rejected_by_run_turn(
    tmp_path: Path, origin: MessageOrigin, message: Message | None
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())

    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="origin"):
        async for _event in loop.run_turn(
            "unattributed text", origin=origin, user_message=message
        ):
            pass

    assert store.entries == []
    assert store.path.read_bytes() == before
    await loop.close()


def test_historical_unmarked_user_row_still_replays_and_reconciles_as_harness_unknown(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="historical-session")
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("historical text")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    store.close()
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[-1]["data"]["message"].pop("metadata")
    store.path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    reopened = ConversationStore(tmp_path, session_id="historical-session")

    assert reopened.messages() == [
        Message(MessageRole.USER, [TextContent("historical text")])
    ]
    assert _label(reopened.entries[-1].to_dict()) == "harness_unknown"
