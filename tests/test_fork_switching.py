from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.fork_session import (
    ForkRuntimeMixin,
    ForkStackController,
    OwnedApproval,
)
from zeta.tui.runtime_close import RuntimeCloseMixin
from zeta.tui.slash_handlers.fork_view import ForkContext, ForkViewMixin

# -- request_open_fork / request_return_to_main keep the main turn alive -----


def test_request_open_fork_suspends_without_aborting_the_turn() -> None:
    aborts: list[str] = []
    app = SimpleNamespace(
        _open_fork_target=None,
        _exit_requested=False,
        _active_session=None,
        abort_active=lambda *a, **k: aborts.append("abort"),
    )
    TUIApp.request_open_fork(app, "fork-123")
    assert app._open_fork_target == "fork-123"
    assert getattr(app, "_run_result", None).name == "OPEN_CHILD"
    assert app._exit_requested is True
    # The in-flight main turn is never aborted when a discussion opens.
    assert aborts == []
    assert TUIApp.take_open_fork(app) == "fork-123"
    assert app._open_fork_target is None


def test_request_return_to_main_does_not_abort() -> None:
    aborts: list[str] = []
    app = SimpleNamespace(
        _exit_requested=False,
        _active_session=None,
        abort_active=lambda *a, **k: aborts.append("abort"),
    )
    TUIApp.request_return_to_main(app)
    assert getattr(app, "_run_result", None).name == "RETURN_TO_PARENT"
    assert app._exit_requested is True
    assert aborts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("finder_kind", ["history", "transcript"])
async def test_fork_switch_cancels_finder_and_restores_composer(
    tmp_path: Path, finder_kind: str
) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
            skip_mcp_mount=True,
        ),
        provider="codex",
        model="offline",
    )
    session = app._make_session()
    draft = Document("draft before switch", cursor_position=6)
    session.default_buffer.set_document(draft)
    app._active_session = session

    if finder_kind == "history":
        app._history_finder_open(session.default_buffer)
    else:
        app._finder_open()
    assert app._active_finder.finder_active
    app.request_open_fork("fork-123")
    app._prepare_ui_session_change()
    resumed = app._make_session()

    assert not app._active_finder.finder_active
    assert resumed.default_buffer.document == draft
    await app._close_finder_workers()


# -- return arming (review minor #2) ----------------------------------------


class _FakeForkApp(ForkViewMixin):
    def __init__(self, *, item_open: bool) -> None:
        self._fork_context = ForkContext(
            "Pick DB", "aid", "sid", "orchestrator", Path("/tmp/x")
        )
        self._fork_return_armed = False
        self._fork_returning = False
        self._item_open = item_open
        self.printed: list[str] = []
        self.returned = False

    def _fork_item_open(self) -> bool:
        return self._item_open

    def _print_system(self, message: str) -> None:
        self.printed.append(message)

    def request_return_to_main(self) -> None:
        self.returned = True


def test_open_item_arms_once_then_returns() -> None:
    app = _FakeForkApp(item_open=True)
    app._return_to_main()
    assert app._fork_return_armed is True
    assert app.returned is False
    app._return_to_main()
    assert app.returned is True


def test_esc_clears_arm_so_return_asks_again() -> None:
    app = _FakeForkApp(item_open=True)
    app._return_to_main()  # Enter on (main): arms.
    app.clear_fork_return_arm()  # Esc exits navigation.
    assert app._fork_return_armed is False
    app._return_to_main()  # Enter on (main) again: must ask, not leave.
    assert app.returned is False
    assert app._fork_return_armed is True


def test_resolved_item_returns_immediately() -> None:
    app = _FakeForkApp(item_open=False)
    app._return_to_main()
    assert app.returned is True


# -- resuming a suspended TUI does not replay its transcript -----------------


def test_tui_run_replays_transcript_only_on_initial_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def driver() -> None:
        app = object.__new__(TUIApp)
        replay_count = 0
        read_count = 0
        watchdog_events: list[tuple[str, str]] = []

        class Watchdog:
            async def close(self) -> None:
                watchdog_events.append(("close", "test-session"))

        def start_watchdog(
            loop: object, *, zeta_home: Path, session_id: str
        ) -> Watchdog:
            watchdog_events.append(("start", session_id))
            return Watchdog()

        monkeypatch.setattr("zeta.tui.app.start_tui_stall_watchdog", start_watchdog)

        class Loop:
            store = SimpleNamespace(session_id="test-session")

            async def activate(self) -> None:
                pass

            async def ensure_mcp_servers(self) -> None:
                pass

        async def rebuild() -> bool:
            nonlocal replay_count
            replay_count += 1
            return True

        async def read_prompt(session: object) -> None:
            nonlocal read_count
            read_count += 1
            if read_count == 1:
                TUIApp.request_open_fork(app, "fork-1")

        async def close() -> None:
            pass

        app.loop = Loop()
        app._zeta_home = None
        app._exit_requested = False
        app._decisions_switching = False
        app._session = None
        app._active_session = None
        app._active_task = None
        app._fork_controller = None
        app._open_fork_target = None
        app._input_loop_active = False
        app._startup_presented = True
        app._begin_startup_replay = lambda: None
        app._attach_draft = lambda session: None
        app._rebuild_transcript_async = rebuild
        app._finish_startup_replay = lambda **kwargs: None
        app._present_pending_approvals = lambda: None
        app.start_decisions_poll = lambda: None
        app._read_prompt = read_prompt
        app.close = close
        session = SimpleNamespace()

        first_result = await TUIApp.run(app, session)
        app._open_fork_target = None
        second_result = await TUIApp.run(app, session, resume_ui=True)

        assert getattr(first_result, "name", None) == "OPEN_CHILD"
        assert getattr(second_result, "name", None) == "EXIT"
        assert replay_count == 1
        assert watchdog_events == [
            ("start", "test-session"),
            ("close", "test-session"),
            ("start", "test-session"),
            ("close", "test-session"),
        ]

    asyncio.run(driver())


def test_approval_shortcut_submits_owner_qualified_handle() -> None:
    submitted: list[tuple[str, bool]] = []
    app = object.__new__(TUIApp)
    request = ApprovalRequest("same", ToolCall("same", "read", {"path": "x"}))
    app._fork_controller = SimpleNamespace(
        pending_approvals=(OwnedApproval(SimpleNamespace(), request, "approval-7"),)
    )
    app._submit_input = lambda value, *, internal: submitted.append((value, internal))

    TUIApp._answer_first_pending(app, "approve")

    assert submitted == [("/approve approval-7", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["approve", "deny", "always"])
async def test_live_approval_handle_survives_refresh_and_submission(
    tmp_path: Path, verb: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[dict[str, object]] = []

    def checked(arguments: dict[str, object]) -> str:
        executed.append(arguments)
        return "done"

    call = ToolCall("pending-call", "checked", {})
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(default="ask", store=store)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("finished")]),
        ]
    )
    app = TUIApp(
        AgentLoop(
            backend,
            store,
            tools={"checked": checked},
            approval_policy=policy,
            skill_catalog=SkillCatalog.empty(),
            skip_mcp_mount=True,
        ),
        provider="codex",
        model="offline",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    controller = ForkStackController(app, pytest.fail)
    app._active_session = app._make_session()
    ready = asyncio.Event()
    present = app._present_pending_approvals

    def show_approvals() -> None:
        present()
        ready.set()

    monkeypatch.setattr(app, "_present_pending_approvals", show_approvals)
    submission = asyncio.create_task(app._handle_prompt_value("run the tool"))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        handle = next(iter(app._approval_units))
        # Real policy reads reconstruct requests; the displayed handle must survive.
        assert app.first_pending_approval_key == handle
        controller.approvals_changed()
        assert tuple(app._approval_units) == (handle,)

        turn = app._submissions.provider_task
        assert turn is not None
        app._answer_first_pending(verb)
        await asyncio.wait_for(turn, timeout=5)

        expected = "deny" if verb == "deny" else "allow"
        assert store.approval_states()[call.id][1] == expected
        assert executed == ([] if verb == "deny" else [{}])
        assert not controller.pending_approvals
        assert not app._approval_units
        assert bool(policy.always_allow) == (verb == "always")
    finally:
        await app.close()
        await asyncio.gather(submission, return_exceptions=True)


def test_visible_approval_card_is_removed_when_request_ends() -> None:
    removed: list[object] = []
    unit = object()

    class Visible(ForkRuntimeMixin):
        pass

    app = Visible()
    app._init_fork_runtime()
    app._presenter = SimpleNamespace(print_unit=lambda card: unit)
    app._transcript = SimpleNamespace(
        remove=lambda old, *, leading_blank: removed.append(old)
    )
    app._invalidate_prompt = lambda: None
    owner = SimpleNamespace()
    request = ApprovalRequest("main", ToolCall("main", "read", {"path": "x"}))

    app.sync_visible_approvals((OwnedApproval(owner, request),))
    app.sync_visible_approvals(())

    assert removed == [unit]


def test_shortcut_card_tracks_first_pending_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    class Visible(ForkRuntimeMixin):
        def __init__(self, approvals: list[ApprovalRequest]) -> None:
            self._init_fork_runtime()
            self.approvals = approvals
            self.resolved: list[str | None] = []
            self._presenter = SimpleNamespace(print_unit=lambda card: card)
            self._transcript = SimpleNamespace(
                remove=lambda unit, *, leading_blank: None
            )
            self._invalidate_prompt = lambda: None

        @property
        def local_pending_approvals(self) -> tuple[ApprovalRequest, ...]:
            return tuple(self.approvals)

        async def resolve_local_approval(
            self,
            decision: object,
            requested_key: str | None,
            *,
            always: bool = False,
        ) -> None:
            self.resolved.append(requested_key)
            self.approvals = [
                request
                for request in self.approvals
                if str(request.key) != requested_key
            ]

    def render_card(
        tool_name: str, arguments: dict[str, object], **kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(key=kwargs["key"], shortcut=kwargs["shortcut"])

    async def driver() -> None:
        main = Visible(
            [ApprovalRequest("main", ToolCall("main", "read", {"path": "main"}))]
        )
        fork = Visible([])
        controller = ForkStackController(main, pytest.fail)
        controller._stack.append(fork)
        fork.set_fork_controller(controller)

        controller.approvals_changed()
        main_handle = controller.pending_approvals[0].handle
        assert [
            (unit.key, shortcut)
            for unit, shortcut in fork._approval_units.values()
        ] == [(main_handle, True)]

        fork.approvals = [
            ApprovalRequest("fork", ToolCall("fork", "read", {"path": "fork"}))
        ]
        controller.approvals_changed()
        first_handle = fork.first_pending_approval_key
        shortcut_handles = [
            unit.key
            for unit, shortcut in fork._approval_units.values()
            if shortcut
        ]
        assert shortcut_handles == [first_handle], (
            "exactly the approval resolved by y/n must advertise the shortcut"
        )

        await controller.resolve_approval(
            ApprovalDecision.ALLOW, fork.first_pending_approval_key
        )
        assert fork.resolved == ["fork"], "y must resolve the advertised fork approval"
        assert [
            (unit.key, shortcut)
            for unit, shortcut in fork._approval_units.values()
        ] == [(main_handle, True)], "main must gain y/n after the fork approval resolves"

    monkeypatch.setattr("zeta.tui.fork_session.render_approval_card", render_card)
    asyncio.run(driver())


def test_replaced_approval_handle_renders_new_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[object] = []
    cards: list[tuple[str, dict[str, object], str | None]] = []
    units: list[object] = []

    class Visible(ForkRuntimeMixin):
        pass

    def render_card(
        tool_name: str, arguments: dict[str, object], **kwargs: object
    ) -> tuple[str, dict[str, object], str | None]:
        card = (tool_name, arguments, kwargs.get("key"))
        cards.append(card)
        return card

    def print_unit(card: object) -> object:
        unit = object()
        units.append(unit)
        return unit

    monkeypatch.setattr("zeta.tui.fork_session.render_approval_card", render_card)
    app = Visible()
    app._init_fork_runtime()
    app._presenter = SimpleNamespace(print_unit=print_unit)
    app._transcript = SimpleNamespace(
        remove=lambda old, *, leading_blank: removed.append(old)
    )
    app._invalidate_prompt = lambda: None
    owner = SimpleNamespace()
    old_request = ApprovalRequest("same", ToolCall("same", "read", {"path": "x"}))
    replacement = ApprovalRequest(
        "same", ToolCall("same", "bash", {"command": "pwd"})
    )

    app.sync_visible_approvals((OwnedApproval(owner, old_request, "approval-1"),))
    app.sync_visible_approvals((OwnedApproval(owner, replacement, "approval-2"),))

    assert cards == [
        ("read", {"path": "x"}, "approval-1"),
        ("bash", {"command": "pwd"}, "approval-2"),
    ]
    assert removed == [units[0]]
    assert set(app._approval_units) == {"approval-2"}


def test_tui_close_failure_still_runs_later_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    cleaned: list[str] = []

    class ClosingApp(RuntimeCloseMixin):
        pass

    class Loop:
        store = SimpleNamespace(agent_notifications=lambda: ())
        tool_registry = SimpleNamespace(
            background_tasks=SimpleNamespace(
                set_notice_sink=lambda value: cleaned.append("task sink")
            )
        )

        def set_background_event_sink(self, value: object) -> None:
            cleaned.append("event sink")

        def set_background_wake_callback(self, value: object) -> None:
            cleaned.append("wake callback")

        def set_mcp_notice_sink(self, value: object) -> None:
            cleaned.append("mcp sink")

        def set_mcp_prompt_refresh(self, value: object) -> None:
            cleaned.append("prompt refresh")

    async def fail_finder() -> None:
        cleaned.append("finder")
        raise RuntimeError("finder close failed")

    async def async_cleanup(name: str) -> None:
        cleaned.append(name)

    async def close_session(*args: object, **kwargs: object) -> None:
        cleaned.append("session")

    monkeypatch.setattr("zeta.tui.runtime_close.close_session", close_session)
    app = ClosingApp()
    app._init_runtime_close()
    app.loop = Loop()
    app._terminal_restored = False
    app._workspace_snapshot_store = None
    app._hooks = None
    app._active_session = app._draft_session = app._session = object()
    app.stop_decisions_poll = lambda: cleaned.append("poll")
    app._stop_decisions_refresh = lambda: cleaned.append("refresh")
    app._close_finder_workers = fail_finder
    app._agent_navigation = SimpleNamespace(
        close=lambda: cleaned.append("navigation close"),
        unbind_layout=lambda: cleaned.append("navigation unbind"),
    )
    app._cancel_mcp_wizard = lambda: async_cleanup("wizard")
    app._submissions = SimpleNamespace(close=lambda: async_cleanup("submissions"))
    app._draft = SimpleNamespace(detach=lambda: cleaned.append("draft"))

    with pytest.raises(RuntimeError, match="finder close failed"):
        asyncio.run(app.close())

    assert app._closed is True
    assert "submissions" in cleaned
    assert "session" in cleaned
    assert "task sink" in cleaned
    assert cleaned.index("navigation close") < cleaned.index("navigation unbind")
    assert cleaned.index("navigation unbind") < cleaned.index("wizard")
