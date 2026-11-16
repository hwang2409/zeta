"""Real-TUI journey through a many-page root session.

Keys go through prompt-toolkit's real key processor (pipe input), and every
assertion reads the rows the renderer last drew for the root transcript window.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import shutil
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import StringIO
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.accounts.availability import AccountDisplay
from zeta.accounts.registry import AccountId
from zeta.accounts.session import AccountListSnapshot
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.remote.client import RemoteSessionClient
from zeta.remote.discovery import RemoteSession
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.transcript.paged_history import (
    HistoryCursor,
    PagedTranscriptHistory,
    TUIHistoryPage,
    TUIHistoryUnit,
    TUIPaintSegment,
)

TURNS = 100
TOOLS_PER_TURN = 15
# Unique words that only one old turn contains, both far outside the tail pages.
OLD_WORDS = {3: "quokka", 40: "wombat"}
CTRL_UP = "\x1b[1;5A"
CTRL_DOWN = "\x1b[1;5B"
UP = "\x1b[A"
PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"
CTRL_F = "\x06"
CTRL_R = "\x12"
ENTER = "\r"
ESCAPE = "\x1b"
BACKSPACE = "\x7f"
ROW_ORDER = re.compile(r"(prompt|answer|line one)[- ](\d+)(?:-(\d+))?")


def _prompt(turn: int) -> str:
    return f"prompt-{turn:04d}"


def _answer(turn: int) -> str:
    return f"answer-{turn:04d}"


@dataclass(frozen=True)
class RootHome:
    home: Path
    root: Path
    session_id: str


@pytest.fixture(scope="module")
def root_home(tmp_path_factory: pytest.TempPathFactory) -> RootHome:
    """Write a ~3,600-row root session with the shape of a real long session.

    Each turn has an agent notification that the TUI already presented (a row
    the history pages skip). Compactions larger than the 1 MiB page budget sit
    just before or just after some prompts, so the page next to such a prompt
    holds only a few rows.
    """

    home = tmp_path_factory.mktemp("root-nav")
    store = ConversationStore(home / "sessions")
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("session start")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    parent_id = store.branch_refs()[-1].id
    seq = store.branch_refs()[-1].seq
    root, session_id, log_path = store.root_dir, store.session_id, store.path
    store.close()
    huge_summary = "x" * (1024 * 1024 + 1)

    with log_path.open("a", encoding="utf-8") as handle:

        def write(entry_type: str, data: dict) -> str:
            nonlocal parent_id, seq
            seq += 1
            entry_id = f"{seq:032x}"
            row = {
                "seq": seq,
                "id": entry_id,
                "parent_id": parent_id,
                "lane": "main",
                "type": entry_type,
                "data": data,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            parent_id = entry_id
            return entry_id

        def message(value: Message) -> None:
            write("message", {"message": value.to_dict()})

        def compaction() -> None:
            write(
                "compaction",
                {
                    "replaces": [],
                    "source_seq_start": 1,
                    "source_seq_end": seq,
                    "summary": huge_summary,
                },
            )

        for turn in range(TURNS):
            word = OLD_WORDS.get(turn, "")
            if turn % 10 == 5:
                compaction()
            message(
                Message(
                    MessageRole.USER,
                    [TextContent(f"{_prompt(turn)} please check the build {word}".strip())],
                    metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
                )
            )
            if turn % 10 == 7:
                compaction()
            # The last turn is short, so its prompt cannot reach the top row.
            for tool in range(TOOLS_PER_TURN if turn < TURNS - 1 else 0):
                call = ToolCall(f"call-{turn}-{tool}", "read", {"path": f"f{tool}.txt"})
                message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
                message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        tool_result=ToolResult(
                            call.id,
                            # Uneven card heights, so a page edge can land on any row.
                            f"line one {turn}-{tool}" + "\nline two" * ((turn * 7 + tool) % 4),
                        ),
                    )
                )
                if tool == TOOLS_PER_TURN // 2:
                    notification = write(
                        "notification",
                        {
                            "child_instance_id": f"{session_id}:{turn + 1}",
                            "child_session_path": str(home),
                            "description": "review",
                            "kind": "agent_completion",
                            "origin": "notification",
                            "status": "completed",
                            "text": "done: reviewed",
                        },
                    )
                    write("notification_tui_presented", {"notification_id": notification})
                    write("notification_ack", {"notification_id": notification})
            message(
                Message(
                    MessageRole.ASSISTANT,
                    [TextContent(f"{_answer(turn)} done.\nsecond line\nthird line")],
                )
            )
    return RootHome(home, root, session_id)


@dataclass
class Driver:
    app: TUIApp
    session: object
    pipe: object

    def rows(self) -> list[str]:
        """Rows the renderer last drew for the viewed transcript window."""

        screen = self.session.app.renderer._last_screen
        if screen is None:
            return []
        navigation = self.app._agent_navigation
        control = (
            navigation.transcript_control
            if navigation.child_view_active
            else self.app._transcript
        )
        position = next(
            (
                position
                for window, position in screen.visible_windows_to_write_positions.items()
                if window.content is control
            ),
            None,
        )
        if position is None:
            return []
        return [
            "".join(
                screen.data_buffer[y][x].char
                for x in range(position.xpos, position.xpos + position.width)
            ).rstrip()
            for y in range(position.ypos, position.ypos + position.height)
        ]

    def all_rows(self) -> list[str]:
        """Every row the renderer last drew, composer included."""

        screen = self.session.app.renderer._last_screen
        if screen is None:
            return []
        size = self.session.app.output.get_size()
        return [
            "".join(screen.data_buffer[y][x].char for x in range(size.columns)).rstrip()
            for y in range(size.rows)
        ]

    def top(self) -> str:
        return next((row for row in self.rows() if row.strip()), "")

    def finder_drawn(self) -> bool:
        """True while the renderer last drew the finder overlay."""

        screen = self.session.app.renderer._last_screen
        return screen is not None and any(
            window.content is self.app._finder_control
            for window in screen.visible_windows_to_write_positions
        )

    def order(self) -> list[tuple[int, int]]:
        """(turn, place) of each numbered row drawn, top to bottom."""

        places = []
        for row in self.rows():
            for match in ROW_ORDER.finditer(row):
                kind, turn, tool = match.groups()
                place = {"prompt": 0, "answer": 99}.get(kind, int(tool or 0) + 1)
                places.append((int(turn), place))
        return places

    def assert_in_order(self, what: str) -> None:
        places = self.order()
        assert places == sorted(set(places)), (
            f"{what}: rows out of order or repeated {places}; screen:\n{self.screen()}"
        )

    def screen(self) -> str:
        return "\n".join(self.rows())

    async def send(self, data: str) -> None:
        self.pipe.send_bytes(data.encode())
        await asyncio.sleep(0)

    async def until(
        self, check: Callable[[], bool], *, what: str, seconds: float = 20
    ) -> None:
        try:
            async with asyncio.timeout(seconds):
                while not check():
                    # test-contract: latency real prompt-toolkit input and paint
                    await asyncio.sleep(0.01)
        except TimeoutError as exc:
            raise AssertionError(
                f"{what}; top row {self.top()!r}; screen:\n{self.screen()}"
            ) from exc

    async def settle(self, what: str) -> None:
        """Wait until idle holds across one more loop turn, after queued callbacks."""

        for _turn in range(2):
            await self.until(self.idle, what=what)
            # test-contract: latency let callbacks queued by a finished load run
            await asyncio.sleep(0.01)
        await self.until(self.idle, what=what)

    def idle(self) -> bool:
        return (
            not self.session.app.invalidated
            and not self.app._agent_navigation.transcript_control.history_pager._loads
            and self.app._finder_prepare_task is None
            and self.app._finder_rank_task is None
            and (
                getattr(self.app, "_user_nav_task", None) is None
                or self.app._user_nav_task.done()
            )
            and (
                self.app._agent_navigation.transcript_control.search_task is None
                or self.app._agent_navigation.transcript_control.search_task.done()
            )
            and not self.app._agent_navigation.transcript_control.prompt_history.busy
        )


@asynccontextmanager
async def running_tui(
    home: RootHome, backend: FakeBackend | None = None, *, painted: bool = True
) -> AsyncIterator[Driver]:
    store = await asyncio.to_thread(
        ConversationStore, home.root, session_id=home.session_id
    )
    app = TUIApp(
        AgentLoop(backend or FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        resumed=True,
        zeta_home=home.home,
    )
    output = Vt100_Output(StringIO(), lambda: Size(rows=40, columns=120), enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        session = app._make_session()
        run_task = asyncio.create_task(app.run(session))
        try:
            driver = Driver(app, session, pipe)
            await driver.until(
                (lambda: _answer(TURNS - 1) in driver.screen())
                if painted
                # The screen is up and takes keys; the first history page is not drawn.
                else (lambda: app._full_screen_active() and bool(driver.all_rows())),
                what="first paint",
                seconds=90,
            )
            yield driver
        finally:
            app.request_exit()
            await await_or_fail(run_task, what="root navigation TUI shutdown")


async def _ctrl_up_to(driver: Driver, oldest: int) -> None:
    """Press ctrl+up from the tail; each press puts the next older prompt on top."""

    # The newest prompt already shows on the tail screen and cannot go higher.
    for turn in range(TURNS - 2, oldest - 1, -1):
        await driver.until(driver.idle, what="settled screen before ctrl+up")
        await driver.send(CTRL_UP)
        await driver.until(
            lambda turn=turn: _prompt(turn) in driver.top(),
            what=f"ctrl+up did not put {_prompt(turn)} on the top row",
        )
        await driver.until(driver.idle, what="settled screen after ctrl+up")
        driver.assert_in_order(f"after ctrl+up to {_prompt(turn)}")


async def _page_up_to(driver: Driver, turn: int) -> None:
    """Press PageUp until the prompt of an old turn shows, past evicted tail pages."""

    for _press in range(400):
        if _prompt(turn) in driver.screen():
            return
        await driver.until(driver.idle, what="settled screen before PageUp")
        rows, pages = driver.rows(), driver.app._agent_navigation.transcript_control.history_pager.descriptors
        await driver.send(PAGE_UP)
        await driver.until(
            lambda rows=rows, pages=pages: driver.rows() != rows
            or driver.app._agent_navigation.transcript_control.history_pager.descriptors != pages,
            what="PageUp neither moved the view nor loaded an older page",
        )
    raise AssertionError(f"PageUp never reached {_prompt(turn)}")


async def _ctrl_up_past(driver: Driver, turn: int) -> None:
    """Reach old history by prompt jumps without checking their order."""

    for _press in range(TURNS):
        top = driver.top()
        if top.startswith("prompt-") and int(top[7:11]) <= turn:
            return
        await driver.until(driver.idle, what="settled screen before ctrl+up")
        await driver.send(CTRL_UP)
        await driver.until(
            lambda top=top: driver.top() != top, what="ctrl+up did not move the view"
        )
    raise AssertionError(f"ctrl+up never reached {_prompt(turn)}")


@pytest.mark.asyncio
async def test_page_up_across_pages_shows_every_prompt(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        for turn in range(TURNS - 1, TURNS - 31, -1):
            await _page_up_to(driver, turn)
            driver.assert_in_order(f"PageUp to {_prompt(turn)}")


@pytest.mark.asyncio
async def test_ctrl_up_lands_on_each_older_prompt_across_pages(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        # 70 prompts back crosses several page reveals and compaction pages.
        await _ctrl_up_to(driver, TURNS - 70)


@pytest.mark.asyncio
async def test_ctrl_down_returns_to_the_tail(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        # 30 turns back is far past the 4 resident pages, so the tail is evicted.
        oldest = TURNS - 30
        await _ctrl_up_past(driver, oldest)
        first = int(driver.top()[7:11]) + 1
        for turn in range(first, TURNS):
            await driver.until(driver.idle, what="settled screen before ctrl+down")
            await driver.send(CTRL_DOWN)
            await driver.until(
                lambda turn=turn: _prompt(turn) in driver.top()
                or (_prompt(turn) in driver.screen() and _answer(TURNS - 1) in driver.screen()),
                what=f"ctrl+down did not bring {_prompt(turn)} to the top row",
            )
        await driver.until(driver.idle, what="settled screen before last ctrl+down")
        await driver.send(CTRL_DOWN)
        await driver.until(
            lambda: _answer(TURNS - 1) in driver.screen(),
            what="ctrl+down did not return to the newest answer",
        )


async def _page_down_to_tail(driver: Driver) -> None:
    """PageDown until the newest answer shows; no press may move back, stall, or skip."""

    seen: set[tuple[int, int]] = set()
    for press in range(400):
        await driver.until(driver.idle, what="settled screen before PageDown")
        driver.assert_in_order(f"before PageDown {press + 1}")
        seen.update(driver.order())
        if _answer(TURNS - 1) in driver.screen():
            first = min(seen)[0]
            skipped = [_prompt(turn) for turn in range(first + 1, TURNS) if (turn, 0) not in seen]
            assert not skipped, f"PageDown never showed {skipped}"
            return
        before = driver.rows()
        top_before = driver.order()[:1]
        await driver.send(PAGE_DOWN)
        await driver.until(
            lambda before=before: driver.rows() != before,
            what=f"PageDown {press + 1} did not move the view",
        )
        await driver.until(driver.idle, what="settled screen after PageDown")
        assert driver.order()[:1] >= top_before, (
            f"PageDown {press + 1} jumped back from {top_before} to "
            f"{driver.order()[:1]}; screen:\n{driver.screen()}"
        )
    raise AssertionError("PageDown never reached the tail")


@pytest.mark.asyncio
async def test_page_down_loads_newer_pages_back_to_the_tail(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        await _ctrl_up_past(driver, TURNS - 30)
        await _page_down_to_tail(driver)


@pytest.mark.asyncio
async def test_page_down_from_a_prompt_behind_a_compaction(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        # A compaction sits right before prompt 75, so its page holds few rows
        # and the view keeps line numbers instead of the virtual window.
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send(_prompt(75))
        await driver.settle("the finder settles on the old prompt")
        await driver.send(ENTER)
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(75) in driver.top(),
            what=f"Enter did not put {_prompt(75)} on the top row",
        )
        # End the match highlight so PageDown scrolls the transcript.
        await driver.settle("accept settles")
        await driver.send(ESCAPE)
        await _page_down_to_tail(driver)


@pytest.mark.asyncio
async def test_finder_escape_after_old_preview_restores_the_tail(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.until(driver.idle, what="settled tail before the finder")
        tail = driver.rows()
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send(OLD_WORDS[3])
        await driver.until(
            lambda: _prompt(3) in driver.screen() and driver.idle(),
            what="the finder did not preview the old match",
        )
        # vi insert mode: the first Esc leaves insert mode, the second closes.
        await driver.send(ESCAPE)
        await driver.until(driver.idle, what="settled screen after the first Esc")
        await driver.send(ESCAPE)
        await driver.until(
            lambda: not driver.finder_drawn() and driver.idle(),
            what="Esc did not close the finder",
        )
        assert driver.rows() == tail, (
            f"Esc did not restore the tail; screen:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_finder_match_in_unloaded_history_opens_on_enter(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        app = driver.app
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send(OLD_WORDS[3])
        await driver.until(
            lambda: app._agent_navigation.transcript_control.search_task is not None and app._agent_navigation.transcript_control.search_task.done(),
            what="durable search for the old match",
        )
        await driver.until(driver.idle, what="settled screen before Enter")
        await driver.send(ENTER)
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(3) in driver.screen(),
            what="Enter on the old match did not close the finder and show it",
        )


@pytest.mark.asyncio
async def test_finder_enter_typed_while_rows_load_is_kept(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        # The query and Enter arrive in one write, before any row is ready.
        await driver.send(OLD_WORDS[40] + ENTER)
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(40) in driver.screen(),
            what="Enter typed during the load was dropped",
        )


@pytest.mark.asyncio
async def test_finder_typing_after_a_queued_enter_does_not_accept(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        # Enter queued for one query, then the query changes before rows are ready.
        await driver.send(OLD_WORDS[40] + ENTER + BACKSPACE * 6 + OLD_WORDS[3])
        await driver.until(
            lambda: _prompt(3) in driver.screen() and driver.idle(),
            what="the finder did not settle on the changed query",
        )
        await driver.settle("settled screen after the changed query")
        assert driver.finder_drawn(), (
            f"the Enter for the old query accepted the new one; screen:\n{driver.screen()}"
        )


@dataclass(frozen=True)
class Opener:
    """A key that opens an overlay, and what each typed burst must do in it."""

    key: str
    # A word that matches one entry, and the state once Enter accepts it.
    word: str
    accepted: Callable[[Driver], bool]
    drawn: Callable[[Driver], bool]
    # The overlay's own text field; None for an overlay that takes no text.
    field: Callable[[Driver], str] | None
    escapes: int
    # The state once Enter on an empty query is taken.
    empty_accepted: Callable[[Driver], bool]
    # Fakes the overlay's slow source before the keys (remote hosts, accounts).
    setup: Callable[[TUIApp, pytest.MonkeyPatch], None] = lambda _app, _patch: None


def _window_drawn(driver: Driver, window: object) -> bool:
    screen = driver.session.app.renderer._last_screen
    return screen is not None and window in screen.visible_windows_to_write_positions


def _decisions_drawn(driver: Driver) -> bool:
    return _window_drawn(driver, driver.app._decisions_panel_window)


def _panel(drawn: Callable[[Driver], bool]) -> dict:
    """A panel takes no text: typed letters do nothing, Enter keeps it open."""

    def kept(driver: Driver) -> bool:
        return drawn(driver) and driver.session.default_buffer.text == ""

    return {"drawn": drawn, "accepted": kept, "empty_accepted": kept}


PICKER_CARDS = (
    "select a model",
    "select an account",
    "enter select · esc cancel",
    "accounts · loading…",
    "remote · loading…",
)


def _picker_card_drawn(driver: Driver) -> bool:
    screen = driver.screen()
    return any(card in screen for card in PICKER_CARDS)


def _model_picker_closed(driver: Driver) -> bool:
    return not driver.app.model_picker_active and driver.session.default_buffer.text == ""


def _fake_remote_hosts(_app: TUIApp, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = (
        RemoteSession("lab", "a1", "alpha", None),
        RemoteSession("lab", "b2", "beta", None),
    )

    async def discover(_client: object, _hosts: object = None) -> tuple:
        return rows, ()

    async def attach(_client: object, _host: str, _session: object) -> None:
        raise ValueError("test: attach is not available")

    monkeypatch.setattr(RemoteSessionClient, "discover", discover)
    monkeypatch.setattr(RemoteSessionClient, "attach", attach)


def _fake_accounts(app: TUIApp, monkeypatch: pytest.MonkeyPatch) -> None:
    accounts = (AccountId("codex", "work"), AccountId("codex", "home"))

    async def list_snapshot() -> AccountListSnapshot:
        return AccountListSnapshot(
            tuple(AccountDisplay(account, account.name) for account in accounts),
            {"codex": accounts[0]},
            {"codex": accounts[0]},
        )

    context = SimpleNamespace(list_snapshot=list_snapshot, switching_unavailable=None)
    monkeypatch.setattr(app.loop, "account_context", context, raising=False)


def _loading_picker(active: Callable[[TUIApp], bool], setup: Callable) -> dict:
    """A picker whose rows load first: Enter before they show must not pick one."""

    def shown(driver: Driver) -> bool:
        return (
            active(driver.app)
            and _picker_card_drawn(driver)
            and driver.session.default_buffer.text == ""
        )

    return {
        "drawn": lambda driver: driver.app._picker_active and _picker_card_drawn(driver),
        "accepted": shown,
        "empty_accepted": shown,
        "setup": setup,
    }


OPENERS = [
    pytest.param(
        Opener(
            CTRL_F,
            OLD_WORDS[40],
            lambda driver: not driver.finder_drawn() and _prompt(40) in driver.screen(),
            Driver.finder_drawn,
            lambda driver: driver.app._finder_field.text,
            escapes=2,
            empty_accepted=lambda driver: not driver.finder_drawn(),
        ),
        id="ctrl+f",
    ),
    pytest.param(
        Opener(
            CTRL_R,
            _prompt(TURNS - 2),
            lambda driver: not driver.finder_drawn()
            and driver.session.default_buffer.text.startswith(_prompt(TURNS - 2)),
            Driver.finder_drawn,
            lambda driver: driver.app._finder_field.text,
            escapes=2,
            empty_accepted=lambda driver: not driver.finder_drawn()
            and driver.session.default_buffer.text.startswith(_prompt(TURNS - 1)),
        ),
        id="ctrl+r",
    ),
    pytest.param(
        Opener(
            "\x14",
            "zz",
            lambda driver: _decisions_drawn(driver)
            and driver.session.default_buffer.text == "",
            _decisions_drawn,
            None,
            escapes=1,
            empty_accepted=lambda driver: _decisions_drawn(driver)
            and driver.session.default_buffer.text == "",
        ),
        id="ctrl+t",
    ),
    # A bare slash command that opens a panel opens it from its own Enter.
    pytest.param(
        Opener(
            "/tasks" + ENTER,
            "zz",
            field=None,
            escapes=1,
            **_panel(lambda driver: _window_drawn(driver, driver.app._tasks_panel_window)),
        ),
        id="/tasks",
    ),
    pytest.param(
        Opener(
            "/status" + ENTER,
            "zz",
            field=None,
            escapes=1,
            **_panel(
                lambda driver: _window_drawn(driver, driver.app._status_card_window)
                and not driver.app._mcp_manager_open
            ),
        ),
        id="/status",
    ),
    pytest.param(
        Opener(
            "/mcp" + ENTER,
            "zz",
            field=None,
            escapes=1,
            **_panel(
                lambda driver: _window_drawn(driver, driver.app._status_card_window)
                and driver.app._mcp_manager_open
            ),
        ),
        id="/mcp",
    ),
    pytest.param(
        Opener("/decisions" + ENTER, "zz", field=None, escapes=1, **_panel(_decisions_drawn)),
        id="/decisions",
    ),
    # The pickers' keys are the arrows, Enter, and Esc.
    pytest.param(
        Opener(
            "/model" + ENTER,
            UP,
            _model_picker_closed,
            lambda driver: driver.app.model_picker_active and _picker_card_drawn(driver),
            None,
            escapes=1,
            empty_accepted=_model_picker_closed,
        ),
        id="/model",
    ),
    pytest.param(
        Opener(
            "/accounts" + ENTER,
            UP,
            field=None,
            escapes=1,
            **_loading_picker(lambda app: app.accounts_picker_active, _fake_accounts),
        ),
        id="/accounts",
    ),
    pytest.param(
        Opener(
            "/remote" + ENTER,
            UP,
            field=None,
            escapes=1,
            **_loading_picker(lambda app: app.remote_picker_active, _fake_remote_hosts),
        ),
        id="/remote",
    ),
]
# Pickers leave typed letters to the composer by design: no miss+enter case.
LETTERS_TO_COMPOSER = {"/model", "/accounts", "/remote"}
MISS = "zzsettled"


async def _own_home_with_held_startup(
    root_home: RootHome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    before_first_page: bool = False,
) -> tuple[RootHome, asyncio.Event]:
    """A copy of the home whose startup waits until the returned event is set.

    The wait is after the first history page is drawn (MCP servers start), or
    with ``before_first_page`` before it: the slowest part of a long session's
    startup, whose draw clears the transcript.
    """

    home = tmp_path / "home"
    # Own copy: on a failure the typed keys land in the session as a prompt.
    await asyncio.to_thread(shutil.copytree, root_home.home, home)
    copy = RootHome(home, home / root_home.root.relative_to(root_home.home), root_home.session_id)
    startup = asyncio.Event()
    owner, name = (
        (TUIApp, "_rebuild_transcript_async")
        if before_first_page
        else (AgentLoop, "ensure_mcp_servers")
    )
    step = getattr(owner, name)

    async def held_startup(self: object, *args: object, **kwargs: object) -> object:
        await await_or_fail(startup.wait(), timeout=60, what="startup gate")
        return await step(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, held_startup)
    return copy, startup


OPENER_CASES = [
    pytest.param(opener.values[0], phase, burst, id=f"{opener.id}-{phase}-{burst}")
    for opener in OPENERS
    for phase in ("before-first-page", "startup", "settled")
    for burst in ("word+enter", "miss+enter", "word+esc", "enter", "word")
    if not (opener.id in LETTERS_TO_COMPOSER and burst == "miss+enter")
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("opener", "phase", "burst"), OPENER_CASES)
async def test_keys_typed_right_after_an_overlay_opener_belong_to_the_overlay(
    root_home: RootHome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opener: Opener,
    phase: str,
    burst: str,
) -> None:
    # The opener and the keys after it arrive in one write, before the overlay's
    # first paint: they must reach the overlay, never composer submit.
    copy, startup = await _own_home_with_held_startup(
        root_home, tmp_path, monkeypatch, before_first_page=phase == "before-first-page"
    )
    if phase == "settled":
        startup.set()
    backend = FakeBackend([])
    async with running_tui(copy, backend, painted=phase != "before-first-page") as driver:
        app = driver.app
        opener.setup(app, monkeypatch)
        if phase != "settled":
            assert app._startup_replay_active, "startup finished before the keys"
        else:
            await driver.until(
                lambda: not app._startup_replay_active, what="startup did not finish"
            )
            await driver.settle("settled screen before the opener")

        def submitted() -> list[str]:
            # Slash commands are not prompts: the opener's own command may queue.
            return [
                value
                for value, _internal in app._startup_submissions
                if not value.startswith("/")
            ] + [str(messages[-1].content) for messages, _tools in backend.calls]

        def composer() -> str:
            return driver.session.default_buffer.text

        typed = {"miss+enter": MISS, "enter": ""}.get(burst, opener.word)
        if burst == "word":
            # Left open while startup ends.
            await driver.send(opener.key + typed)
            await driver.until(lambda: opener.drawn(driver), what="the overlay did not open")
            await driver.settle("overlay holds the typed keys")
            assert not submitted() and (
                opener.field is None or opener.field(driver) == typed
            ), f"the overlay lost {typed!r}: sent {submitted()}; screen:\n{driver.screen()}"
        elif burst == "word+esc":
            await driver.send(opener.key + typed)
            for _ in range(opener.escapes):
                await driver.until(lambda: opener.drawn(driver), what="the overlay did not open")
                await driver.settle("overlay holds the typed keys")
                await driver.send(ESCAPE)
                await driver.until(
                    lambda: not opener.drawn(driver)
                    or driver.session.app.vi_state.input_mode is InputMode.NAVIGATION,
                    what="Esc did nothing",
                )
            await driver.until(lambda: not opener.drawn(driver), what="Esc did not close it")
            await driver.settle("settled screen after Esc")
            assert not submitted(), f"keys typed for the overlay were sent: {submitted()}"
            assert not opener.drawn(driver) and composer() == "", (
                f"after Esc: overlay drawn={opener.drawn(driver)}, stray composer"
                f" text {composer()!r}; screen:\n{driver.screen()}"
            )
        else:
            await driver.send(opener.key + typed + ENTER)
            if phase == "before-first-page":
                # Nothing to accept before the first page: the keys wait for it.
                await driver.settle("keys held before the first page")
                assert not submitted(), f"keys typed for the overlay were sent: {submitted()}"
                startup.set()
                await driver.until(
                    lambda: not app._startup_replay_active, what="startup did not finish"
                )
            expected = {
                "word+enter": lambda: opener.accepted(driver),
                "enter": lambda: opener.empty_accepted(driver),
            }.get(
                burst,
                lambda: opener.drawn(driver)
                and (opener.field is None or opener.field(driver) == MISS),
            )
            await driver.until(
                lambda: bool(submitted()) or expected(),
                what=f"the overlay did not take {typed!r} + Enter",
            )
            await driver.settle("settled screen after Enter")
            assert not submitted(), f"keys typed for the overlay were sent: {submitted()}"
            if burst != "miss+enter":
                assert expected(), (
                    f"Enter did not accept {typed!r}: composer {composer()!r};"
                    f" screen:\n{driver.screen()}"
                )
            else:
                # Enter waits for rows that match the query; none match, so it stays.
                assert opener.drawn(driver) and composer() == "", (
                    f"Enter on a query with no match: overlay drawn={opener.drawn(driver)},"
                    f" composer {composer()!r}; screen:\n{driver.screen()}"
                )
                if opener.field is not None:
                    assert opener.field(driver) == MISS, (
                        f"the overlay query is {opener.field(driver)!r}, typed {MISS!r}"
                    )
        drawn = opener.drawn(driver)
        startup.set()
        # Startup sends its queued prompts in the same step that ends it; wait
        # for any turn they started, so a sent prompt reaches the backend.
        await driver.until(
            lambda: not app._startup_replay_active
            and (app._active_task is None or app._active_task.done()),
            what="startup or its queued turn did not finish",
        )
        await driver.settle("settled screen after startup")
        assert not backend.calls, (
            f"keys typed for the overlay were sent to the model: {submitted()}"
        )
        # The first history page draw clears the transcript: an open overlay
        # stays drawn, and a picker's keys never act on a card that is gone.
        assert opener.drawn(driver) == drawn, (
            f"startup changed the overlay: drawn {drawn} -> {opener.drawn(driver)};"
            f" screen:\n{driver.screen()}"
        )
        assert app._picker_active == _picker_card_drawn(driver), (
            f"picker keys active={app._picker_active}, card drawn="
            f"{_picker_card_drawn(driver)}; screen:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_a_prompt_queued_before_a_panel_command_is_still_sent(
    root_home: RootHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The panel opens from its Enter during startup; the prompt typed before it
    # still waits for startup and is sent, and only it.
    copy, startup = await _own_home_with_held_startup(root_home, tmp_path, monkeypatch)
    backend = FakeBackend([])
    async with running_tui(copy, backend) as driver:
        app = driver.app
        assert app._startup_replay_active, "startup finished before the keys"
        await driver.send("hello" + ENTER + "/tasks" + ENTER + "zz" + ENTER)
        await driver.settle("settled screen before startup ends")
        startup.set()
        await driver.until(
            lambda: bool(backend.calls)
            and (app._active_task is None or app._active_task.done()),
            what="the queued prompt was not sent after startup",
        )
        await driver.settle("settled screen after startup")
        sent = [str(messages[-1].content) for messages, _tools in backend.calls]
        assert sent == [str([TextContent("hello")])], f"sent {sent}, expected only 'hello'"
        assert _window_drawn(driver, app._tasks_panel_window), "the panel closed"


@pytest.mark.asyncio
async def test_esc_while_a_picker_loads_stops_it(
    root_home: RootHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy, startup = await _own_home_with_held_startup(root_home, tmp_path, monkeypatch)
    startup.set()
    _fake_remote_hosts(None, monkeypatch)
    loaded = RemoteSessionClient.discover
    gate = asyncio.Event()

    async def held_discover(client: object, hosts: object = None) -> tuple:
        await await_or_fail(gate.wait(), what="remote discovery gate")
        return await loaded(client, hosts)

    monkeypatch.setattr(RemoteSessionClient, "discover", held_discover)
    async with running_tui(copy) as driver:
        app = driver.app
        await driver.until(lambda: not app._startup_replay_active, what="startup did not finish")
        await driver.settle("settled screen before /remote")
        try:
            await driver.send("/remote" + ENTER)
            await driver.until(lambda: app._picker_active, what="/remote opened nothing")
            await driver.send(ESCAPE)
            await driver.until(lambda: not app._picker_active, what="Esc did not stop the load")
        finally:
            gate.set()
        await driver.settle("settled screen after the load would have ended")
        assert not app.remote_picker_active and driver.session.default_buffer.text == "", (
            f"the picker opened after Esc; screen:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_a_panel_behind_a_running_slash_command_owns_its_keys(
    root_home: RootHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # /compact holds the slash lock; /tasks only shows information, so it opens
    # from its Enter and the keys after it never wait in the composer as a prompt.
    copy, startup = await _own_home_with_held_startup(root_home, tmp_path, monkeypatch)
    startup.set()
    gate = asyncio.Event()

    async def held_compact(_app: TUIApp) -> str:
        await await_or_fail(gate.wait(), what="compact gate")
        return "compact: nothing to compact"

    monkeypatch.setattr(TUIApp, "slash_compact", held_compact)
    backend = FakeBackend([])
    async with running_tui(copy, backend) as driver:
        app = driver.app
        await driver.until(lambda: not app._startup_replay_active, what="startup did not finish")
        await driver.settle("settled screen before /compact")
        try:
            await driver.send("/compact" + ENTER)
            await driver.until(
                lambda: app.slash_lock.holder is not None, what="/compact did not start"
            )
            await driver.send("/tasks" + ENTER + "zz" + ENTER)
            await driver.until(
                lambda: _window_drawn(driver, app._tasks_panel_window),
                what="/tasks waited for /compact",
            )
        finally:
            gate.set()
        await driver.until(lambda: app.slash_lock.holder is None, what="/compact did not end")
        await driver.settle("settled screen after /compact")
        sent = [str(messages[-1].content) for messages, _tools in backend.calls]
        assert not sent and driver.session.default_buffer.text == "", (
            f"keys typed for the panel were sent {sent} or left in the composer"
            f" {driver.session.default_buffer.text!r}"
        )
        assert _window_drawn(driver, app._tasks_panel_window), "the panel closed"


@pytest.mark.asyncio
async def test_a_second_picker_command_replaces_the_one_still_loading(
    root_home: RootHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy, startup = await _own_home_with_held_startup(root_home, tmp_path, monkeypatch)
    startup.set()
    gate = asyncio.Event()
    async with running_tui(copy) as driver:
        app = driver.app
        _fake_accounts(app, monkeypatch)
        listed = app.loop.account_context.list_snapshot

        async def held_list() -> AccountListSnapshot:
            await await_or_fail(gate.wait(), what="accounts gate")
            return await listed()

        monkeypatch.setattr(app.loop.account_context, "list_snapshot", held_list)
        await driver.until(lambda: not app._startup_replay_active, what="startup did not finish")
        await driver.settle("settled screen before /accounts")
        try:
            await driver.send("/accounts" + ENTER + "/accounts" + ENTER)
            await driver.until(
                lambda: "accounts · loading…" in driver.screen(), what="/accounts opened nothing"
            )
            await driver.settle("both commands taken")
            assert driver.screen().count("accounts · loading…") == 1, (
                f"two loads for one picker; screen:\n{driver.screen()}"
            )
            await driver.send(ESCAPE)
            await driver.until(lambda: not app._picker_active, what="Esc did not stop the load")
        finally:
            gate.set()
        await driver.settle("settled screen after the loads would have ended")
        assert not app._picker_active and not _picker_card_drawn(driver), (
            f"a picker opened after Esc; screen:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_esc_while_remote_attaches_lets_the_attach_finish(
    root_home: RootHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One remote session: /remote attaches without a picker. Esc then must not
    # cancel the attach halfway and leave the client attached with no view.
    copy, startup = await _own_home_with_held_startup(root_home, tmp_path, monkeypatch)
    startup.set()
    row = RemoteSession("lab", "a1", "alpha", None)
    started, gate = asyncio.Event(), asyncio.Event()
    attached: list[str] = []

    async def discover(_client: object, _hosts: object = None) -> tuple:
        return (row,), ()

    async def attach(_client: object, _host: str, session: RemoteSession) -> None:
        started.set()
        await await_or_fail(gate.wait(), what="attach gate")
        attached.append(session.session_id)

    async def next_event(_client: object) -> str:
        await asyncio.Event().wait()
        return ""

    monkeypatch.setattr(RemoteSessionClient, "discover", discover)
    monkeypatch.setattr(RemoteSessionClient, "attach", attach)
    monkeypatch.setattr(RemoteSessionClient, "next_event", next_event)
    async with running_tui(copy) as driver:
        app = driver.app
        await driver.until(lambda: not app._startup_replay_active, what="startup did not finish")
        await driver.settle("settled screen before /remote")
        try:
            await driver.send("/remote" + ENTER)
            await await_or_fail(started.wait(), what="/remote did not attach")
            await driver.send(ESCAPE)
            await driver.settle("Esc taken during the attach")
        finally:
            gate.set()
        await driver.until(lambda: not app._picker_active, what="the attach did not end")
        await driver.settle("settled screen after the attach")
        assert attached == ["a1"] and app._remote_view is not None, (
            f"Esc cut the attach: attached={attached}, view={app._remote_view}"
        )
        await app.slash_detach("")


def _assert_pages_adjacent(driver: Driver, what: str) -> None:
    """Resident pages touch, come in seq order, and no page or row shows twice."""

    durable = [
        descriptor
        for descriptor in driver.app._agent_navigation.transcript_control.history_pager.descriptors
        if not descriptor.key.startswith("live:")
    ]
    keys = [descriptor.key for descriptor in durable]
    assert len(keys) == len(set(keys)), f"{what}: a page is resident twice {keys}"
    for older, newer in pairwise(durable):
        assert older.page.newer == newer.page.older, (
            f"{what}: pages not adjacent: {older.page.newer} then {newer.page.older}"
        )
    seqs = [
        unit.value.seq
        for descriptor in durable
        for unit in descriptor.page.units
        if ":dependency:" not in unit.entry_id
    ]
    assert seqs == sorted(set(seqs)), f"{what}: resident rows out of seq order"
    driver.assert_in_order(what)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [11, 23, 37])
async def test_random_navigation_keeps_pages_adjacent_and_in_order(
    root_home: RootHome, seed: int
) -> None:
    rng = random.Random(seed)
    keys = {"ctrl+up": CTRL_UP, "ctrl+down": CTRL_DOWN, "pageup": PAGE_UP, "pagedown": PAGE_DOWN}
    done: list[str] = []
    async with running_tui(root_home) as driver:
        for _step in range(40):
            action = rng.choice([*keys, "find+enter", "find+esc"])
            if action.startswith("find"):
                turn = rng.randrange(TURNS)
                done.append(f"{action} {_prompt(turn)}")
                await driver.send(CTRL_F)
                await driver.until(
                    driver.finder_drawn, what=f"seed {seed}: no finder after {done}"
                )
                await driver.send(_prompt(turn))
                await driver.settle(f"seed {seed}: finder settles after {done}")
                if action == "find+enter":
                    await driver.send(ENTER)
                else:
                    await driver.send(ESCAPE)
                    await driver.settle(f"seed {seed}: first Esc after {done}")
                    await driver.send(ESCAPE)
                await driver.until(
                    lambda: not driver.finder_drawn(),
                    what=f"seed {seed}: the finder stayed open after {done}",
                )
                if action == "find+enter":
                    # Accept leaves match highlighting on, which takes ctrl+f; Esc ends it.
                    await driver.settle(f"seed {seed}: accept settles after {done}")
                    await driver.send(ESCAPE)
            else:
                done.append(action)
                await driver.send(keys[action])
            await driver.settle(f"seed {seed}: screen settles after {done}")
            _assert_pages_adjacent(driver, f"seed {seed} after {done}")


class _RowPages:
    """Fake history source: page i holds the one row ``e{i}``; reads can be gated."""

    def __init__(self, total: int = 20) -> None:
        self.total = total
        self.gates: dict[str, asyncio.Event] = {}
        self.waiting: dict[str, asyncio.Event] = {}

    def row(self, index: int) -> TUIHistoryPage:
        return TUIHistoryPage(
            (TUIPaintSegment((TUIHistoryUnit(f"e{index}", index),)),),
            older=HistoryCursor(str(index)) if index else None,
            newer=HistoryCursor(str(index + 1)) if index + 1 < self.total else None,
        )

    async def _pass(self, name: str) -> None:
        if name in self.gates:
            self.waiting.setdefault(name, asyncio.Event()).set()
            await asyncio.wait_for(self.gates[name].wait(), 10)

    async def page(self, cursor=None, *, direction="older") -> TUIHistoryPage:
        if direction == "newer":
            await self._pass("newer")
            return self.row(int(cursor.value))
        return self.row(int(cursor.value) - 1 if cursor else self.total - 1)

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        await self._pass(entry_id)
        return self.row(int(entry_id[1:]))


def _resident(pager: PagedTranscriptHistory) -> list[str]:
    return [unit.entry_id for page in pager.pages for unit in page.units]


@pytest.mark.asyncio
async def test_a_page_whose_edge_moved_while_it_loaded_is_dropped() -> None:
    source = _RowPages()
    pager = PagedTranscriptHistory(source)
    pager.adopt(source.row(10))
    await pager.load("newer")
    await pager.load("newer")
    source.gates["newer"] = asyncio.Event()
    stale = asyncio.create_task(pager.load("newer"))
    try:
        await await_or_fail(
            asyncio.ensure_future(source.waiting.setdefault("newer", asyncio.Event()).wait()),
            what="the newer read of e13 started",
        )
        # Two older pages evict e12, so e13 no longer touches the newest page.
        await pager.load("older")
        await pager.load("older")
        assert _resident(pager) == ["e8", "e9", "e10", "e11"]
        source.gates["newer"].set()
        await await_or_fail(
            asyncio.gather(stale, return_exceptions=True), what="the stale e13 read"
        )
        # e13 is dropped; the newer edge is read again and e12 comes back.
        assert _resident(pager) == ["e9", "e10", "e11", "e12"], (
            "a page read for an edge that moved was attached"
        )
        await pager.load("older")
        assert _resident(pager) == ["e8", "e9", "e10", "e11"], "older paging is stuck"
    finally:
        stale.cancel()


@pytest.mark.asyncio
async def test_an_earlier_reveal_that_finishes_last_is_dropped() -> None:
    source = _RowPages()
    pager = PagedTranscriptHistory(source)
    source.gates["e3"] = asyncio.Event()
    first = asyncio.create_task(pager.reveal("e3"))
    try:
        await await_or_fail(
            asyncio.ensure_future(source.waiting.setdefault("e3", asyncio.Event()).wait()),
            what="the first reveal started",
        )
        await pager.reveal("e15")
        source.gates["e3"].set()
        await await_or_fail(first, what="the first reveal")
        assert _resident(pager)[0] == "e15", (
            f"the older reveal replaced the newer one: {_resident(pager)}"
        )
    finally:
        first.cancel()


async def _close_finder_with_escape(driver: Driver) -> None:
    # vi insert mode: the first Esc leaves insert mode, the second closes.
    await driver.send(ESCAPE)
    await driver.until(
        lambda: driver.session.app.vi_state.input_mode is InputMode.NAVIGATION,
        what="the first Esc did not leave insert mode",
    )
    await driver.send(ESCAPE)
    await driver.until(lambda: not driver.finder_drawn(), what="Esc did not close the finder")


@pytest.mark.asyncio
async def test_finder_escape_before_the_old_match_lands_keeps_the_view(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.settle("settled tail before the finder")
        tail = driver.rows()
        source = driver.app._agent_navigation.transcript_control.history_pager.source
        real_reveal, entered, gate = source.reveal, asyncio.Event(), asyncio.Event()

        async def gated_reveal(entry_id: str) -> TUIHistoryPage:
            entered.set()
            await asyncio.wait_for(gate.wait(), 10)
            return await real_reveal(entry_id)

        source.reveal = gated_reveal
        try:
            await driver.send(CTRL_F)
            await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
            await driver.send(OLD_WORDS[3])
            await await_or_fail(
                asyncio.ensure_future(entered.wait()), what="the old match started to load"
            )
            await _close_finder_with_escape(driver)
        finally:
            gate.set()
        await driver.settle("settled screen after the old match loaded")
        assert driver.rows() == tail, (
            f"the cancelled match moved the view; screen:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_clearing_the_query_before_the_old_match_lands_keeps_the_view(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.settle("settled tail before the finder")
        pager = driver.app._agent_navigation.transcript_control.history_pager
        tail_pages = _resident(pager)
        source = pager.source
        real_search, entered, gate = source.search, asyncio.Event(), asyncio.Event()

        async def gated_search(query: str, **kwargs: object) -> object:
            entered.set()
            await asyncio.wait_for(gate.wait(), 10)
            return await real_search(query, **kwargs)

        source.search = gated_search
        try:
            await driver.send(CTRL_F)
            await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
            await driver.send(OLD_WORDS[3])
            await await_or_fail(
                asyncio.ensure_future(entered.wait()), what="the old-history search started"
            )
            await driver.send(BACKSPACE * len(OLD_WORDS[3]))
            await driver.until(
                lambda: not driver.session.app.invalidated, what="the cleared query was drawn"
            )
        finally:
            gate.set()
        await driver.settle("settled screen after the old search ended")
        assert _resident(pager) == tail_pages, (
            "the search of a cleared query replaced the resident pages"
        )
        assert _prompt(3) not in driver.screen(), (
            f"the search of a cleared query drew its old match:\n{driver.screen()}"
        )


@pytest.mark.asyncio
async def test_finder_shows_searching_until_old_history_answers(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        app = driver.app
        source = app._agent_navigation.transcript_control.history_pager.source
        real_search, entered, gate = source.search, asyncio.Event(), asyncio.Event()

        async def gated_search(query: str, **kwargs: object) -> object:
            entered.set()
            await asyncio.wait_for(gate.wait(), 10)
            return await real_search(query, **kwargs)

        source.search = gated_search
        try:
            await driver.send(CTRL_F)
            await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
            await driver.send(OLD_WORDS[3])
            await await_or_fail(
                asyncio.ensure_future(entered.wait()), what="the old-history search started"
            )
            await driver.until(
                lambda: app._finder_prepare_task is None
                and app._finder_rank_task is None
                and not driver.session.app.invalidated,
                what="resident rows ranked",
            )
            drawn = "\n".join(driver.all_rows())
            assert "searching…" in drawn and "no matches" not in drawn, (
                f"the finder did not show that old history is still searched:\n{drawn}"
            )
            await driver.send(ENTER)
            await driver.until(
                lambda: not driver.session.app.invalidated, what="Enter was drawn"
            )
            assert driver.finder_drawn(), "Enter accepted before the old-history answer"
        finally:
            gate.set()
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(3) in driver.screen(),
            what="Enter did not open the old match once it was found",
        )


@pytest.mark.asyncio
async def test_finder_shows_no_matches_once_old_history_finds_nothing(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send("zzqqxx")
        await driver.until(
            lambda: driver.app._agent_navigation.transcript_control.search_task is not None
            and driver.app._agent_navigation.transcript_control.search_task.done(),
            what="durable search for a query with no hit",
        )
        await driver.settle("screen after the empty search")
        drawn = "\n".join(driver.all_rows())
        assert "no matches" in drawn and "searching…" not in drawn, (
            f"the finder still shows searching after the search ended:\n{drawn}"
        )


@pytest.mark.asyncio
async def test_finder_escape_returns_the_composer_to_insert_mode(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send(OLD_WORDS[40])
        await driver.settle("the finder settles")
        await _close_finder_with_escape(driver)
        await driver.settle("finder closed")
        # In vi normal mode "x" deletes and "a" appends, so this draws differently.
        await driver.send("xabc")
        await driver.settle("typed text drawn")
        composer = [row for row in driver.all_rows() if row.strip()][-6:]
        assert any("› xabc" in row for row in composer), (
            "typed text after finder Esc did not reach the composer as typed:\n"
            + "\n".join(composer)
        )


@pytest.mark.asyncio
async def test_page_up_while_a_reveal_loads_newer_pages_keeps_paging(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        app = driver.app
        await _ctrl_up_past(driver, TURNS - 30)
        await driver.settle("settled old history")
        source = app._agent_navigation.transcript_control.history_pager.source
        real_page, entered, gate = source.page, asyncio.Event(), asyncio.Event()

        async def gated_page(cursor=None, *, direction="older"):
            if direction == "newer":
                entered.set()
                await asyncio.wait_for(gate.wait(), 10)
            return await real_page(cursor, direction=direction)

        source.page = gated_page
        try:
            # ctrl+down past the resident pages reveals the next prompt from the
            # store, then reads newer pages after it.
            for _press in range(TURNS):
                rows = driver.rows()
                await driver.send(CTRL_DOWN)
                await driver.until(
                    lambda rows=rows: entered.is_set()
                    or (driver.rows() != rows and driver.idle()),
                    what="ctrl+down neither moved nor read a newer page",
                )
                if entered.is_set():
                    break
            assert entered.is_set(), "ctrl+down never revealed a prompt from the store"
            held = app._agent_navigation.transcript_control.history_pager.scalar_tokens[1]
            for press in range(400):
                await driver.send(PAGE_UP)
                await driver.until(
                    lambda: not driver.session.app.invalidated
                    and not any(key[2] == "older" for key in app._agent_navigation.transcript_control.history_pager._loads),
                    what=f"PageUp {press + 1} during the held reveal",
                )
                if app._agent_navigation.transcript_control.history_pager.scalar_tokens[1] != held:
                    break
            assert app._agent_navigation.transcript_control.history_pager.scalar_tokens[1] != held, (
                "PageUp never evicted the page the held newer read extends"
            )
        finally:
            gate.set()
        await driver.settle("settled after the held newer read")
        _assert_pages_adjacent(driver, "after a newer read whose edge moved")
        await _page_down_to_tail(driver)


@pytest.mark.asyncio
async def test_a_burst_of_ctrl_down_moves_one_prompt_per_press(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        await _ctrl_up_past(driver, TURNS - 40)
        await driver.settle("settled old prompt")
        start = int(driver.top()[7:11])
        # Six presses in one write, before any reveal finishes.
        await driver.send(CTRL_DOWN * 6)
        await driver.settle("settled after the burst")
        await driver.until(
            lambda: _prompt(start + 6) in driver.top(),
            what=f"six ctrl+down from {_prompt(start)} did not put {_prompt(start + 6)} on top",
        )


@pytest.mark.asyncio
async def test_ctrl_f_after_an_accepted_match_opens_a_new_search(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send(OLD_WORDS[40])
        await driver.settle("the finder settles on the first query")
        await driver.send(ENTER)
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(40) in driver.screen(),
            what="Enter on the first query did not show its match",
        )
        await driver.settle("accept settles")
        # The match highlight is still on; ctrl+f starts a new search anyway.
        await driver.send(CTRL_F)
        await driver.until(
            driver.finder_drawn, what="ctrl+f during the accepted search drew no finder"
        )
        await driver.send(OLD_WORDS[3])
        await driver.settle("the finder settles on the second query")
        await driver.send(ENTER)
        await driver.until(
            lambda: not driver.finder_drawn() and _prompt(3) in driver.screen(),
            what="Enter on the second query did not show its match",
        )
        composer = [row for row in driver.all_rows() if row.lstrip().startswith("›")]
        assert composer and not any(OLD_WORDS[3] in row for row in composer), (
            f"the second query went into the composer: {composer}"
        )


async def _home_with_a_child(root_home: RootHome, tmp_path: Path) -> RootHome:
    """A copy of the home whose root session has one running child: the same history."""

    home = tmp_path / "home"
    await asyncio.to_thread(shutil.copytree, root_home.home, home)
    copy = RootHome(home, home / root_home.root.relative_to(root_home.home), root_home.session_id)
    child = copy.root / copy.session_id / "agents" / "1"
    child.mkdir(parents=True)
    log = (copy.root / copy.session_id / "conversation.jsonl").read_text().splitlines()
    header = json.loads(log[0])
    header["data"]["session_id"] = "1"
    (child / "conversation.jsonl").write_text("\n".join([json.dumps(header), *log[1:]]) + "\n")
    (child / "agent_lifecycle.json").write_text(
        json.dumps({"agent_type": "worker", "description": "child", "depth": 1, "state": "running"})
    )
    return copy


async def _enter_the_child(driver: Driver) -> None:
    navigation = driver.app._agent_navigation
    await driver.until(lambda: bool(navigation.entries), what="no child in the agent list")
    await driver.until(driver.app.composer_can_focus_agent_list, what="agent list not reachable")
    await driver.send("\x1b[B")
    await driver.until(navigation.list_focused, what="Down did not focus the agent list")
    while navigation.entries[navigation.selected_index].path.name != "1":
        before = navigation.selected_index
        await driver.send("\x1b[B")
        await driver.until(lambda: navigation.selected_index != before, what="no next row")
    await driver.send(ENTER)
    await driver.until(
        lambda: navigation.child_view_active and _answer(TURNS - 1) in driver.screen(),
        what="the child view did not show its newest answer",
        seconds=90,
    )
    await navigation.wait_for_publication()


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["root", "child"])
@pytest.mark.parametrize("last_key", ["a", BACKSPACE], ids=["miss", "cleared"])
async def test_finder_typed_one_key_per_frame_searches_the_view_it_opened_on(
    root_home: RootHome, tmp_path: Path, last_key: str, view: str
) -> None:
    # "q" alone matches only turn 3's quokka, far behind the tail, so its search
    # replaces the pages. The next key must not leave the finder ranking those
    # pages: "qa" fuzzy-matches quokka but no text holds "qa", and a burst of the
    # same keys finds nothing.
    home = root_home if view == "root" else await _home_with_a_child(root_home, tmp_path)
    async with running_tui(home) as driver:
        if view == "child":
            await _enter_the_child(driver)
        await driver.settle("settled tail before the finder")
        pager = driver.app._agent_navigation.transcript_control.history_pager
        tail_pages, tail = _resident(pager), driver.rows()
        await driver.send(CTRL_F)
        await driver.until(driver.finder_drawn, what="ctrl+f did not draw the finder")
        await driver.send("q")
        await driver.until(
            lambda: _prompt(3) in driver.screen() and driver.idle(),
            what="the search for 'q' did not reveal turn 3",
        )
        await driver.settle("settled on 'q'")
        await driver.send(last_key)
        await driver.settle(f"settled after {last_key!r}")
        await driver.send(ENTER)
        await driver.settle("settled after Enter")
        if last_key == "a":
            assert driver.finder_drawn(), (
                "Enter on 'qa' accepted a row that only the search for 'q' brought in;"
                f" screen:\n{driver.screen()}"
            )
            await _close_finder_with_escape(driver)
            await driver.settle("settled after Esc")
            assert driver.rows() == tail, f"Esc did not restore the tail:\n{driver.screen()}"
        else:
            # An empty query lists the newest rows of the view the finder opened on.
            assert not driver.finder_drawn() and _answer(TURNS - 1) in driver.screen(), (
                "Enter on the cleared query jumped into pages the search for 'q'"
                f" revealed; screen:\n{driver.screen()}"
            )
        assert _resident(pager) == tail_pages, "the pages 'q' revealed are still resident"
