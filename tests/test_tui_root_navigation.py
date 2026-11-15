"""Real-TUI journey through a many-page root session.

Keys go through prompt-toolkit's real key processor (pipe input), and every
assertion reads the rows the renderer last drew for the root transcript window.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
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
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

TURNS = 100
TOOLS_PER_TURN = 15
# Unique words that only one old turn contains, both far outside the tail pages.
OLD_WORDS = {3: "quokka", 40: "wombat"}
CTRL_UP = "\x1b[1;5A"
CTRL_DOWN = "\x1b[1;5B"
PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"
CTRL_F = "\x06"
ENTER = "\r"
ESCAPE = "\x1b"


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
    """Write a 3,200-row root session (25 pages of 128 rows) once per module."""

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

    def message(value: Message) -> dict:
        return {"message": value.to_dict()}

    with log_path.open("a", encoding="utf-8") as handle:

        def write(value: Message) -> None:
            nonlocal parent_id, seq
            seq += 1
            entry_id = f"{seq:032x}"
            row = {
                "seq": seq,
                "id": entry_id,
                "parent_id": parent_id,
                "lane": "main",
                "type": "message",
                "data": message(value),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            parent_id = entry_id

        for turn in range(TURNS):
            word = OLD_WORDS.get(turn, "")
            write(
                Message(
                    MessageRole.USER,
                    [TextContent(f"{_prompt(turn)} please check the build {word}".strip())],
                    metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
                )
            )
            for tool in range(TOOLS_PER_TURN):
                call = ToolCall(f"call-{turn}-{tool}", "read", {"path": f"f{tool}.txt"})
                write(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
                write(
                    Message(
                        MessageRole.TOOL_RESULT,
                        tool_result=ToolResult(call.id, f"line one {turn}-{tool}\nline two"),
                    )
                )
            write(
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
        """Rows the renderer last drew for the root transcript window."""

        screen = self.session.app.renderer._last_screen
        if screen is None:
            return []
        position = next(
            (
                position
                for window, position in screen.visible_windows_to_write_positions.items()
                if window.content is self.app._transcript
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

    def top(self) -> str:
        return next((row for row in self.rows() if row.strip()), "")

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

    def idle(self) -> bool:
        return (
            not self.session.app.invalidated
            and not self.app._history_pager._loads
            and (
                self.app._history_search_task is None
                or self.app._history_search_task.done()
            )
        )


@asynccontextmanager
async def running_tui(home: RootHome) -> AsyncIterator[Driver]:
    store = await asyncio.to_thread(
        ConversationStore, home.root, session_id=home.session_id
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
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
                lambda: _answer(TURNS - 1) in driver.screen(),
                what="first paint shows the newest answer",
                seconds=90,
            )
            yield driver
        finally:
            app.request_exit()
            await await_or_fail(run_task, what="root navigation TUI shutdown")


async def _ctrl_up_to(driver: Driver, oldest: int) -> None:
    """Press ctrl+up from the tail; each press puts the next older prompt on top."""

    for turn in range(TURNS - 1, oldest - 1, -1):
        await driver.until(driver.idle, what="settled screen before ctrl+up")
        await driver.send(CTRL_UP)
        await driver.until(
            lambda turn=turn: _prompt(turn) in driver.top(),
            what=f"ctrl+up did not put {_prompt(turn)} on the top row",
        )


async def _page_up_to(driver: Driver, turn: int) -> None:
    """Press PageUp until the prompt of an old turn shows, past evicted tail pages."""

    for _press in range(400):
        if _prompt(turn) in driver.screen():
            return
        await driver.until(driver.idle, what="settled screen before PageUp")
        rows, pages = driver.rows(), driver.app._history_pager.descriptors
        await driver.send(PAGE_UP)
        await driver.until(
            lambda rows=rows, pages=pages: driver.rows() != rows
            or driver.app._history_pager.descriptors != pages,
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


@pytest.mark.asyncio
async def test_ctrl_up_lands_on_each_older_prompt_across_pages(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        # 70 prompts back crosses several 128-row page reveals.
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



@pytest.mark.asyncio
async def test_page_down_loads_newer_pages_back_to_the_tail(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        await _ctrl_up_past(driver, TURNS - 30)
        for press in range(400):
            if _answer(TURNS - 1) in driver.screen():
                break
            await driver.until(driver.idle, what="settled screen before PageDown")
            before = driver.rows()
            await driver.send(PAGE_DOWN)
            await driver.until(
                lambda before=before: driver.rows() != before,
                what=f"PageDown {press + 1} did not move the view",
            )
        assert _answer(TURNS - 1) in driver.screen(), "PageDown never reached the tail"


@pytest.mark.asyncio
async def test_finder_match_in_unloaded_history_opens_on_enter(
    root_home: RootHome,
) -> None:
    async with running_tui(root_home) as driver:
        app = driver.app
        await driver.send(CTRL_F)
        await driver.until(lambda: app._active_finder.finder_active, what="open finder")
        await driver.send(OLD_WORDS[3])
        await driver.until(
            lambda: app._history_search_task is not None and app._history_search_task.done(),
            what="durable search for the old match",
        )
        await driver.until(driver.idle, what="settled screen before Enter")
        await driver.send(ENTER)
        await driver.until(
            lambda: not app._active_finder.finder_active and _prompt(3) in driver.screen(),
            what="Enter on the old match did not close the finder and show it",
        )


@pytest.mark.asyncio
async def test_finder_enter_typed_while_rows_load_is_kept(root_home: RootHome) -> None:
    async with running_tui(root_home) as driver:
        app = driver.app
        await driver.send(CTRL_F)
        await driver.until(lambda: app._active_finder.finder_active, what="open finder")
        # The query and Enter arrive in one write, before any row is ready.
        await driver.send(OLD_WORDS[40] + ENTER)
        await driver.until(
            lambda: not app._active_finder.finder_active and _prompt(40) in driver.screen(),
            what="Enter typed during the load was dropped",
        )
