from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Callable
from io import StringIO
from time import perf_counter

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.text import Text

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.tui_child_view_home import (
    CHILD_ID,
    CHILD_PROMPT,
    CHILD_PROMPT_END,
    ROOT_MARKER,
    ROOT_PAIR_MARKER,
    VISIBLE_PROMPT,
    ChildViewHome,
    generate_child_view_home,
)
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp


async def _wait_until(check: Callable[[], bool], *, what: str) -> None:
    try:
        async with asyncio.timeout(90):
            while not check():
                # test-contract: latency real prompt-toolkit input processing
                await asyncio.sleep(0.01)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


def _drawn_terminal_text(app: TUIApp) -> str:
    """Read the characters from the terminal's current completed screen."""

    screen = app._active_session.app.renderer._last_screen
    return "\n".join(
        "".join(screen.data_buffer[row][column].char for column in range(120))
        for row in range(42)
    )


async def _terminal_text(app: TUIApp) -> str:
    """Read the characters the terminal last drew."""

    session = app._active_session
    await _wait_until(lambda: not session.app.invalidated, what="drawn screen")
    return _drawn_terminal_text(app)


async def _wait_for_screen(app: TUIApp, text: str, what: str) -> None:
    try:
        async with asyncio.timeout(30):
            while text not in await _terminal_text(app):
                # test-contract: latency real prompt-toolkit redraw
                await asyncio.sleep(0.05)
    except TimeoutError as exc:
        raise AssertionError(f"{what} is not on the terminal screen") from exc


def _screen_text(app: TUIApp) -> str:
    """Read the active widget through the terminal's Rich render path."""

    return Text.from_ansi(app._viewed_transcript().render(120)).plain


async def _send(pipe, data: str) -> None:
    pipe.send_bytes(data.encode())
    await asyncio.sleep(0)


async def _click_composer(app: TUIApp, pipe) -> None:
    """Click the composer's last row where the terminal last drew it."""

    session = app._active_session
    await _wait_until(
        lambda: not session.app.invalidated, what="drawn screen before composer click"
    )
    screen = session.app.renderer._last_screen
    row = next(
        position.ypos + position.height
        for window, position in screen.visible_windows_to_write_positions.items()
        if isinstance(window.content, BufferControl)
        and window.content.buffer is session.default_buffer
    )
    await _send(pipe, f"\x1b[<0;10;{row}M\x1b[<0;10;{row}m")


async def _enter_child(app: TUIApp, pipe, *, expect_loading: bool = False) -> None:
    navigation = app._agent_navigation
    await _wait_until(
        lambda: any(entry.path.name == CHILD_ID for entry in navigation.entries),
        what="child 651 in the real agent picker",
    )
    await _wait_until(
        lambda: not navigation.transcript_control.transition_pending(),
        what="root history transition idle",
    )
    if not navigation.list_focused():
        await _wait_until(
            lambda: bool(navigation.entries), what="visible agent picker rows"
        )
        if not app._active_session.layout.has_focus(app._active_session.default_buffer):
            await _click_composer(app, pipe)
            await _wait_until(
                lambda: app._active_session.layout.has_focus(
                    app._active_session.default_buffer
                ),
                what="visible composer before picker key",
            )
        await _wait_until(app.interaction_ready, what="ready TUI interaction")
        await _wait_until(
            app.composer_can_focus_agent_list,
            what="composer ready to open the visible agent picker",
        )
        await _send(pipe, "\x1b[B")
        await _wait_until(navigation.list_focused, what="agent picker focus")
    await navigation.wait_for_publication()
    await _wait_until(navigation.list_focused, what="published focused agent picker")
    while navigation.entries[navigation.selected_index].path.name != CHILD_ID:
        before = navigation.selected_index
        await _send(pipe, "\x1b[B")
        await _wait_until(
            lambda before=before: navigation.selected_index != before,
            what="next visible child picker row",
        )
    await _send(pipe, "\r")
    if expect_loading:
        # The 23 MiB child takes a while to open; the picker row says so.
        await _wait_for_screen(
            app, "loading… ", "child-loading: no loading state while the child opens"
        )
    await _wait_until(
        lambda: navigation.child_view_active
        and navigation.current_path.name == CHILD_ID
        and bool(navigation.transcript_control.transcript.units),
        what="focused child transcript",
    )
    await navigation.wait_for_publication()


async def _return_to_root(app: TUIApp, pipe) -> None:
    navigation = app._agent_navigation
    if not navigation.child_view_focused() and not navigation.list_focused():
        await _wait_until(
            lambda: bool(navigation.transcript_control.transcript.units),
            what="visible child transcript before transcript click",
        )
        await _send(pipe, "\x1b[<0;10;10M\x1b[<0;10;10m")
        await _wait_until(
            navigation.child_view_focused, what="visible child transcript focus"
        )
    await _wait_until(
        lambda: navigation.child_view_focused() or navigation.list_focused(),
        what="visible child navigation focus",
    )
    await navigation.wait_for_publication()
    await _send(pipe, "h")
    await _wait_until(
        lambda: not app._agent_navigation.child_view_active,
        what="root transcript after h",
    )
    await navigation.wait_for_publication()


async def _leave_via_picker(app: TUIApp, pipe) -> None:
    navigation = app._agent_navigation
    session = app._active_session
    if not navigation.list_focused():
        await _click_composer(app, pipe)
        await _wait_until(
            lambda: session.layout.has_focus(session.default_buffer)
            and app.composer_can_focus_agent_list(),
            what="child composer ready to open the agent picker",
        )
        await _send(pipe, "\x1b[B")
        await _wait_until(navigation.list_focused, what="agent picker from the child")
    while navigation.entries[navigation.selected_index].path != navigation.root_path:
        before = navigation.selected_index
        await _send(pipe, "\x1b[A")
        await _wait_until(
            lambda before=before: navigation.selected_index != before,
            what="previous picker row toward root",
        )
    await _send(pipe, "\r")
    await _wait_until(
        lambda: not navigation.child_view_active, what="root transcript after picker"
    )
    await navigation.wait_for_publication()


async def _child_rows(app: TUIApp) -> list[str]:
    """Read the rows the terminal drew for the focused child transcript."""

    await _terminal_text(app)
    screen = app._active_session.app.renderer._last_screen
    control = app._agent_navigation.transcript_control
    position = next(
        position
        for window, position in screen.visible_windows_to_write_positions.items()
        if window.content is control
    )
    return [
        "".join(screen.data_buffer[row][column].char for column in range(120)).rstrip()
        for row in range(position.ypos, position.ypos + position.height)
    ]


async def _scroll_to_separator(app: TUIApp, pipe) -> list[str]:
    """Scroll the child up line by line until a blank separator is the top row.

    Child 651 in the real home parks with a separator on top (ux r5 M7).
    """

    navigation = app._agent_navigation
    if not navigation.child_view_focused():
        await _send(pipe, "\x1b[<0;10;10M\x1b[<0;10;10m")
        await _wait_until(navigation.child_view_focused, what="child transcript focus")
    rows = await _child_rows(app)
    for _press in range(40):
        # The prompt draws its own blank first line; any other blank top row
        # is a separator.
        if not rows[0] and rows[1] and VISIBLE_PROMPT not in rows[1]:
            return rows
        before = rows
        await _send(pipe, "k")
        try:
            async with asyncio.timeout(30):
                while (rows := await _child_rows(app)) == before:
                    # test-contract: latency real prompt-toolkit redraw
                    await asyncio.sleep(0.05)
        except TimeoutError as exc:
            raise AssertionError("child-keeps-place: k did not scroll the child") from exc
    raise AssertionError("child-keeps-place: no separator row reached the top")


async def _assert_child_keeps_place(app: TUIApp, rows: list[str], how: str) -> None:
    navigation = app._agent_navigation
    await _wait_until(
        lambda: not navigation.transcript_control.transition_pending(),
        what="published child re-entry",
    )
    # A refresh tick after the commit must not move the restored place.
    ticked_at = navigation._last_refresh_at
    await _wait_until(
        lambda: navigation._last_refresh_at > ticked_at, what="child refresh tick"
    )
    await navigation.wait_for_publication()
    after = await _child_rows(app)
    assert after == rows, (
        f"child-keeps-place ({how}): re-entry drew other rows; "
        f"before={rows[:3]!r} after={after[:3]!r}"
    )
    assert re.search(r"line \d+/~?\d+", await _terminal_text(app)), (
        f"child-keeps-place ({how}): no line label after re-entry"
    )


@pytest.fixture(scope="module")
def child_view_fixture(tmp_path_factory: pytest.TempPathFactory) -> ChildViewHome:
    return generate_child_view_home(tmp_path_factory.mktemp("child-view-home"))


@pytest.mark.skipif(
    bool(os.environ.get("CI")) and sys.version_info < (3, 14),
    reason="real-TUI finder regression runs once in the Python 3.14 PR job",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["scroll-first", "restore-first", "switch-first"])
async def test_finder_return_respects_new_navigation(
    child_view_fixture: ChildViewHome, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    fixture = child_view_fixture
    store = await asyncio.to_thread(
        ConversationStore, fixture.home / "sessions", session_id=fixture.session_id
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex", model="offline", resumed=True, zeta_home=fixture.home,
    )
    output = Vt100_Output(StringIO(), lambda: Size(rows=42, columns=120), enable_cpr=False)
    release = asyncio.Event()
    started = asyncio.Event()
    finished = asyncio.Event()
    restore = app._return_to_view
    restore_task = None

    async def held_restore(follow_tail, anchor):
        nonlocal restore_task
        restore_task = asyncio.current_task()
        restore_task.add_done_callback(lambda _: finished.set())
        started.set()
        await await_or_fail(release.wait(), what="release finder return")
        await restore(follow_tail, anchor)

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        session = app._make_session()
        run_task = asyncio.create_task(app.run(session))

        async def next_frame(check, *, what, key=None):
            drawn = asyncio.Event()

            def after_render(_):
                if check():
                    drawn.set()

            session.app.after_render += after_render
            try:
                if key is None:
                    session.app.invalidate()
                else:
                    pipe.send_bytes(key)
                await await_or_fail(drawn.wait(), what=what)
            finally:
                session.app.after_render -= after_render

        try:
            await _wait_until(lambda: bool(app._transcript.units), what="root first paint")
            await _enter_child(app, pipe)
            navigation = app._agent_navigation
            control = navigation.transcript_control
            transcript = control.transcript
            child_path = navigation.current_path
            def anchor():
                return transcript.durable_scroll_anchor

            await next_frame(anchor, key=b"\x1b[5~", what="draw pre-finder PageUp")
            saved = anchor()
            pages = app._resident_page_keys()

            await next_frame(
                lambda: transcript.finder_active, key=b"\x06", what="draw open child finder"
            )
            pipe.send_bytes(b"visible assistant 95")
            await _wait_until(
                lambda: control.search_task is not None, what="child finder scan started"
            )
            await await_or_fail(control.search_task, timeout=90, what="child finder scan")
            await next_frame(
                lambda: app._resident_page_keys() != pages,
                what="draw child finder preview on different resident pages",
            )

            monkeypatch.setattr(app, "_return_to_view", held_restore)
            pipe.send_bytes(b"\x03")
            await await_or_fail(started.wait(), what="finder closed and return held")
            assert not transcript.finder_active

            if order == "switch-first":
                await _return_to_root(app, pipe)
                root_screen = await _terminal_text(app)
                root_pages = app._resident_page_keys()
                release.set()
                await await_or_fail(finished.wait(), what="old child restore completed")
                await next_frame(lambda: True, what="draw root after old child restore")
                assert app._resident_page_keys() == root_pages
                assert _drawn_terminal_text(app) == root_screen
                return

            if order == "restore-first":
                release.set()
                await await_or_fail(finished.wait(), what="finder return completed before PageUp")
                await next_frame(lambda: anchor() == saved, what="draw restored bookmark")

            before_scroll = anchor()
            before_pages = app._resident_page_keys()
            await next_frame(
                lambda: anchor() is not None and anchor() != before_scroll,
                key=b"\x1b[5~", what="draw latest PageUp",
            )
            scrolled = anchor()
            rows = (await _child_rows(app))[:3]
            assert app._resident_page_keys() == before_pages
            assert not transcript.at_resident_top

            release.set()
            await await_or_fail(finished.wait(), what="finder return settled after PageUp")
            await next_frame(lambda: True, what="draw child after finder return settled")
            assert anchor() == scrolled, "finder return overwrote a newer drawn PageUp"
            assert (await _child_rows(app))[:3] == rows

            await _return_to_root(app, pipe)
            parked = control._saved_view(child_path)
            assert parked is not None and parked.anchor == scrolled
            await _enter_child(app, pipe)
            await next_frame(lambda: True, what="draw re-entered child")
            assert anchor() == parked.anchor
            assert (await _child_rows(app))[:3] == rows
        finally:
            release.set()
            if restore_task is not None:
                await await_or_fail(finished.wait(), what="held finder return stopped")
                if not restore_task.cancelled():
                    restore_task.result()
            app.request_exit()
            await await_or_fail(run_task, what="real child-view TUI shutdown")


@pytest.mark.skipif(
    bool(os.environ.get("CI")) and sys.version_info < (3, 14),
    reason="real-TUI journey runs once in the Python 3.14 PR job",
)
@pytest.mark.asyncio
async def test_real_tui_child_view_journey(
    child_view_fixture: ChildViewHome,
) -> None:
    fixture = child_view_fixture
    store = await asyncio.to_thread(
        ConversationStore,
        fixture.home / "sessions",
        session_id=fixture.session_id,
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        resumed=True,
        zeta_home=fixture.home,
    )
    output_stream = StringIO()
    output = Vt100_Output(
        output_stream,
        lambda: Size(rows=42, columns=120),
        enable_cpr=False,
    )
    failures: list[str] = []
    journey_started = perf_counter()

    # CI 43788114830: prompt_toolkit postpones a redraw while input is ready,
    # so a PageUp can run before the first frame of the first page. Force that
    # order: hold the first page until frames are held, hold frames until the
    # key ran, and hold any older-page read it starts until the PageUp drew.
    source = app._history_source
    read_page = source.page
    first_page_gate = asyncio.Event()
    older_gate = asyncio.Event()
    older_reads: list[asyncio.Event] = []

    async def gated_page(cursor=None, *, direction="older"):
        if cursor is None:
            await await_or_fail(first_page_gate.wait(), what="held first root page")
            return await read_page(cursor, direction=direction)
        if direction != "older":
            return await read_page(cursor, direction=direction)
        finished = asyncio.Event()
        older_reads.append(finished)
        try:
            await await_or_fail(older_gate.wait(), what="held older root page")
            return await read_page(cursor, direction=direction)
        finally:
            finished.set()

    source.page = gated_page
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        session = app._make_session()
        redraw_events: list[str] = []
        invalidate = session.app.invalidate

        def record_invalidate() -> None:
            redraw_events.append(
                f"{sys._getframe(1).f_code.co_name}:"
                f"render={session.app.render_counter}:"
                f"pending={session.app.invalidated}"
            )
            del redraw_events[:-8]
            invalidate()

        session.app.invalidate = record_invalidate
        run_task = asyncio.create_task(app.run(session))
        try:
            await _wait_until(lambda: session.app.is_running, what="running TUIApp")
            await _wait_until(
                lambda: app._transcript._viewport_height > 1, what="laid-out root"
            )
            assert not app._transcript.units, "the first root page was not held"
            session.app._redraw = lambda *args, **kwargs: None
            try:
                first_page_gate.set()
                await _wait_until(
                    lambda: bool(app._transcript.units), what="root first paint"
                )
                assert app._transcript.follow_tail
                await _send(pipe, "\x1b[5~")
                await _wait_until(
                    lambda: not app._transcript.follow_tail,
                    what="root PageUp handled before the first frame",
                )
            finally:
                del session.app._redraw
                session.app.invalidate()
            try:
                await _wait_until(
                    lambda: app._transcript.durable_scroll_anchor is not None,
                    what="complex root anchor after PageUp",
                )
                root_anchor = app._transcript.durable_scroll_anchor
                pager = app._agent_navigation.transcript_control.history_pager
                resident = pager.descriptors
                older_gate.set()
                if older_reads:
                    await await_or_fail(older_reads[0].wait(), what="older root page read")
                    # The load adopts the page and repaints in one step.
                    await _wait_until(
                        lambda: pager.descriptors != resident,
                        what="older root page adopted",
                    )
                await _terminal_text(app)
                assert app._transcript.durable_scroll_anchor == root_anchor, (
                    "root-early-pageup: the page load started by a PageUp before "
                    "the first frame moved the root view; "
                    f"expected={root_anchor!r} "
                    f"actual={app._transcript.durable_scroll_anchor!r}"
                )
            except AssertionError as exc:
                failures.append(str(exc))
            finally:
                older_gate.set()
            root_anchor = app._transcript.durable_scroll_anchor
            before = Text.from_ansi(app._transcript.render(120)).plain
            assert ROOT_MARKER in before, "root anchor fixture marker was not rendered"

            await _enter_child(app, pipe, expect_loading=True)
            await _wait_until(
                app._agent_navigation.child_view_focused,
                what="visible focused child before finder key",
            )

            # ux r5 M7: mid-window in the newest page, park on a separator row,
            # leave with h, and come back to the same rows.
            try:
                await _send(pipe, "\x1b[5~")
                await _wait_until(
                    lambda: not app._agent_navigation.transcript_control.transcript.follow_tail,
                    what="child mid-window after PageUp",
                )
                rows = await _scroll_to_separator(app, pipe)
                await _return_to_root(app, pipe)
                await _enter_child(app, pipe)
                await _assert_child_keeps_place(app, rows, "mid-window, h")
            except AssertionError as exc:
                failures.append(str(exc))

            # ux b4 / review T7: page up a few pages, let a real refresh tick
            # run, and keep paging older without a jump to the newest window.
            navigation = app._agent_navigation
            control = navigation.transcript_control
            pager = control.history_pager

            def oldest_resident() -> str:
                return pager.pages[0].units[0].entry_id

            def child_place() -> tuple[int, str, object]:
                # The pager generation moves on every reset, so it proves no pager
                # work; the durable anchor is the top row the user sees.
                return (
                    pager.generation,
                    oldest_resident(),
                    control.transcript.durable_scroll_anchor,
                )

            async def page_up_until_older_loads() -> None:
                oldest = oldest_resident()
                for _press in range(400):
                    await _send(pipe, "\x1b[5~")
                    await navigation.wait_for_publication()
                    await _terminal_text(app)
                    if oldest_resident() != oldest:
                        return
                    if control.transcript.scroll_offset == 0:
                        break
                else:
                    raise AssertionError("child-paging: PageUp never reached the top")
                # At the top, PageUp asks for the older page, as a user presses it.
                await _send(pipe, "\x1b[5~")
                try:
                    await _wait_until(
                        lambda: oldest_resident() != oldest,
                        what="older child page after PageUp at the top",
                    )
                except AssertionError as exc:
                    raise AssertionError(
                        "child-paging: PageUp at the top stopped loading older rows "
                        f"(can_load_older={pager.can_load_older})"
                    ) from exc

            try:
                for _page in range(5):
                    await page_up_until_older_loads()
                assert pager.can_load_newer, (
                    "child-paging: after five older loads the newest page came back "
                    "(the scrolled view jumped back to the newest window)"
                )
                await _terminal_text(app)
                before_tick = child_place()
                ticked_at = navigation._last_refresh_at
                await _wait_until(
                    lambda: navigation._last_refresh_at > ticked_at,
                    what="a real child refresh tick",
                )
                await navigation.wait_for_publication()
                assert child_place() == before_tick, (
                    "child-paging: a refresh tick moved the scrolled child view "
                    f"before={before_tick!r} after={child_place()!r}"
                )
                await page_up_until_older_loads()
                assert pager.can_load_newer, (
                    "child-paging: paging older after a refresh tick jumped to the newest window"
                )
            except AssertionError as exc:
                failures.append(str(exc))

            # ux r5 M7: after older loads, park on a separator row, leave via the
            # picker, and come back to the same rows.
            try:
                rows = await _scroll_to_separator(app, pipe)
                await _leave_via_picker(app, pipe)
                await _enter_child(app, pipe)
                await _assert_child_keeps_place(app, rows, "older loads, picker")
            except AssertionError as exc:
                failures.append(str(exc))

            await _send(pipe, "\x06")
            await _wait_until(
                lambda: app._active_finder.finder_active,
                what="focused child finder",
            )
            await _send(pipe, "Henry")
            await _wait_until(
                lambda: app._agent_navigation.transcript_control.search_task is not None,
                what="focused child history scan",
            )
            search_task = app._agent_navigation.transcript_control.search_task
            assert search_task is not None
            # A full scan of the 23 MiB fixture child can pass 10 s on a loaded CI runner.
            await await_or_fail(search_task, timeout=90, what="focused child history scan")
            try:
                await _wait_until(
                    lambda: (
                        (state := app._agent_navigation.transcript_control.transcript.finder_state())
                        is not None
                        and state.complete
                    ),
                    what="terminal focused child finder state",
                )
                selected = app._agent_navigation.transcript_control.search_results[
                    app._agent_navigation.transcript_control.search_index
                ]
                await _wait_until(
                    lambda: VISIBLE_PROMPT in _screen_text(app),
                    what="selected child search result paint",
                )
                rendered = _screen_text(app)
                assert selected.entry_id.endswith("1"), (
                    "child-search: seq 1 is not the selected visible Henry match"
                )
                assert VISIBLE_PROMPT in rendered, (
                    "child-search: seq 1 prompt is not on the rendered terminal screen"
                )
                assert CHILD_PROMPT_END in rendered, (
                    "child-long-message: the end of a long child message was cut"
                )
            except (AssertionError, IndexError) as exc:
                failures.append(str(exc))

            await _wait_until(
                lambda: VISIBLE_PROMPT in _screen_text(app),
                what="visible search result before finder close key",
            )
            await _send(pipe, "\x03")
            await _wait_until(
                lambda: not app._active_finder.finder_active,
                what="finder close",
            )
            await _wait_until(
                lambda: not app._active_finder.finder_active,
                what="visible closed finder before escape key",
            )
            await _send(pipe, "\x1b")
            await _wait_until(
                lambda: not app._viewed_transcript().search_active,
                what="transcript search close",
            )
            await _click_composer(app, pipe)
            await _wait_until(
                lambda: session.layout.has_focus(session.default_buffer),
                what="child composer focus from terminal click",
            )
            await _wait_until(
                lambda: session.layout.has_focus(session.default_buffer),
                what="visible child composer before clear key",
            )
            await _send(pipe, "\x15")
            await _wait_until(
                lambda: not session.default_buffer.text,
                what="empty child composer",
            )
            await _wait_until(
                lambda: session.layout.has_focus(session.default_buffer)
                and not session.default_buffer.text,
                what="visible empty composer before paste",
            )
            await _send(pipe, "\x1b[200~draft-child-ux1309\x1b[201~")
            await _wait_until(
                lambda: session.default_buffer.text == "draft-child-ux1309",
                what="typed child draft",
            )
            if (
                session.app.editing_mode is EditingMode.VI
                and session.app.vi_state.input_mode is not InputMode.INSERT
            ):
                await _send(pipe, "i")
                await _wait_until(
                    lambda: session.app.vi_state.input_mode is InputMode.INSERT,
                    what="vi insert mode for prompt history",
                )
            await _wait_until(
                lambda: session.default_buffer.text == "draft-child-ux1309",
                what="visible child draft before history-up key",
            )
            await _send(pipe, "\x1b[A")
            try:
                await _wait_until(
                    lambda: session.default_buffer.text == CHILD_PROMPT,
                    what="old child prompt in composer",
                )
                assert session.default_buffer.text == CHILD_PROMPT, (
                    "child-prompt-history: Up did not restore the exact old child prompt"
                )
            except AssertionError as exc:
                failures.append(
                    f"{exc}; composer={session.default_buffer.text[:80]!r} "
                    f"length={len(session.default_buffer.text)}"
                )

            child_transcript = app._agent_navigation.transcript_control.transcript
            prior_child_anchor = child_transcript.durable_scroll_anchor
            await _wait_until(
                lambda: (
                    session.default_buffer.text == CHILD_PROMPT
                    and bool(child_transcript.units)
                ),
                what="visible selected child prompt and transcript before PageUp",
            )
            await _send(pipe, "\x1b[5~")
            await _wait_until(
                lambda: (
                    child_transcript.durable_scroll_anchor is not None
                    and child_transcript.durable_scroll_anchor != prior_child_anchor
                ),
                what="child anchor after PageUp",
            )
            child_anchor = child_transcript.durable_scroll_anchor
            assert child_anchor is not None

            await _return_to_root(app, pipe)
            try:
                await _wait_until(
                    lambda: bool(app._transcript.units),
                    what="restored root transcript",
                )
                after = Text.from_ansi(app._transcript.render(120)).plain
                assert app._transcript.durable_scroll_anchor == root_anchor, (
                    "root-anchor-return: durable complex-card anchor changed after child visit"
                )
                assert ROOT_MARKER in after, (
                    "root-anchor-return: complex root marker is not in the restored rendered widget"
                )
                anchor_before_reentry = child_transcript.durable_scroll_anchor
                await _enter_child(app, pipe)
                anchor_after_reentry = child_transcript.durable_scroll_anchor
                await _wait_until(
                    lambda: not app._agent_navigation.transcript_control.transition_pending(),
                    what="published child re-entry",
                )
                restored_anchor = child_transcript.durable_scroll_anchor
                assert restored_anchor == child_anchor, (
                    "child-return: durable anchor changed; "
                    f"before_pageup={prior_child_anchor!r} "
                    f"parked={child_anchor!r} "
                    f"before_reentry={anchor_before_reentry!r} "
                    f"after_reentry={anchor_after_reentry!r} "
                    f"at_check={restored_anchor!r}; "
                    f"redraws={redraw_events!r}; "
                    f"screen={_drawn_terminal_text(app)!r}"
                )
                assert not app._transcript.units, (
                    "child-return: the inactive root widget still holds rendered units"
                )
                assert len(
                    app._agent_navigation.transcript_control.history_pager.descriptors
                ) <= 4, "child-return: retained pages exceed the global four-page budget"
                assert session.default_buffer.text == CHILD_PROMPT, (
                    "child-return: selected prompt was not restored"
                )
                await _wait_until(
                    lambda: session.default_buffer.text == CHILD_PROMPT,
                    what="visible restored prompt before history-down key",
                )
                assert (
                    app._agent_navigation.transcript_control.prompt_history.navigating
                ), "child-return: prompt cursor was not restored"
                await _click_composer(app, pipe)
                await _wait_until(
                    lambda: session.layout.has_focus(session.default_buffer),
                    what="visible restored child composer before history-down key",
                )
                await _send(pipe, "\x1b[B")
                await _wait_until(
                    lambda: session.default_buffer.text == "draft-child-ux1309",
                    what="child draft restoration after re-entry",
                )
                assert session.default_buffer.text == "draft-child-ux1309", (
                    "child-return: parked draft was not restored after prompt navigation"
                )
            except AssertionError as exc:
                failures.append(str(exc))

            # ux b2: child ctrl+f finds the old seq-1 prompt and Enter shows it.
            try:
                await _click_composer(app, pipe)
                await _wait_until(
                    lambda: session.layout.has_focus(session.default_buffer),
                    what="child composer before finder key",
                )
                await _send(pipe, "\x06")
                await _wait_until(
                    lambda: app._active_finder.finder_active, what="child finder"
                )
                await _send(pipe, "Henry")
                owner = app._agent_navigation.transcript_control
                await _wait_until(
                    lambda: owner.search_task is not None, what="child history scan"
                )
                search_task = owner.search_task
                assert search_task is not None
                await await_or_fail(search_task, timeout=90, what="child history scan")
                await _wait_until(
                    lambda: (
                        (state := owner.transcript.finder_state()) is not None
                        and state.complete
                        and state.query == "Henry"
                    ),
                    what="terminal child finder state",
                )
                state = owner.transcript.finder_state()
                assert state is not None
                target = next(
                    (
                        index
                        for index, row in enumerate(state.rows)
                        if VISIBLE_PROMPT in row.candidate.text
                    ),
                    None,
                )
                assert target is not None, (
                    "child-search: the old seq-1 prompt has no finder row"
                )
                for _step in range(target):
                    await _send(pipe, "\x1b[B")
                await _wait_until(
                    lambda: (
                        (state := owner.transcript.finder_state()) is not None
                        and state.selected == target
                    ),
                    what="seq-1 finder row selected",
                )
                await _send(pipe, "\r")
                await _wait_until(
                    lambda: not app._active_finder.finder_active,
                    what="child finder closed after Enter",
                )
                await _wait_for_screen(
                    app, VISIBLE_PROMPT, "child-search: Enter on the old seq-1 match"
                )
                await _send(pipe, "\x1b")
            except AssertionError as exc:
                failures.append(str(exc))

            # ux b3 / m2: root ctrl+f reveals an old unloaded row, then n moves.
            await _return_to_root(app, pipe)
            try:
                await _send(pipe, "\x06")
                await _wait_until(
                    lambda: app._active_finder.finder_active, what="root finder"
                )
                await _send(pipe, ROOT_PAIR_MARKER)
                owner = app._agent_navigation.transcript_control
                await _wait_until(
                    lambda: owner.search_task is not None, what="root history scan"
                )
                search_task = owner.search_task
                assert search_task is not None
                await await_or_fail(search_task, timeout=90, what="root history scan")
                await _wait_until(
                    lambda: (
                        (state := app._transcript.finder_state()) is not None
                        and state.complete
                        and state.query == ROOT_PAIR_MARKER
                    ),
                    what="terminal root finder state",
                )
                await _send(pipe, "\r")
                await _wait_until(
                    lambda: not app._active_finder.finder_active,
                    what="root finder closed after Enter",
                )
                await _wait_for_screen(
                    app, "root row 20 ", "root-search: the newest old root match"
                )
                await _send(pipe, "n")
                await _wait_for_screen(
                    app, "root row 10 ", "root-search-next: n to the next old root match"
                )
            except AssertionError as exc:
                failures.append(str(exc))
        except AssertionError as exc:
            # A later step cannot run; keep the failures found before it.
            failures.append(f"journey stopped: {exc}")
        finally:
            first_page_gate.set()
            older_gate.set()
            vars(source).pop("page", None)
            app.request_exit()
            await await_or_fail(run_task, what="real child-view TUI shutdown")

    journey_seconds = perf_counter() - journey_started
    print(
        f"child-view fixture generation: {fixture.generation_seconds:.3f}s; "
        f"real TUI journey: {journey_seconds:.3f}s"
    )
    assert not failures, "real TUI child-view journey failures:\n- " + "\n- ".join(failures)


async def _top_rows_after(app: TUIApp) -> tuple[str, ...]:
    """Draw the child widget at the size of its window; return its top rows."""

    control = app._agent_navigation.transcript_control
    positions = app._active_session.app.renderer._last_screen.visible_windows_to_write_positions
    position = next(
        (position for window, position in positions.items() if window.content is control),
        None,
    )
    if position is None:
        rows = await _child_rows(app)
        return tuple(rows[:3])
    content = control.create_content(position.width, position.height)
    return tuple(
        "".join(fragment[1] for fragment in content.get_line(row)).rstrip()
        for row in range(3)
    )


@pytest.mark.skipif(
    bool(os.environ.get("CI")) and sys.version_info < (3, 14),
    reason="real-TUI sweep runs once in the Python 3.14 PR job",
)
@pytest.mark.asyncio
async def test_child_keeps_every_top_row_across_park_and_reentry(
    child_view_fixture: ChildViewHome,
) -> None:
    """ux r5 M7: for every top row from the last row before the tail up past a
    page boundary, and at the top of history, the parked anchor brings back the
    same rows: through show_pages with the source open at every row, and through
    the real re-entry transition at every 25th row."""

    from zeta.tui.agent_card import _anchored_pages

    fixture = child_view_fixture
    store = await asyncio.to_thread(
        ConversationStore, fixture.home / "sessions", session_id=fixture.session_id
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        resumed=True,
        zeta_home=fixture.home,
    )
    output = Vt100_Output(StringIO(), lambda: Size(rows=42, columns=120), enable_cpr=False)
    misses: list[str] = []
    rows_checked = 0
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        session = app._make_session()
        run_task = asyncio.create_task(app.run(session))
        try:
            await _wait_until(lambda: session.app.is_running, what="running TUIApp")
            await _wait_until(lambda: bool(app._transcript.units), what="root first paint")
            await _enter_child(app, pipe)
            navigation = app._agent_navigation
            control = navigation.transcript_control
            pager = control.history_pager
            child_path = navigation.current_path
            pages_by_entry: dict[str, tuple[object, ...]] = {}

            async def check(where: str, *, transition: bool) -> None:
                nonlocal rows_checked
                rows_checked += 1
                before = await _top_rows_after(app)
                anchor = control.transcript.durable_scroll_anchor
                if anchor is None:
                    misses.append(f"{where}: no durable anchor while scrolled, top={before!r}")
                    return
                if transition:
                    assert control._park_active_view()
                    control._start_transition(child_path, refresh=False)
                    await _wait_until(
                        lambda: not control.transition_pending(), what="child re-entry"
                    )
                    await navigation.wait_for_publication()
                else:
                    # The re-entry's page load: the anchor's page and the next one
                    # (one read per entry; the rows of an entry share it).
                    if anchor[0] not in pages_by_entry:
                        pages_by_entry.clear()
                        pages_by_entry[anchor[0]] = await _anchored_pages(
                            control._source, anchor
                        )
                    control.show_pages(pages_by_entry[anchor[0]], anchor)
                after = await _top_rows_after(app)
                if after != before:
                    misses.append(f"{where}: before={before!r} after={after!r}")

            async def sweep(*, transition: bool) -> None:
                control.bottom()
                await _top_rows_after(app)
                newest_page_start = pager.pages[-1].units[0].entry_id
                past_boundary = 0
                for step in range(3_000):
                    control.scroll(-1)
                    if not transition or step % 25 == 0:
                        await check(f"line {step + 1} above the tail", transition=transition)
                        if misses:
                            return
                    anchor = control.transcript.durable_scroll_anchor
                    # Fixture ids sort by seq: go one screen past the newest page.
                    if anchor is not None and anchor[0] < newest_page_start:
                        past_boundary += 1
                        if past_boundary == 40:
                            return
                raise AssertionError("child-sweep: the sweep never crossed a page boundary")

            # Every 25th row through the real re-entry transition, then every row
            # through the page load and paint that the transition ends with.
            await sweep(transition=True)
            assert not misses, (
                f"child-sweep (transition): {len(misses)} of {rows_checked} top rows "
                "moved:\n- " + "\n- ".join(misses[:8])
            )
            await sweep(transition=False)

            for _load in range(400):
                if not pager.can_load_older:
                    break
                oldest = pager.pages[0].units[0].entry_id
                control.top()
                await _top_rows_after(app)
                # test-contract: latency the edge load runs on the history worker
                await asyncio.sleep(0.05)
                await navigation.wait_for_publication()
                if pager.pages[0].units[0].entry_id == oldest:
                    control.scroll(-1)
            assert not pager.can_load_older, "child-sweep: never reached the top of history"
            control.top()
            for step in range(4):
                await check(f"line {step + 1} of history", transition=step == 0)
                control.scroll(1)
        finally:
            app.request_exit()
            await await_or_fail(run_task, what="real child-view TUI shutdown")
    print(f"child-sweep: {rows_checked} top rows checked")
    assert not misses, (
        f"child-sweep: {len(misses)} of {rows_checked} top rows moved:\n- "
        + "\n- ".join(misses[:8])
    )


@pytest.mark.asyncio
async def test_child_source_validates_its_log_once_across_reads(
    child_view_fixture: ChildViewHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a child store validates the whole log; each read only refreshes it."""

    from zeta.tui.transcript import child_history_source

    opened: list[ConversationStore] = []

    class CountingStore(ConversationStore):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            opened.append(self)

    # test-contract: seam count store opens behind the source's page reads
    monkeypatch.setattr(child_history_source, "ConversationStore", CountingStore)
    source = child_history_source.ChildLogHistorySource(child_view_fixture.child_path)
    try:
        newest = await source.page()
        older = await source.page(newest.older)
        await source.reveal(older.segments[0].units[0].entry_id)
        await source.search("child", limit=1)
        assert len(opened) == 1, (
            f"child source opened {len(opened)} stores for 4 reads of an unchanged log"
        )
    finally:
        await source.close()
    assert all(store.closed for store in opened), "child source left its store open"


@pytest.mark.asyncio
async def test_child_source_closes_a_store_from_a_read_whose_caller_left(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second cancel lets a read's caller leave while its worker thread still
    holds a store. A later read keeps its own store; when the late thread ends it
    must close its store, not replace the kept one (2 leaked descriptors each)."""

    import threading

    from zeta.protocol.types import Message, MessageRole, TextContent
    from zeta.tui.transcript import child_history_source

    writer = ConversationStore(tmp_path / "agents", session_id="1")
    writer.append_message(Message(MessageRole.ASSISTANT, [TextContent("child reply")]))
    path = writer.session_dir
    writer.close()

    opened: list[ConversationStore] = []

    class CountingStore(ConversationStore):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            opened.append(self)

    # test-contract: seam count store opens behind the source's reads
    monkeypatch.setattr(child_history_source, "ConversationStore", CountingStore)
    source = child_history_source.ChildLogHistorySource(path)
    started, release, late_done = threading.Event(), threading.Event(), threading.Event()
    run_sync = source._run_sync

    def tracked_run_sync(operation, *args):
        try:
            return run_sync(operation, *args)
        finally:
            if operation is held_read:
                late_done.set()

    def held_read(store: ConversationStore) -> None:
        started.set()
        release.wait(10)

    # test-contract: seam mark when the held read's worker thread returns
    monkeypatch.setattr(source, "_run_sync", tracked_run_sync)
    late = asyncio.create_task(source._run(held_read))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 10), 10)
        late.cancel()
        # test-contract: one loop turn so the first cancel reaches the drain
        await asyncio.sleep(0)
        late.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(late, 10)
        await asyncio.wait_for(source.page(), 10)
        release.set()
        assert await asyncio.wait_for(asyncio.to_thread(late_done.wait, 10), 10)
    finally:
        release.set()
        await source.close()
    assert len(opened) == 2, f"expected the late read and the next read to open stores: {opened}"
    leaked = [store for store in opened if not store.closed]
    assert not leaked, f"{len(leaked)} of 2 child stores left open after close"


def test_bounded_text_scans_each_character_a_bounded_number_of_times() -> None:
    """A many-line tool result is split in one pass, not one scan per line."""

    from zeta.tui.agent_view_state import _bounded_text

    scanned = 0

    class CountingText(str):
        def find(self, sub: str, start: int = 0, *rest: int) -> int:  # type: ignore[override]
            nonlocal scanned
            hit = str.find(self, sub, start, *rest)
            scanned += (hit + 1 if hit >= 0 else len(self)) - start
            return hit

    text = CountingText(("\n" + "h" * 99) * 2_000)
    assert _bounded_text(text) == _bounded_text(str(text))
    assert scanned <= 3 * len(text), (
        f"split scanned {scanned} characters of a {len(text)}-character text"
    )


@pytest.mark.parametrize(
    ("text", "lines"),
    [
        ("", [""]),
        ("a\r\nb", ["a", "b"]),
        ("a\rb\nc", ["a", "b", "c"]),
        ("a\n\rb", ["a", "", "b"]),
        ("a\r", ["a"]),
        ("\r\n\r\n", ["", ""]),
        ("a\n\nb\n", ["a", "", "b"]),
    ],
)
def test_text_lines_splits_on_cr_lf_and_crlf(text: str, lines: list[str]) -> None:
    from zeta.tui.agent_view_state import _text_lines

    assert list(_text_lines(text)) == lines
