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


async def _terminal_text(app: TUIApp) -> str:
    """Read the characters the terminal last drew."""

    session = app._active_session
    await _wait_until(lambda: not session.app.invalidated, what="drawn screen")
    screen = session.app.renderer._last_screen
    return "\n".join(
        "".join(screen.data_buffer[row][column].char for column in range(120))
        for row in range(42)
    )


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


async def _enter_child(app: TUIApp, pipe) -> None:
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

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        session = app._make_session()
        run_task = asyncio.create_task(app.run(session))
        try:
            await _wait_until(lambda: session.app.is_running, what="running TUIApp")
            await _wait_until(
                lambda: bool(app._transcript.units), what="root first paint"
            )

            initial_anchor = app._transcript.durable_scroll_anchor
            await _send(pipe, "\x1b[5~")
            await _wait_until(
                lambda: (
                    app._transcript.durable_scroll_anchor is not None
                    and app._transcript.durable_scroll_anchor != initial_anchor
                ),
                what="complex root anchor after PageUp",
            )
            root_anchor = app._transcript.durable_scroll_anchor
            before = Text.from_ansi(app._transcript.render(120)).plain
            assert ROOT_MARKER in before, "root anchor fixture marker was not rendered"

            await _enter_child(app, pipe)
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
                await _enter_child(app, pipe)
                await _wait_until(
                    lambda: not app._agent_navigation.transcript_control.transition_pending(),
                    what="published child re-entry",
                )
                restored_anchor = (
                    app._agent_navigation.transcript_control.transcript.durable_scroll_anchor
                )
                assert restored_anchor == child_anchor, (
                    "child-return: durable anchor changed; "
                    f"expected={child_anchor!r} actual={restored_anchor!r}"
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
            app.request_exit()
            await await_or_fail(run_task, what="real child-view TUI shutdown")

    journey_seconds = perf_counter() - journey_started
    print(
        f"child-view fixture generation: {fixture.generation_seconds:.3f}s; "
        f"real TUI journey: {journey_seconds:.3f}s"
    )
    assert not failures, "real TUI child-view journey failures:\n- " + "\n- ".join(failures)
