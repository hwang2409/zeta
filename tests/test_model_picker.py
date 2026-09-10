"""``/model`` picker: catalog union, matching, card navigation, keys, completion."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.panel import Panel

from zeta.core.fake import FakeBackend
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.model_catalog import PROVIDER_MODELS
from zeta.tui.app import TUIApp
from zeta.tui.composer import SlashCompleter, build_key_bindings
from zeta.tui.models import known_models, match_models
from zeta.tui.slash_handlers.model_picker import MAX_VISIBLE_ROWS, ModelPicker

OPUS_MODELS = (
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
)


async def wait_until(check: Callable[[], bool]) -> None:
    for _ in range(300):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


def _plain(panel: Panel) -> str:
    output = StringIO()
    Console(file=output, width=100, force_terminal=False, no_color=True).print(panel)
    return output.getvalue()


def _app(
    tmp_path: Path,
    *,
    provider: str = "claude",
    model: str = "claude-sonnet-4-6",
    loader: Callable[[str], frozenset[str] | None] | None = None,
    full_screen: bool = False,
) -> tuple[TUIApp, StringIO]:
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider=provider,
        model=model,
        console=Console(file=output, force_terminal=False, width=100),
        history_path=tmp_path / "history",
        model_catalog_loader=loader or (lambda provider: None),
    )
    if full_screen:
        app._active_session = app._make_session()
    return app, output


def test_known_models_merges_static_table_and_live_catalog() -> None:
    merged = known_models("claude", frozenset({"claude-opus-4-1-20250805"}))

    assert merged == tuple(sorted({*PROVIDER_MODELS["claude"], "claude-opus-4-1-20250805"}))
    assert known_models("codex") == tuple(sorted(PROVIDER_MODELS["codex"]))
    assert known_models("fake") == ("faster", "offline")


def test_match_models_is_a_case_insensitive_substring_filter() -> None:
    assert match_models("OPUS", known_models("claude")) == OPUS_MODELS
    assert match_models(" sol ", known_models("codex")) == ("gpt-5.6-sol",)
    assert match_models("nope", known_models("claude")) == ()


def test_model_without_arguments_opens_a_picker_on_the_current_model(
    tmp_path: Path,
) -> None:
    app, output = _app(tmp_path)

    assert create_slash_registry().dispatch(app, "/model") == ""

    picker = app._model_picker
    assert picker is not None
    assert picker.choices == known_models("claude")
    assert picker.selected == "claude-sonnet-4-6"
    assert picker.query == ""
    assert app.model == "claude-sonnet-4-6"
    rendered = output.getvalue()
    assert "select a model · claude" in rendered
    assert "❯ claude-sonnet-4-6" in rendered
    assert "current" in rendered
    assert "1M" in rendered
    assert "enter select · esc cancel" in rendered


def test_model_substring_with_several_matches_narrows_the_picker(tmp_path: Path) -> None:
    app, output = _app(tmp_path)

    assert create_slash_registry().dispatch(app, "/model opus") == ""

    picker = app._model_picker
    assert picker is not None
    assert picker.choices == OPUS_MODELS
    assert picker.query == "opus"
    assert picker.selected == "claude-opus-4-5"
    assert app.model == "claude-sonnet-4-6"
    assert 'matching "opus"' in output.getvalue()


def test_model_unique_substring_switches_directly(tmp_path: Path) -> None:
    app, _ = _app(tmp_path, provider="fake", model="offline")

    assert create_slash_registry().dispatch(app, "/model fast") == "model: faster"
    assert app.model == "faster"
    assert app._model_picker is None


def test_model_exact_name_bypasses_the_picker(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)

    output = create_slash_registry().dispatch(app, "/model claude-opus-4-8")

    assert output.startswith("model: claude-opus-4-8")
    assert app.model == "claude-opus-4-8"
    assert app._model_picker is None


def test_model_name_matching_nothing_is_still_used_as_typed(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)

    output = create_slash_registry().dispatch(app, "/model claude-brand-new")

    assert output == (
        "model: claude-brand-new (model catalog unavailable for claude — using anyway)"
    )
    assert app.model == "claude-brand-new"
    assert app._model_picker is None


@pytest.mark.asyncio
async def test_picker_does_not_open_while_a_turn_is_active(tmp_path: Path) -> None:
    app, _ = _app(tmp_path, provider="fake", model="offline")
    app._active_task = asyncio.create_task(asyncio.sleep(1))

    try:
        output = create_slash_registry().dispatch(app, "/model")
    finally:
        app._active_task.cancel()
        await asyncio.gather(app._active_task, return_exceptions=True)

    assert output == (
        "model: offline (cannot change model while a turn or approval is active)"
    )
    assert app._model_picker is None


@pytest.mark.asyncio
async def test_picker_navigation_wraps_and_select_switches_the_model(
    tmp_path: Path,
) -> None:
    app, output = _app(tmp_path, provider="fake", model="offline")
    create_slash_registry().dispatch(app, "/model")
    picker = app._model_picker
    assert picker is not None and picker.selected == "offline"

    app.model_picker_move(1)
    assert picker.selected == "faster"
    app.model_picker_move(1)
    assert picker.selected == "offline"
    app.model_picker_move(-1)
    assert picker.selected == "faster"

    app.model_picker_select()
    assert app._model_picker is None
    try:
        await wait_until(lambda: app.model == "faster")
        await wait_until(lambda: "model: faster" in output.getvalue())
    finally:
        await app.loop.close()


def test_picker_cancel_removes_the_card_and_its_spacer_in_full_screen(
    tmp_path: Path,
) -> None:
    app, _ = _app(tmp_path, full_screen=True)
    app._print_system("hello")
    before = len(app._transcript._units)

    create_slash_registry().dispatch(app, "/model")

    assert len(app._transcript._units) == before + 2
    assert app._model_picker_unit is app._transcript._units[-1]
    assert "select a model" in _plain(app._model_picker_unit.value)

    app.model_picker_cancel()

    assert app._model_picker is None
    assert app._model_picker_unit is None
    assert len(app._transcript._units) == before
    app.model_picker_cancel()  # idempotent
    assert len(app._transcript._units) == before


def test_picker_moves_repaint_the_card_in_full_screen(tmp_path: Path) -> None:
    app, _ = _app(tmp_path, provider="fake", model="offline", full_screen=True)
    create_slash_registry().dispatch(app, "/model")
    unit = app._model_picker_unit
    assert "❯ offline" in _plain(unit.value)

    app.model_picker_move(1)

    assert app._model_picker_unit is unit
    assert "❯ faster" in _plain(unit.value)


@pytest.mark.asyncio
async def test_open_picker_folds_in_the_catalog_when_it_arrives(tmp_path: Path) -> None:
    dated = "claude-opus-4-1-20250805"
    app, _ = _app(tmp_path, loader=lambda provider: frozenset({dated}))

    create_slash_registry().dispatch(app, "/model opus")
    picker = app._model_picker
    assert picker is not None and dated not in picker.choices

    await wait_until(lambda: app._model_catalog_loaded)

    refreshed = app._model_picker
    assert refreshed is not None
    assert refreshed.choices == (dated, *OPUS_MODELS)
    assert refreshed.selected == picker.selected == "claude-opus-4-5"
    assert refreshed.query == "opus"


@pytest.mark.asyncio
async def test_any_other_submission_dismisses_an_open_picker(tmp_path: Path) -> None:
    app, output = _app(tmp_path, provider="fake", model="offline")
    create_slash_registry().dispatch(app, "/model")
    assert app._model_picker is not None

    app._submit_input("/name")

    assert app._model_picker is None
    try:
        await wait_until(lambda: "session name: (unnamed)" in output.getvalue())
    finally:
        await app.loop.close()


def test_picker_render_windows_long_catalogs_around_the_selection() -> None:
    choices = tuple(f"gpt-{index:02d}" for index in range(30))
    picker = ModelPicker("codex", choices, current="gpt-20")

    assert picker.index == 20
    rendered = _plain(picker.render())
    assert "… 14 more above" in rendered
    assert "… 4 more below" in rendered
    assert "❯ gpt-20" in rendered
    assert "gpt-14" in rendered
    assert "gpt-25" in rendered
    assert "gpt-13" not in rendered
    assert "gpt-26" not in rendered
    assert rendered.count("gpt-") == MAX_VISIBLE_ROWS

    picker.index = 0
    assert "more above" not in _plain(picker.render())


def test_picker_with_choices_keeps_the_highlighted_model() -> None:
    picker = ModelPicker("fake", ("a", "b", "c"), current="zzz")
    picker.move(2)
    refreshed = picker.with_choices(("a", "b", "b2", "c"))

    assert refreshed.selected == "c"
    assert refreshed.current == "zzz"
    with pytest.raises(ValueError, match="at least one choice"):
        ModelPicker("fake", (), current="a")


@pytest.mark.asyncio
async def test_picker_keys_only_fire_on_an_empty_composer() -> None:
    active = True
    moves: list[int] = []
    actions: list[str] = []

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_picker_move=moves.append,
                on_picker_select=lambda: actions.append("select"),
                on_picker_cancel=lambda: actions.append("cancel"),
                picker_active=lambda: active,
            ),
            multiline=True,
        )
        session.app.ttimeoutlen = 0.02
        session.app.timeoutlen = 0.02
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)

        pipe.send_text("\x1b[B\x1b[A")
        await wait_until(lambda: moves == [1, -1])

        pipe.send_text("typed")
        await wait_until(lambda: session.default_buffer.text == "typed")
        pipe.send_text("\x1b[B\x1b[A")
        await asyncio.sleep(0.1)
        assert moves == [1, -1]
        assert actions == []

        session.default_buffer.reset()
        pipe.send_text("\r")
        await wait_until(lambda: actions == ["select"])
        assert session.default_buffer.text == ""

        pipe.send_text("\x1b")
        await wait_until(lambda: actions == ["select", "cancel"])

        active = False
        pipe.send_text("\x1b[B")
        await asyncio.sleep(0.1)
        assert moves == [1, -1]

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_slash_completer_offers_model_names_after_model() -> None:
    registry = create_slash_registry()
    completer = SlashCompleter(
        registry,
        model_choices=lambda: ("claude-opus-4-8", "claude-sonnet-4-6"),
        current_model=lambda: "claude-sonnet-4-6",
    )
    event = CompleteEvent(completion_requested=True)

    partial = list(completer.get_completions(Document("/model op"), event))
    assert [item.text for item in partial] == ["claude-opus-4-8"]
    assert partial[0].start_position == -2

    everything = list(completer.get_completions(Document("/model "), event))
    assert [item.text for item in everything] == ["claude-opus-4-8", "claude-sonnet-4-6"]
    assert [item.display_meta_text for item in everything] == ["", "current"]

    assert list(completer.get_completions(Document("/model a b"), event)) == []
    assert list(completer.get_completions(Document("/theme d"), event)) == []
    assert list(completer.get_completions(Document("/model\nop"), event)) == []
    commands = list(completer.get_completions(Document("/mod"), event))
    assert [item.text for item in commands] == ["model"]

    plain = SlashCompleter(registry)
    assert list(plain.get_completions(Document("/model op"), event)) == []
