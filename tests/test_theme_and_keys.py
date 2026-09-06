"""Tests for ZETA-81 keybinding remap + theme selection."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

import pytest
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.text import Text

from zeta.core.session import SessionError
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui import theme as theme_module
from zeta.tui.app import TUIApp
from zeta.tui.bootstrap import _apply_startup_theme, _validate_keybindings
from zeta.tui.fake_backend import FakeInteractiveBackend
from zeta.tui.key_bindings import (
    ACTIONS,
    DEFAULTS,
    KeybindingError,
    build_key_bindings,
    parse_key_spec,
    resolve_keybindings,
)


@pytest.fixture(autouse=True)
def _restore_theme() -> Generator[None, None, None]:
    """Reset the theme to the built-in dark palette after every test."""

    yield
    theme_module.set_active_palette(theme_module.DARK)


# --- keybinding remap ------------------------------------------------------


def test_defaults_cover_every_advertised_action() -> None:
    assert set(DEFAULTS) == set(ACTIONS)


def test_parse_key_spec_accepts_shorthand_and_chords() -> None:
    assert parse_key_spec("c-r") == ("c-r",)
    assert parse_key_spec("Ctrl-R") == ("c-r",)
    assert parse_key_spec("ctrl+r") == ("c-r",)
    assert parse_key_spec("shift-tab") == ("s-tab",)
    assert parse_key_spec("c-x c-o") == ("c-x", "c-o")
    assert parse_key_spec("pageup") == ("pageup",)


def test_parse_key_spec_rejects_unknown_keys_loudly() -> None:
    with pytest.raises(KeybindingError, match="unknown key"):
        parse_key_spec("ctrl-nope")
    with pytest.raises(KeybindingError, match="empty key spec"):
        parse_key_spec("")
    with pytest.raises(KeybindingError, match="invalid key spec"):
        parse_key_spec("c-")


def test_resolve_keybindings_round_trips_user_remaps() -> None:
    resolved = resolve_keybindings({"retry": "ctrl-r", "toggle-agent": "f5"})
    assert resolved["retry"] == ("c-r",)
    assert resolved["toggle-agent"] == ("f5",)
    # Untouched actions keep their defaults.
    assert resolved["interrupt"] == DEFAULTS["interrupt"]


def test_resolve_keybindings_rejects_unknown_action() -> None:
    with pytest.raises(KeybindingError, match="unknown action"):
        resolve_keybindings({"typo-action": "c-r"})


def test_resolve_keybindings_reports_bad_key_with_action() -> None:
    with pytest.raises(KeybindingError, match="'retry' = 'nope'"):
        resolve_keybindings({"retry": "nope"})


def _binding_keys(bindings: KeyBindings, callback_name: str) -> set[tuple[str, ...]]:
    return {
        tuple(str(k.value) if hasattr(k, "value") else str(k) for k in binding.keys)
        for binding in bindings.bindings
        if binding.handler.__name__ == callback_name
    }


def test_build_key_bindings_applies_remap_over_defaults() -> None:
    calls: list[str] = []
    bindings = build_key_bindings(
        on_interrupt=lambda: calls.append("interrupt"),
        on_exit=lambda: calls.append("exit"),
        on_retry=lambda: calls.append("retry"),
        on_toggle_agent=lambda: calls.append("toggle-agent"),
        key_remap={"retry": "f5", "toggle-agent": "c-t"},
    )
    assert _binding_keys(bindings, "retry") == {("f5",)}
    assert _binding_keys(bindings, "toggle_agent") == {("c-t",)}


def test_build_key_bindings_loud_fails_on_bad_remap() -> None:
    with pytest.raises(KeybindingError):
        build_key_bindings(
            on_interrupt=lambda: None,
            on_exit=lambda: None,
            key_remap={"retry": "not-a-key"},
        )


def test_build_key_bindings_composes_with_vim_mode() -> None:
    """Remaps live on top of vim; the native escape binding stays intact."""

    bindings = build_key_bindings(
        on_interrupt=lambda: None,
        on_exit=lambda: None,
        on_retry=lambda: None,
        key_remap={"retry": "c-r"},
    )
    escape_bindings = [
        binding for binding in bindings.bindings if binding.keys == (Keys.Escape,)
    ]
    # native vi escape stays in the map alongside the full-screen escape handler
    assert len(escape_bindings) >= 2
    # remap took hold
    assert _binding_keys(bindings, "retry") == {("c-r",)}


def test_validate_keybindings_translates_loud_error_into_session_error() -> None:
    with pytest.raises(SessionError, match="unknown action"):
        _validate_keybindings({"bogus": "c-r"})


# --- theme selection -------------------------------------------------------


def test_active_palette_defaults_to_dark() -> None:
    assert theme_module.active_palette().name == "dark"
    assert theme_module.BODY == theme_module.DARK.body
    assert theme_module.CODE_THEME == theme_module.DARK.code_theme


def test_set_active_palette_updates_module_constants() -> None:
    theme_module.set_active_palette(theme_module.LIGHT)
    assert theme_module.BODY == theme_module.LIGHT.body
    assert theme_module.ACCENT == theme_module.LIGHT.accent
    assert theme_module.CODE_THEME == theme_module.LIGHT.code_theme
    # Rich theme is mutated in place, so a Console built earlier picks up
    # the new palette on the next render.
    assert theme_module.RICH_THEME.styles["markdown.paragraph"].color is not None


def test_light_palette_is_readable_under_forced_terminal() -> None:
    """A forced-terminal render of the composer + status bar must use dark
    ink so a light terminal can print visibly."""

    theme_module.set_active_palette(theme_module.LIGHT)
    console = Console(
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=80,
    )
    with console.capture() as capture:
        console.print(Text("hello world", style=theme_module.BODY))
    output = capture.get()
    # Body ink is a deep near-black in the light palette; the ANSI must
    # request that specific color.
    assert "\x1b[38;2;28;30;26m" in output
    # Accent and card border likewise picked up light-palette colors.
    assert theme_module.CARD_BORDER == theme_module.LIGHT.card_border
    assert theme_module.SEARCH_MATCH.endswith(theme_module.LIGHT.search_bg)


def test_apply_startup_theme_falls_back_when_name_unknown(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    home.mkdir()
    notices = _apply_startup_theme("nonexistent", home)
    assert theme_module.active_palette().name == "dark"
    assert notices and "unknown" in notices[0]


def test_user_theme_file_overrides_built_in(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    themes = home / "themes"
    themes.mkdir(parents=True)
    (themes / "dark.toml").write_text(
        dedent(
            """
            accent = "#00ff00"
            body = "#101010"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    palette, notice = theme_module.resolve_palette("dark", home=home)
    assert notice is None
    assert palette is not None
    assert palette.accent == "#00ff00"
    assert palette.body == "#101010"
    # Missing keys inherit the built-in dark defaults.
    assert palette.dim == theme_module.DARK.dim


def test_malformed_theme_file_falls_open_with_notice(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    themes = home / "themes"
    themes.mkdir(parents=True)
    (themes / "custom.toml").write_text("not = valid = toml", encoding="utf-8")
    palette, notice = theme_module.resolve_palette("custom", home=home)
    assert palette is None
    assert notice is not None and "ignored" in notice


def test_malformed_theme_file_at_startup_uses_dark(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    (home / "themes").mkdir(parents=True)
    (home / "themes" / "custom.toml").write_text("bad = = toml", encoding="utf-8")
    notices = _apply_startup_theme("custom", home)
    assert theme_module.active_palette().name == "dark"
    assert notices and "ignored" in notices[0]


def test_theme_file_with_non_string_value_falls_open(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    (home / "themes").mkdir(parents=True)
    (home / "themes" / "custom.toml").write_text('accent = 42\n', encoding="utf-8")
    palette, notice = theme_module.resolve_palette("custom", home=home)
    assert palette is None
    assert notice is not None and "accent" in notice


def test_list_available_themes_merges_built_ins_and_user_files(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    (home / "themes").mkdir(parents=True)
    (home / "themes" / "solarized.toml").write_text(
        'accent = "#268bd2"\n', encoding="utf-8"
    )
    names = theme_module.list_available(home)
    assert set(names) == {"dark", "light", "solarized"}


def test_slash_theme_switches_and_rebuilds(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    app = TUIApp(
        AgentLoop(
            FakeInteractiveBackend(model="offline"),
            ConversationStore(session_dir),
            skip_mcp_mount=True,
        ),
        provider="fake",
        model="offline",
        zeta_home=tmp_path / "zeta-home",
    )
    rebuilt = [0]

    def rebuild() -> None:
        rebuilt[0] += 1

    app._rebuild_transcript = rebuild  # type: ignore[method-assign]
    app._invalidate_prompt = lambda: None  # type: ignore[method-assign]

    result = app.slash_theme("light")

    assert result == "theme: light"
    assert theme_module.active_palette().name == "light"
    assert rebuilt[0] == 1


def test_slash_theme_lists_available(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    home = tmp_path / "zeta-home"
    home.mkdir()
    app = TUIApp(
        AgentLoop(
            FakeInteractiveBackend(model="offline"),
            ConversationStore(session_dir),
            skip_mcp_mount=True,
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
    )
    assert "theme: dark" in app.slash_theme("")
    assert "dark" in app.slash_theme("list")
    assert "light" in app.slash_theme("list")


def test_slash_theme_unknown_name_is_not_applied(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    home = tmp_path / "zeta-home"
    home.mkdir()
    app = TUIApp(
        AgentLoop(
            FakeInteractiveBackend(model="offline"),
            ConversationStore(session_dir),
            skip_mcp_mount=True,
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
    )
    original = theme_module.active_palette().name
    result = app.slash_theme("nope")
    assert result.startswith("theme unchanged")
    assert theme_module.active_palette().name == original
