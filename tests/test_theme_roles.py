"""Semantic role-colour behaviour for the gruvbox theme work.

These guard two promises:

- tool call/result cards render byte-identically to ``origin/main`` (5d8adf6e),
  captured in ``tests/fixtures/tool_cards_gruvbox.json``;
- the non-tool-card surfaces that used to overuse the reddish/accent colour now
  pull from named semantic roles, and red stays reserved for real failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.panel import Panel
from rich.text import Text

from tests.tool_card_snapshots import representative_cards
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.tui import theme
from zeta.tui.cards.approval_card import render_approval_card
from zeta.tui.render import render_event
from zeta.tui.todo import TodoWidget

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_cards_gruvbox.json"

_ROLE_FIELDS = (
    "agent_main",
    "agent_child",
    "notice",
    "warning",
    "success",
)


@pytest.fixture(autouse=True)
def _restore_palette():
    previous = theme.active_palette()
    yield
    theme.set_active_palette(previous)


def test_every_builtin_palette_defines_every_role() -> None:
    for palette in theme.BUILT_IN_PALETTES.values():
        for field in _ROLE_FIELDS:
            value = getattr(palette, field)
            assert isinstance(value, str) and value, (
                f"{palette.name} palette leaves {field} empty"
            )


def test_set_active_palette_exposes_role_constants() -> None:
    theme.set_active_palette(theme.GRUVBOX_DARK)
    assert theme.AGENT_MAIN == theme.GRUVBOX_DARK.agent_main
    assert theme.AGENT_CHILD == theme.GRUVBOX_DARK.agent_child
    assert theme.NOTICE == theme.GRUVBOX_DARK.notice
    assert theme.WARNING == theme.GRUVBOX_DARK.warning
    assert theme.SUCCESS == theme.GRUVBOX_DARK.success
    # Identity and severity read as separate colours, not the shared accent.
    assert theme.AGENT_MAIN != theme.ACCENT
    assert theme.AGENT_CHILD != theme.AGENT_MAIN
    assert theme.WARNING not in {theme.ERROR, theme.ACCENT}


def test_partial_user_override_keeps_builtin_role_colours(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    themes = home / "themes"
    themes.mkdir(parents=True)
    (themes / "gruvbox-dark.toml").write_text('accent = "#ffffff"\n')

    palette, notice = theme.resolve_palette("gruvbox-dark", home=home)

    assert notice is None
    assert palette is not None
    assert palette.accent == "#ffffff"
    # Roles the file omits fall back to the same-named built-in, not the
    # generic dark defaults.
    assert palette.agent_child == theme.GRUVBOX_DARK.agent_child
    assert palette.warning == theme.GRUVBOX_DARK.warning
    assert palette.notice == theme.GRUVBOX_DARK.notice


def test_user_role_override_is_accepted(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    themes = home / "themes"
    themes.mkdir(parents=True)
    (themes / "gruvbox-dark.toml").write_text('agent_child = "#00ff00"\n')

    palette, notice = theme.resolve_palette("gruvbox-dark", home=home)

    assert notice is None
    assert palette is not None
    assert palette.agent_child == "#00ff00"


def test_todo_in_progress_glyph_uses_main_agent_role() -> None:
    theme.set_active_palette(theme.GRUVBOX_DARK)
    item = {"content": "wire roles", "status": "in_progress"}
    rendered = TodoWidget._render_item(item, width=40)  # type: ignore[arg-type]
    glyph_style, _ = rendered[0]
    assert glyph_style == f"fg:{theme.AGENT_MAIN}"


def test_approval_card_border_is_warning_not_accent() -> None:
    theme.set_active_palette(theme.GRUVBOX_DARK)
    card = render_approval_card("bash", {"command": "ls"})
    assert isinstance(card, Panel)
    assert card.border_style == theme.WARNING
    assert card.border_style != theme.ACCENT


def test_inbox_notice_uses_notice_role() -> None:
    theme.set_active_palette(theme.GRUVBOX_DARK)
    rendered = render_event(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={"kind": "project_inbox", "text": "message from peer"},
        )
    )
    assert isinstance(rendered, Text)
    assert rendered.style == theme.NOTICE


def test_agent_failure_notification_stays_error_red() -> None:
    theme.set_active_palette(theme.GRUVBOX_DARK)
    rendered = render_event(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "description": "inspect repository",
                "status": "error",
                "text": "boom",
            },
        )
    )
    assert isinstance(rendered, Text)
    assert rendered.style == theme.ERROR


def test_representative_tool_cards_match_main_fixture() -> None:
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    rendered = representative_cards()
    assert set(rendered) == set(expected)
    for name, ansi in expected.items():
        assert rendered[name] == ansi, f"tool card '{name}' drifted from main"
