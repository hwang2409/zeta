"""Shared visual tokens for the full-screen terminal UI.

Palettes:

- ``dark`` — original zeta palette, tuned for dark terminals.
- ``light`` — foreground/background swap plus a light pygments theme so light
  terminals stay readable.
- user overrides — ``~/.zeta/themes/<name>.toml`` layers arbitrary keys over
  the ``dark`` defaults; malformed files fail open with a notice.

The active palette is selected via ``settings.theme`` at startup, or the
``/theme`` slash command at runtime. Module-level constants are recomputed
in place by :func:`set_active_palette` so callers that already imported them
still see the new values on the next attribute lookup through the ``theme``
module (``from . import theme; theme.BODY``).
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rich.theme import Theme


@dataclass(frozen=True, slots=True)
class Palette:
    """Immutable named color palette."""

    name: str
    accent: str
    dim: str
    body: str
    error: str
    card_border: str
    composer_border: str
    search_bg: str
    code_theme: str
    surface: str = ""
    tint: str = ""
    code_bg: str = "default"
    # Text drawn on top of the accent colour (menu and search highlights).
    on_accent: str = "#000000"


DARK = Palette(
    name="dark",
    accent="#ff8a1f",
    dim="#8e938b",
    body="#f1f2ed",
    error="bold #ff5c57",
    card_border="#50544d",
    composer_border="#50544d",
    search_bg="#50544d",
    code_theme="monokai",
)


# Colors chosen so foreground/borders reach WCAG AA against a white terminal
# background: body #1c1e1a and dim #5f6360 pass 4.5:1; accent #b05300 passes
# 4.5:1 against #ffffff for the composer prompt caret and the /command style.
LIGHT = Palette(
    name="light",
    accent="#b05300",
    dim="#5f6360",
    body="#1c1e1a",
    error="bold #a71d24",
    card_border="#c4c8c0",
    composer_border="#c4c8c0",
    search_bg="#fff2b0",
    code_theme="friendly",
    on_accent="#ffffff",
)


BUILT_IN_PALETTES: Mapping[str, Palette] = MappingProxyType(
    {DARK.name: DARK, LIGHT.name: LIGHT}
)


# Public style constants — set by :func:`set_active_palette` at import time
# and mutated in place on every subsequent switch. Consumers that reach them
# via ``theme.BODY`` (``from . import theme``) see the current values; a
# stale ``from .theme import BODY`` capture pins the value at import.
SURFACE: str
TINT: str
ACCENT: str
DIM: str
BODY: str
ERROR: str
CODE_BG: str
CODE_THEME: str
CHROME: str
CARD_BG: str
CARD_BORDER: str
COMPOSER_BORDER: str
COMPOSER_FOCUS: str
VIM_STATE: str
PLAN_STATE: str
COMMAND: str
RECEIPT: str
THOUGHT: str
AFFORDANCE: str
USER_ROLE: str
SEARCH_MATCH: str
SEARCH_CURRENT: str
MENU_BG: str
ON_ACCENT: str


# Single Rich Theme instance whose ``styles`` dict is mutated in place so
# every Console constructed with it picks up palette swaps.
RICH_THEME: Theme = Theme({})


_ACTIVE: Palette


def active_palette() -> Palette:
    """Return the palette currently applied to the module-level constants."""

    return _ACTIVE


def set_active_palette(palette: Palette) -> None:
    """Rebuild every module-level style token from ``palette``.

    Mutates the shared :data:`RICH_THEME` instance so already-built Consoles
    also flip to the new palette on the next render.
    """

    global _ACTIVE, SURFACE, TINT, ACCENT, DIM, BODY, ERROR, CODE_BG, CODE_THEME
    global CHROME, CARD_BG, CARD_BORDER, COMPOSER_BORDER, COMPOSER_FOCUS
    global VIM_STATE, PLAN_STATE, COMMAND, RECEIPT, THOUGHT, AFFORDANCE
    global USER_ROLE, SEARCH_MATCH, SEARCH_CURRENT, MENU_BG, ON_ACCENT
    _ACTIVE = palette
    SURFACE = palette.surface
    TINT = palette.tint
    ACCENT = palette.accent
    DIM = palette.dim
    BODY = palette.body
    ERROR = palette.error
    CODE_BG = palette.code_bg
    CODE_THEME = palette.code_theme
    CHROME = DIM
    CARD_BG = ""
    CARD_BORDER = palette.card_border
    COMPOSER_BORDER = palette.composer_border
    COMPOSER_FOCUS = ACCENT
    VIM_STATE = f"bold {ACCENT}"
    PLAN_STATE = f"bold {ACCENT}"
    COMMAND = f"bold {ACCENT}"
    RECEIPT = CHROME
    THOUGHT = f"italic {CHROME}"
    AFFORDANCE = f"dim {DIM}"
    USER_ROLE = ACCENT
    SEARCH_MATCH = f"{BODY} on {palette.search_bg}"
    SEARCH_CURRENT = f"black on {ACCENT}"
    MENU_BG = palette.search_bg
    ON_ACCENT = palette.on_accent
    _refresh_rich_theme()


def _refresh_rich_theme() -> None:
    from rich.style import Style

    styles = {
        "markdown.paragraph": BODY,
        "markdown.h1": f"bold {BODY}",
        "markdown.h2": f"bold {BODY}",
        "markdown.h3": BODY,
        "markdown.link": f"{BODY} underline",
        "markdown.link_url": f"{CHROME} underline",
        "markdown.code": BODY,
        "markdown.code_block": BODY,
        "markdown.table.border": CHROME,
        "markdown.table.header": f"bold {BODY}",
        "zeta.card": CARD_BG,
        "zeta.card.border": CARD_BORDER,
        "zeta.command": COMMAND,
        "zeta.receipt": RECEIPT,
        "zeta.thought": THOUGHT,
        "zeta.affordance": AFFORDANCE,
    }
    RICH_THEME.styles.clear()
    RICH_THEME.styles.update(
        {name: Style.parse(value) if value else Style() for name, value in styles.items()}
    )


# --- resolution + loading -------------------------------------------------


class ThemeLoadError(Exception):
    """Raised when a user-supplied theme file cannot be loaded."""


def _themes_dir(home: str | Path | None) -> Path | None:
    if home is None:
        return None
    return Path(home).expanduser() / "themes"


def _display_path(path: Path) -> str:
    try:
        home_path = Path.home()
    except (RuntimeError, OSError):
        return str(path)
    try:
        relative = path.relative_to(home_path)
    except ValueError:
        return str(path)
    return f"~/{relative}"


def resolve_palette(
    name: str,
    *,
    home: str | Path | None = None,
) -> tuple[Palette | None, str | None]:
    """Return ``(palette, notice)`` for ``name``.

    User overrides in ``$home/themes/<name>.toml`` take precedence over
    built-ins so an operator can retune ``dark`` without shipping a new
    binary. Missing files fall back to built-ins; malformed files return
    ``(None, notice)`` so the caller keeps the current palette.
    """

    theme_dir = _themes_dir(home)
    if theme_dir is not None:
        candidate = theme_dir / f"{name}.toml"
        if candidate.is_file():
            try:
                data = _read_theme_file(candidate)
            except ThemeLoadError as exc:
                return None, str(exc)
            palette, error = _palette_from_dict(name, data)
            if palette is not None:
                return palette, None
            return None, (
                f"theme · ignored {_display_path(candidate)}: {error}"
            )
    built_in = BUILT_IN_PALETTES.get(name)
    if built_in is None:
        return None, None
    return built_in, None


def list_available(home: str | Path | None = None) -> tuple[str, ...]:
    """Return the sorted union of built-in and user theme names."""

    names = set(BUILT_IN_PALETTES)
    theme_dir = _themes_dir(home)
    if theme_dir is not None and theme_dir.is_dir():
        for path in theme_dir.glob("*.toml"):
            names.add(path.stem)
    return tuple(sorted(names))


def _read_theme_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ThemeLoadError(f"theme · could not read {_display_path(path)}: {exc}") from exc
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ThemeLoadError(f"theme · ignored {_display_path(path)}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ThemeLoadError(
            f"theme · ignored {_display_path(path)}: top-level is not a table"
        )
    return parsed


_PALETTE_KEYS: frozenset[str] = frozenset(
    {
        "accent",
        "dim",
        "body",
        "error",
        "card_border",
        "composer_border",
        "search_bg",
        "code_theme",
        "surface",
        "tint",
        "code_bg",
        "on_accent",
    }
)


def _palette_from_dict(
    name: str, data: Mapping[str, Any]
) -> tuple[Palette | None, str | None]:
    """Build a palette from ``data``, layering unset keys over ``DARK``."""

    for key, value in data.items():
        if key not in _PALETTE_KEYS:
            return None, f"unknown key '{key}'"
        if not isinstance(value, str):
            return None, f"key '{key}' must be a string"
    return (
        Palette(
            name=name,
            accent=data.get("accent", DARK.accent),
            dim=data.get("dim", DARK.dim),
            body=data.get("body", DARK.body),
            error=data.get("error", DARK.error),
            card_border=data.get("card_border", DARK.card_border),
            composer_border=data.get("composer_border", DARK.composer_border),
            search_bg=data.get("search_bg", DARK.search_bg),
            code_theme=data.get("code_theme", DARK.code_theme),
            surface=data.get("surface", DARK.surface),
            tint=data.get("tint", DARK.tint),
            code_bg=data.get("code_bg", DARK.code_bg),
            on_accent=data.get("on_accent", DARK.on_accent),
        ),
        None,
    )


# Initialize the module-level constants on import so ``from .theme import
# BODY`` at module load time in other files sees a well-formed value.
set_active_palette(DARK)


__all__ = [
    "BUILT_IN_PALETTES",
    "DARK",
    "LIGHT",
    "RICH_THEME",
    "Palette",
    "ThemeLoadError",
    "active_palette",
    "list_available",
    "resolve_palette",
    "set_active_palette",
]
