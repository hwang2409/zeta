"""Launch the TUI with the test-only scripted backend."""

from __future__ import annotations

from zeta.tui import app as tui_app

from .tui_backend import FakeInteractiveBackend


def _build_backend(provider: str, model: str | None, **_kwargs: object):
    selected = model or {
        "claude": "claude-sonnet-4-6",
        "codex": "gpt-5.6-luna",
        "ollama": "qwen3:4b",
    }[provider]
    return FakeInteractiveBackend(delay=0.03, model=selected), selected


def main() -> int:
    tui_app.build_backend = _build_backend
    return tui_app.main()


if __name__ == "__main__":
    raise SystemExit(main())
