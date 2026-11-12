from __future__ import annotations

import asyncio
from pathlib import Path

from zeta.runtime.update_notice import UpdateChecker, source_revision
from zeta.server.connections import EVENT_FEATURES
from zeta.server.protocol import EventDescriptor
from zeta.tui import layout


def test_checker_reports_one_revision_change_without_sleep() -> None:
    revisions = iter(("old", "new", "new"))
    updates: list[tuple[str, str]] = []

    async def on_update(old: str, new: str) -> None:
        updates.append((old, new))

    checker = UpdateChecker(
        Path("/not-used"), on_update, read_revision=lambda _: next(revisions)
    )
    assert asyncio.run(checker.tick())
    assert not asyncio.run(checker.tick())
    assert updates == [("old", "new")]


def test_no_git_checkout_uses_a_source_fingerprint(tmp_path: Path) -> None:
    package = tmp_path / "package.py"
    package.write_text("value = 1")
    first = source_revision(package)
    package.write_text("value = 22")
    assert first is not None
    assert source_revision(package) != first


def test_updated_footer_uses_the_persistent_notice(monkeypatch) -> None:
    class Output:
        def get_size(self):
            return type("Size", (), {"columns": 80})()

    class PromptApp:
        output = Output()

    class App:
        _remote_view = None
        _zeta_updated = True

    monkeypatch.setattr(layout, "get_app", lambda: PromptApp())
    assert "zeta updated · /restart" in "".join(
        text for _style, text in layout.status_toolbar(App())
    )


def test_serve_mapping_declares_update_event() -> None:
    assert EventDescriptor.ZETA_UPDATED.value == "zeta_updated"
    assert EVENT_FEATURES[EventDescriptor.ZETA_UPDATED] == frozenset({"zeta_updated"})
