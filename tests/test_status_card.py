from io import StringIO

import pytest
from prompt_toolkit.document import Document
from rich.console import Console
from rich.text import Text

from zeta.core.fake import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.status_card import StatusCardControl


def test_status_card_is_focusable_and_scrolls_with_a_bounded_offset() -> None:
    card = StatusCardControl()
    card.set_lines([f"line {index}" for index in range(20)])

    assert card.is_focusable
    card.create_content(width=40, height=5)
    card.scroll(3)
    assert card.offset == 3
    card.scroll(100)
    assert card.offset == 15
    card.page(-1)
    assert card.offset == 12
    card.top()
    assert card.offset == 0
    card.bottom()
    assert card.offset == 15


@pytest.mark.asyncio
async def test_status_card_open_close_preserves_composer_and_transcript(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._transcript.append(Text("existing transcript"))
    units = app._transcript.units
    session.default_buffer.set_document(Document("draft text", 5))
    app.open_status_card()

    assert app.status_card_active
    assert app._transcript.units == units
    assert session.layout.has_focus(app._status_card_window)
    app._status_card.create_content(width=40, height=4)
    app._status_card.bottom()
    assert app._status_card.offset > 0

    app.close_status_card()
    assert not app.status_card_active
    assert session.default_buffer.text == "draft text"
    assert session.default_buffer.cursor_position == 5
    assert session.layout.has_focus(session.default_buffer)
    assert app._transcript.units == units


def test_status_card_content_is_width_bounded() -> None:
    card = StatusCardControl()
    card.set_lines(["x" * 100])
    content = card.create_content(width=12, height=3)

    rendered = "".join(text for _, text in content.get_line(0))
    assert len(rendered) == 12
    assert rendered.startswith("│ ")
    assert rendered.endswith(" │")
