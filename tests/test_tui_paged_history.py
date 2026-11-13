from pathlib import Path

from prompt_toolkit.document import Document

from zeta.tui.session_prompt_history import SessionPromptHistory


def test_prompt_owner_keeps_only_two_pages_and_exact_draft() -> None:
    owner = SessionPromptHistory()
    path = Path("session")
    owner.bind(path)
    owner.append(path, "a")
    assert owner.older(path, Document("draft", cursor_position=2)) == Document(
        "a", cursor_position=1
    )
    for _index in range(300):
        owner.append(path, "x")
    assert len(owner.pages) <= 2
    owner.clear()
    assert owner.pages == ()
