from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from zeta.tui.composer import ComposerCompleter, PathCompleter


def _complete(completer, text: str):
    return list(
        completer.get_completions(
            Document(text, len(text)), CompleteEvent(completion_requested=True)
        )
    )


def test_path_completion_supports_relative_parent_and_home_prefixes(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "src").mkdir()
    (cwd / "src" / "main.py").touch()
    (tmp_path / "sibling.txt").touch()
    home = tmp_path / "home"
    home.mkdir()
    (home / "notes.txt").touch()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    completer = PathCompleter(cwd)

    relative = _complete(completer, "@src/ma")
    parent = _complete(completer, "@../sib")
    home_matches = _complete(completer, "@~/not")
    absolute = _complete(completer, f"@{cwd / 'src' / 'ma'}")

    assert [item.text for item in relative] == ["src/main.py"]
    assert [item.text for item in parent] == ["../sibling.txt"]
    assert [item.text for item in home_matches] == ["~/notes.txt"]
    assert [item.text for item in absolute] == [str(cwd / "src" / "main.py")]


def test_directory_completion_adds_slash_and_drills_down(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").touch()
    completer = PathCompleter(tmp_path)

    directory = _complete(completer, "@do")
    nested = _complete(completer, "@docs/gu")

    assert [item.text for item in directory] == ["docs/"]
    assert [item.text for item in nested] == ["docs/guide.md"]


def test_spaced_paths_complete_in_quoted_form(tmp_path: Path) -> None:
    (tmp_path / "a b.txt").touch()
    completer = PathCompleter(tmp_path)

    unquoted = _complete(completer, "@a")
    quoted = _complete(completer, '@"a b')

    assert [item.text for item in unquoted] == ['"a b.txt"']
    assert [item.text for item in quoted] == ['a b.txt"']


def test_hidden_entries_require_a_dot_prefix(tmp_path: Path) -> None:
    (tmp_path / ".env").touch()
    (tmp_path / "visible").touch()
    completer = PathCompleter(tmp_path)

    assert [item.text for item in _complete(completer, "@e")] == []
    assert [item.text for item in _complete(completer, "@.")] == [".env"]


def test_path_completion_caps_matches_and_sorts_directories_first(
    tmp_path: Path,
) -> None:
    for index in range(60):
        (tmp_path / f"match-{index:02}.txt").touch()
    (tmp_path / "match-directory").mkdir()
    completer = PathCompleter(tmp_path)

    matches = _complete(completer, "@match-")

    assert len(matches) == 50
    assert matches[0].text == "match-directory/"
    assert [item.text for item in matches[1:]] == [
        f"match-{index:02}.txt" for index in range(49)
    ]


def test_bare_unknown_mention_stays_text(tmp_path: Path) -> None:
    assert _complete(PathCompleter(tmp_path), "@word") == []


def test_composer_completer_routes_slash_and_at_tokens(tmp_path: Path) -> None:
    (tmp_path / "file.txt").touch()
    registry = SimpleNamespace(
        completion_entries=(("status", "show status", ""),)
    )
    completer = ComposerCompleter(registry, tmp_path)

    slash = _complete(completer, "/sta")
    path = _complete(completer, "say @fi")

    assert [item.text for item in slash] == ["status"]
    assert [item.text for item in path] == ["file.txt"]
