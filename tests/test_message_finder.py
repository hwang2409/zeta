"""Behaviour tests for the transcript message finder ranking model."""

from __future__ import annotations

from zeta.tui.transcript.message_finder import Candidate, MessageFinder, Role


def _candidate(index: int, text: str, role: Role = Role.USER) -> Candidate:
    return Candidate(index=index, role=role, marker=f"#{index}", text=text, preview=(text,))


def _finder(texts: list[str], **kwargs) -> MessageFinder:
    candidates = [_candidate(index, text) for index, text in enumerate(texts)]
    return MessageFinder(candidates, **kwargs)


def test_empty_query_lists_candidates_newest_first() -> None:
    finder = _finder(["first", "second", "third"])
    assert [row.candidate.index for row in finder.rows] == [2, 1, 0]
    assert all(row.highlights == () for row in finder.rows)


def test_query_filters_and_ranks_by_score() -> None:
    finder = _finder(
        [
            "run the pytest suite",
            "a mid py thing snappy",
            "docs about python",
        ]
    )
    finder.set_query("py")
    texts = [row.candidate.text for row in finder.rows]
    # Every surviving row contains the subsequence; the boundary match ranks
    # above the mid-word one.
    assert "docs about python" in texts
    assert texts[0] in {"docs about python", "run the pytest suite"}
    assert "a mid py thing snappy" in texts


def test_highlights_point_at_matched_excerpt_characters() -> None:
    finder = _finder(["src/app.py"])
    finder.set_query("app")
    row = finder.rows[0]
    assert "".join(row.excerpt[column] for column in row.highlights) == "app"


def test_long_text_excerpt_is_windowed_with_ellipsis() -> None:
    long_text = "x" * 300 + " needle tail"
    finder = _finder([long_text], excerpt_width=40)
    finder.set_query("needle")
    row = finder.rows[0]
    assert len(row.excerpt) <= 42  # width plus two ellipses
    assert row.excerpt.startswith("…")
    assert "".join(row.excerpt[column] for column in row.highlights) == "needle"


def test_no_matches_clears_rows() -> None:
    finder = _finder(["alpha", "beta"])
    finder.set_query("zzzz")
    assert finder.rows == ()
    assert finder.selected is None


def test_selection_moves_and_clamps() -> None:
    finder = _finder(["one", "two", "three"])
    assert finder.selected_index == 0
    finder.move(1)
    assert finder.selected_index == 1
    finder.move(10)
    assert finder.selected_index == 2
    finder.move(-10)
    assert finder.selected_index == 0


def test_selection_clamps_when_results_shrink() -> None:
    finder = _finder(["apple", "apricot", "banana"])
    finder.set_query("ap")
    finder.move(1)
    assert finder.selected_index == 1
    finder.set_query("banana")
    assert len(finder.rows) == 1
    assert finder.selected_index == 0


def test_preview_returns_selected_candidate_lines() -> None:
    candidates = [
        Candidate(0, Role.ASSISTANT, "#0", "hello world", preview=("hello", "world")),
        Candidate(1, Role.USER, "#1", "goodbye", preview=("goodbye",)),
    ]
    finder = MessageFinder(candidates)
    finder.set_query("hello")
    assert finder.preview() == ("hello", "world")


def test_bounded_ranking_matches_full_ranking() -> None:
    texts = [f"message {index} about pytest" for index in range(2000)]
    finder = _finder(texts)
    finder.set_query("pytest")
    # The first slice does not finish a 2000-candidate scan.
    assert not finder.complete
    guard = 0
    while not finder.rank_more(budget=400):
        guard += 1
        assert guard < 100
    full = _finder(texts)
    full.set_query("pytest")
    full.rank_all()
    assert [row.candidate.index for row in finder.rows] == [
        row.candidate.index for row in full.rows
    ]


def test_results_are_capped() -> None:
    texts = [f"pytest candidate {index}" for index in range(500)]
    finder = _finder(texts, max_results=50)
    finder.set_query("pytest")
    finder.rank_all()
    assert len(finder.rows) == 50
    # The cap keeps the newest matches (highest index on score ties).
    assert finder.rows[0].candidate.index == 499


def test_extended_syntax_flows_through() -> None:
    finder = _finder(["src/app.py", "src/app.js", "lib/app.py"])
    finder.set_query("py$ !lib")
    texts = {row.candidate.text for row in finder.rows}
    assert texts == {"src/app.py"}
