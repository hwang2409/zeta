"""Behaviour tests for the transcript message finder ranking model."""

from __future__ import annotations

from zeta.tui.transcript.message_finder import Candidate, MessageFinder, Role


def _candidate(index: int, text: str, role: Role = Role.USER) -> Candidate:
    return Candidate(index=index, role=role, marker=f"#{index}", text=text, preview=(text,))


def _finder(texts: list[str], **kwargs) -> MessageFinder:
    candidates = [_candidate(index, text) for index, text in enumerate(texts)]
    return MessageFinder(candidates, **kwargs)


def _query(finder: MessageFinder, query: str) -> None:
    finder.set_query(query)
    finder.rank_all()


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
    _query(finder, "py")
    texts = [row.candidate.text for row in finder.rows]
    # Every surviving row contains the subsequence; the boundary match ranks
    # above the mid-word one.
    assert "docs about python" in texts
    assert texts[0] in {"docs about python", "run the pytest suite"}
    assert "a mid py thing snappy" in texts


def test_highlights_point_at_matched_excerpt_characters() -> None:
    finder = _finder(["src/app.py"])
    _query(finder, "app")
    row = finder.rows[0]
    assert "".join(row.excerpt[column] for column in row.highlights) == "app"


def test_long_text_excerpt_is_windowed_with_ellipsis() -> None:
    long_text = "x" * 300 + " needle tail"
    finder = _finder([long_text], excerpt_width=40)
    _query(finder, "needle")
    row = finder.rows[0]
    assert len(row.excerpt) <= 42  # width plus two ellipses
    assert row.excerpt.startswith("…")
    assert "".join(row.excerpt[column] for column in row.highlights) == "needle"


def test_no_matches_clears_rows() -> None:
    finder = _finder(["alpha", "beta"])
    _query(finder, "zzzz")
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
    _query(finder, "ap")
    finder.move(1)
    assert finder.selected_index == 1
    _query(finder, "banana")
    assert len(finder.rows) == 1
    assert finder.selected_index == 0


def test_preview_returns_selected_candidate_lines() -> None:
    candidates = [
        Candidate(0, Role.ASSISTANT, "#0", "hello world", preview=("hello", "world")),
        Candidate(1, Role.USER, "#1", "goodbye", preview=("goodbye",)),
    ]
    finder = MessageFinder(candidates)
    _query(finder, "hello")
    assert finder.preview() == ("hello", "world")


def test_stale_generation_is_never_published() -> None:
    finder = _finder(["alpha only", "beta only"])
    finder.set_query("alpha")
    stale = finder.rank()
    finder.set_query("beta")

    assert not finder.publish(stale)
    assert finder.rows == ()
    finder.rank_all()
    assert [row.candidate.text for row in finder.rows] == ["beta only"]


def test_loading_candidates_invalidates_empty_collection_ranking() -> None:
    finder = MessageFinder(())
    finder.set_query("pytest")
    stale = finder.rank()
    finder.load_candidates([_candidate(1, "run pytest")])

    assert not finder.publish(stale)
    finder.rank_all()
    assert [row.candidate.text for row in finder.rows] == ["run pytest"]


def test_results_are_capped() -> None:
    texts = [f"pytest candidate {index}" for index in range(500)]
    finder = _finder(texts, max_results=50)
    _query(finder, "pytest")
    assert len(finder.rows) == 50
    # The cap keeps the newest matches (highest index on score ties).
    assert finder.rows[0].candidate.index == 499


def test_extended_syntax_flows_through() -> None:
    finder = _finder(["src/app.py", "src/app.js", "lib/app.py"])
    _query(finder, "py$ !lib")
    texts = {row.candidate.text for row in finder.rows}
    assert texts == {"src/app.py"}
