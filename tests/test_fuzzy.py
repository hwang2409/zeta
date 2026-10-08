"""Unit tests for the fzf-style fuzzy matcher."""

from __future__ import annotations

from zeta.tui.transcript.fuzzy import match, parse_query


def _score(query: str, text: str) -> int | None:
    result = match(query, text)
    return None if result is None else result.score


def test_requires_subsequence() -> None:
    assert match("abc", "xyz") is None
    assert match("abc", "a b c") is not None
    assert match("zyx", "xyz") is None


def test_positions_point_at_matched_characters() -> None:
    result = match("app", "src/app.py")
    assert result is not None
    assert "".join("src/app.py"[index] for index in result.positions) == "app"


def test_boundary_beats_midword() -> None:
    boundary = _score("app", "the application")
    midword = _score("app", "snappy")
    assert boundary is not None and midword is not None
    assert boundary > midword


def test_fewer_gaps_scores_higher() -> None:
    tight = _score("abc", "abc")
    one_gap = _score("abc", "ab_c")
    two_gaps = _score("abc", "a_b_c")
    assert tight is not None and one_gap is not None and two_gaps is not None
    assert tight > one_gap > two_gaps


def test_consecutive_run_beats_scattered() -> None:
    consecutive = _score("abc", "abcdef")
    scattered = _score("abc", "a_b_c_d")
    assert consecutive is not None and scattered is not None
    assert consecutive > scattered


def test_later_contiguous_alignment_beats_early_sparse_alignment() -> None:
    result = match("ab", "a---b ab")
    spaced = _score("ab", "a b")
    assert result is not None and spaced is not None
    assert result.positions == (6, 7)
    assert result.score > spaced


def test_oversized_fuzzy_input_uses_bounded_contiguous_matching() -> None:
    query = "a" * 300
    text = f"prefix {query} suffix"
    result = match(query, text)
    assert result is not None
    assert result.positions == tuple(range(7, 307))
    assert match(query, "a " * 300) is None


def test_camelcase_bonus() -> None:
    assert match("cCB", "camelCaseBonus") is not None
    camel = _score("ab", "AlphaBravo")
    midword = _score("ab", "alphabet")
    assert camel is not None and midword is not None
    assert camel > midword


def test_path_separator_bonus() -> None:
    after_sep = _score("app", "src/app.py")
    midword = _score("app", "mapplication")
    assert after_sep is not None and midword is not None
    assert after_sep > midword


def test_smart_case_lowercase_matches_any_case() -> None:
    assert match("app", "APP") is not None
    assert match("app", "App") is not None


def test_smart_case_uppercase_is_case_sensitive() -> None:
    assert match("App", "App") is not None
    assert match("App", "app") is None
    assert match("APP", "app") is None


def test_exact_term_requires_substring() -> None:
    assert match("'app", "src/app.py") is not None
    assert match("'app", "a p p") is None


def test_prefix_term() -> None:
    assert match("^src", "src/app.py") is not None
    assert match("^app", "src/app.py") is None


def test_suffix_term() -> None:
    assert match("py$", "src/app.py") is not None
    assert match("js$", "src/app.py") is None


def test_equal_term() -> None:
    assert match("^read$", "read") is not None
    assert match("^read$", "reader") is None


def test_negated_term_rejects_match() -> None:
    assert match("!test", "src/app.py") is not None
    assert match("!test", "tests/test_app.py") is None


def test_negated_operator_terms() -> None:
    assert match("!^src", "lib/app.py") is not None
    assert match("!^src", "src/app.py") is None
    assert match("!py$", "src/app.go") is not None
    assert match("!py$", "src/app.py") is None


def test_and_terms_all_required() -> None:
    assert match("app py", "src/app.py") is not None
    assert match("app js", "src/app.py") is None


def test_and_terms_sum_positions() -> None:
    result = match("app py", "src/app.py")
    assert result is not None
    text = "src/app.py"
    assert "".join(text[index] for index in result.positions) == "apppy"


def test_negated_term_contributes_no_positions() -> None:
    plain = match("app", "src/app.py")
    with_negation = match("app !zzz", "src/app.py")
    assert plain is not None and with_negation is not None
    assert plain.positions == with_negation.positions


def test_empty_query_matches_with_zero_score() -> None:
    result = match("   ", "anything")
    assert result is not None
    assert result.score == 0
    assert result.positions == ()


def test_parse_query_splits_terms() -> None:
    query = parse_query("one two three")
    assert len(query.terms) == 3
    assert not parse_query("   ").terms


def test_ranking_order_on_known_candidates() -> None:
    candidates = [
        "src/zeta/tui/app.py",
        "src/zeta/app_helpers.py",
        "docs/apps.md",
        "tests/test_mapping.py",
    ]
    ranked = sorted(
        (c for c in candidates if match("app", c) is not None),
        key=lambda c: match("app", c).score,  # type: ignore[union-attr]
        reverse=True,
    )
    # The file that ends in app.py (boundary + consecutive) ranks first; the
    # mid-word "mapping" match ranks last of the survivors.
    assert ranked[0] == "src/zeta/tui/app.py"
    assert ranked[-1] == "tests/test_mapping.py"
