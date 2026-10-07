"""fzf-style fuzzy matching for the transcript message finder.

This module is the single owner of query parsing and scoring. It knows nothing
about the terminal UI: it takes a query string and a candidate string and
returns a :class:`Match` (score plus the matched character positions) or
``None`` when the candidate is rejected. The finder overlay renders those
positions; the transcript decides what to do with the ranking.

The matcher follows fzf's model closely:

* A query is split on whitespace into terms that are ANDed together.
* Each term is fuzzy by default, with fzf's extended syntax:
  ``'exact`` (substring), ``^prefix``, ``suffix$``, ``^whole$`` and a leading
  ``!`` to negate (reject candidates the remainder matches).
* Matching is smart-case per term: a term with any uppercase letter matches
  case-sensitively, otherwise case-insensitively.
* Fuzzy scoring rewards matches on word boundaries, camelCase humps, matches
  after path/underscore separators, and consecutive runs, and prefers shorter,
  earlier matches -- the same bonuses fzf uses, so the ranking order matches a
  user's muscle memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# fzf scoring weights (fzf/src/algo/algo.go). Kept as module constants so the
# ordering is auditable against upstream rather than hidden in the DP.
_SCORE_MATCH = 16
_SCORE_GAP_START = -3
_SCORE_GAP_EXTENSION = -1
_BONUS_BOUNDARY = _SCORE_MATCH // 2  # 8
_BONUS_NON_WORD = _SCORE_MATCH // 2  # 8
_BONUS_CAMEL = _BONUS_BOUNDARY + _SCORE_GAP_EXTENSION  # 7
_BONUS_CONSECUTIVE = -(_SCORE_GAP_START + _SCORE_GAP_EXTENSION)  # 4
_BONUS_FIRST_CHAR_MULTIPLIER = 2
_BONUS_BOUNDARY_WHITE = _BONUS_BOUNDARY + 2  # 10
_BONUS_BOUNDARY_DELIMITER = _BONUS_BOUNDARY + 1  # 9

_DELIMITERS = frozenset("/,:;|")


class _CharClass(Enum):
    WHITE = 0
    NON_WORD = 1
    DELIMITER = 2
    LOWER = 3
    UPPER = 4
    LETTER = 5
    NUMBER = 6


def _char_class(char: str) -> _CharClass:
    if char.isspace():
        return _CharClass.WHITE
    if char in _DELIMITERS:
        return _CharClass.DELIMITER
    if char.isdigit():
        return _CharClass.NUMBER
    if char.isalpha():
        if char.islower() or not char.isupper():
            # ``not isupper`` keeps caseless scripts in the lower bucket so a
            # camel bonus never triggers spuriously between two such letters.
            return _CharClass.LOWER if char.islower() else _CharClass.LETTER
        return _CharClass.UPPER
    return _CharClass.NON_WORD


def _bonus_for(prev: _CharClass, current: _CharClass) -> int:
    """Return the boundary bonus earned by ``current`` following ``prev``."""

    if current in (_CharClass.LOWER, _CharClass.UPPER, _CharClass.LETTER, _CharClass.NUMBER):
        if prev is _CharClass.WHITE:
            return _BONUS_BOUNDARY_WHITE
        if prev is _CharClass.DELIMITER:
            return _BONUS_BOUNDARY_DELIMITER
        if prev is _CharClass.NON_WORD:
            return _BONUS_BOUNDARY
    if (
        (prev is _CharClass.LOWER and current is _CharClass.UPPER)
        or (prev is not _CharClass.NUMBER and current is _CharClass.NUMBER)
    ):
        return _BONUS_CAMEL
    if current is _CharClass.NON_WORD:
        return _BONUS_NON_WORD
    if current is _CharClass.WHITE:
        return _BONUS_BOUNDARY_WHITE
    return 0


@dataclass(frozen=True, slots=True)
class Match:
    """A successful match: a total ``score`` and the matched ``positions``.

    ``positions`` are character indices into the candidate string, sorted and
    de-duplicated across every term, ready for highlighting.
    """

    score: int
    positions: tuple[int, ...]


class _TermKind(Enum):
    FUZZY = 0
    EXACT = 1
    PREFIX = 2
    SUFFIX = 3
    EQUAL = 4


@dataclass(frozen=True, slots=True)
class _Term:
    kind: _TermKind
    text: str
    negate: bool
    case_sensitive: bool


@dataclass(frozen=True, slots=True)
class Query:
    """A parsed query: whitespace-separated terms ANDed together.

    ``is_empty`` is true when the query contributes no constraints, so the
    finder can short-circuit to the unfiltered, newest-first candidate order.
    """

    terms: tuple[_Term, ...]

    @property
    def is_empty(self) -> bool:
        return not self.terms


def _smart_case(text: str) -> bool:
    return any(character.isupper() for character in text)


def _parse_term(token: str) -> _Term | None:
    negate = token.startswith("!")
    if negate:
        token = token[1:]
    if not token:
        return None
    kind = _TermKind.FUZZY
    if token.startswith("'"):
        token = token[1:]
        kind = _TermKind.EXACT
    elif token.startswith("^") and token.endswith("$") and len(token) > 1:
        token = token[1:-1]
        kind = _TermKind.EQUAL
    elif token.startswith("^"):
        token = token[1:]
        kind = _TermKind.PREFIX
    elif token.endswith("$"):
        token = token[:-1]
        kind = _TermKind.SUFFIX
    if not token:
        return None
    # A negated term with no explicit operator is an inverse substring match,
    # matching fzf's behaviour for bare ``!foo``.
    if negate and kind is _TermKind.FUZZY:
        kind = _TermKind.EXACT
    return _Term(kind, token, negate, _smart_case(token))


def parse_query(query: str) -> Query:
    """Parse ``query`` into ANDed terms with fzf extended syntax."""

    terms = tuple(
        term for token in query.split() if (term := _parse_term(token)) is not None
    )
    return Query(terms)


def _cased(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def _fuzzy_match(
    text: str, pattern: str, case_sensitive: bool
) -> tuple[int, tuple[int, ...]] | None:
    """Score a fuzzy term against ``text`` with fzf's bonus model.

    Returns ``(score, positions)`` for the optimal alignment, or ``None`` when
    the pattern is not a subsequence of ``text``.
    """

    if not pattern:
        return 0, ()
    haystack = _cased(text, case_sensitive)
    needle = _cased(pattern, case_sensitive)

    # Precompute the boundary bonus available at each text position.
    classes = [_CharClass.WHITE]
    classes.extend(_char_class(character) for character in text)
    bonuses = [
        _bonus_for(classes[index], classes[index + 1]) for index in range(len(text))
    ]

    # Greedy forward scan to confirm the subsequence and bound the window; then
    # a backward scan to pull the match as far right as possible, preferring the
    # tighter, boundary-aligned alignment fzf favours.
    first_index: int | None = None
    needle_pos = 0
    for text_index, character in enumerate(haystack):
        if needle_pos < len(needle) and character == needle[needle_pos]:
            if first_index is None:
                first_index = text_index
            needle_pos += 1
            if needle_pos == len(needle):
                break
    if needle_pos != len(needle) or first_index is None:
        return None
    last_index = text_index

    return _score_window(haystack, needle, bonuses, first_index, last_index)


def _score_window(
    haystack: str,
    needle: str,
    bonuses: list[int],
    start: int,
    end: int,
) -> tuple[int, tuple[int, ...]]:
    """Dynamic-programming score over ``haystack[start:end + 1]``.

    This mirrors fzf's V2 matrix: ``score`` tracks the best alignment ending at
    each cell and ``consecutive`` tracks run length so consecutive matches earn
    the run bonus. Back-pointers recover the matched positions for highlight.
    """

    width = end - start + 1
    height = len(needle)
    best_score = 0
    best_cell = (0, 0)
    # ``consecutive[j]`` from the previous needle row; ``prev_score`` likewise.
    prev_scores = [0] * width
    prev_consec = [0] * width
    # Back-pointer: for each (row, col) whether the best path matched here.
    pointers: list[list[bool]] = []

    for row in range(height):
        current_scores = [0] * width
        current_consec = [0] * width
        row_pointers = [False] * width
        needle_char = needle[row]
        in_gap = False
        for col in range(width):
            text_index = start + col
            score1 = 0  # score if we do NOT match needle_char at this column
            if col > 0:
                if in_gap:
                    score1 = current_scores[col - 1] + _SCORE_GAP_EXTENSION
                else:
                    score1 = current_scores[col - 1] + _SCORE_GAP_START
            score2 = 0  # score if we DO match needle_char at this column
            consec = 0
            if haystack[text_index] == needle_char:
                diag = prev_scores[col - 1] if col > 0 else 0
                if row == 0:
                    # First needle char may start anywhere; fzf scores it the
                    # same regardless of absolute offset (position is only a
                    # tiebreaker), with the boundary bonus amplified.
                    bonus = bonuses[text_index]
                    score2 = _SCORE_MATCH + bonus * _BONUS_FIRST_CHAR_MULTIPLIER
                    consec = 1
                elif col > 0:
                    prev_consecutive = prev_consec[col - 1]
                    bonus = bonuses[text_index]
                    if prev_consecutive > 0:
                        consec = prev_consecutive + 1
                        # Keep the stronger of a fresh boundary bonus or the
                        # consecutive-run bonus, as fzf does.
                        if bonus >= _BONUS_BOUNDARY and bonus > bonuses[
                            text_index - consec + 1
                        ]:
                            consec = 1
                        else:
                            bonus = max(bonus, _BONUS_CONSECUTIVE, _BONUS_BOUNDARY if consec > 1 else 0)
                    else:
                        consec = 1
                    score2 = diag + _SCORE_MATCH + bonus
            if score2 > 0 and score2 >= score1:
                current_scores[col] = score2
                current_consec[col] = consec
                row_pointers[col] = True
                in_gap = False
            else:
                current_scores[col] = max(0, score1)
                current_consec[col] = 0
                row_pointers[col] = False
                in_gap = score1 > 0
            if row == height - 1 and current_scores[col] > best_score and row_pointers[col]:
                best_score = current_scores[col]
                best_cell = (row, col)
        pointers.append(row_pointers)
        prev_scores = current_scores
        prev_consec = current_consec

    if best_score <= 0:
        # Degenerate fallback: a subsequence always has a positive alignment,
        # but guard against pathological inputs by returning the greedy run.
        positions = _greedy_positions(haystack, needle, start)
        return max(1, len(needle) * _SCORE_MATCH), positions

    # Recover positions by walking back-pointers from the best final cell.
    positions: list[int] = []
    row, col = best_cell
    while row >= 0 and col >= 0:
        if pointers[row][col]:
            positions.append(start + col)
            row -= 1
            col -= 1
        else:
            col -= 1
        if row < 0:
            break
    positions.reverse()
    return best_score, tuple(positions)


def _greedy_positions(haystack: str, needle: str, start: int) -> tuple[int, ...]:
    positions: list[int] = []
    needle_pos = 0
    for index in range(start, len(haystack)):
        if needle_pos < len(needle) and haystack[index] == needle[needle_pos]:
            positions.append(index)
            needle_pos += 1
            if needle_pos == len(needle):
                break
    return tuple(positions)


def _boundary_bonus(text: str, index: int, length: int) -> int:
    """Approximate score for a contiguous literal match of ``length`` at ``index``."""

    classes = [_CharClass.WHITE]
    classes.extend(_char_class(character) for character in text)
    bonus = _bonus_for(classes[index], classes[index + 1])
    consecutive = (length - 1) * _BONUS_CONSECUTIVE
    # Earlier matches score higher, mirroring fzf's leading-gap penalty.
    return length * _SCORE_MATCH + bonus * _BONUS_FIRST_CHAR_MULTIPLIER + consecutive - index


def _exact_match(
    text: str, term: _Term
) -> tuple[int, tuple[int, ...]] | None:
    haystack = _cased(text, term.case_sensitive)
    needle = _cased(term.text, term.case_sensitive)
    index = haystack.find(needle)
    if index < 0:
        return None
    positions = tuple(range(index, index + len(needle)))
    return _boundary_bonus(text, index, len(needle)), positions


def _prefix_match(
    text: str, term: _Term
) -> tuple[int, tuple[int, ...]] | None:
    haystack = _cased(text, term.case_sensitive)
    needle = _cased(term.text, term.case_sensitive)
    if not haystack.startswith(needle):
        return None
    positions = tuple(range(len(needle)))
    return _boundary_bonus(text, 0, len(needle)), positions


def _suffix_match(
    text: str, term: _Term
) -> tuple[int, tuple[int, ...]] | None:
    haystack = _cased(text, term.case_sensitive)
    needle = _cased(term.text, term.case_sensitive)
    if not haystack.endswith(needle):
        return None
    start = len(text) - len(needle)
    positions = tuple(range(start, len(text)))
    return _boundary_bonus(text, start, len(needle)), positions


def _equal_match(
    text: str, term: _Term
) -> tuple[int, tuple[int, ...]] | None:
    haystack = _cased(text, term.case_sensitive)
    needle = _cased(term.text, term.case_sensitive)
    if haystack != needle:
        return None
    return _boundary_bonus(text, 0, len(needle)), tuple(range(len(needle)))


_MATCHERS = {
    _TermKind.FUZZY: lambda text, term: _fuzzy_match(text, term.text, term.case_sensitive),
    _TermKind.EXACT: _exact_match,
    _TermKind.PREFIX: _prefix_match,
    _TermKind.SUFFIX: _suffix_match,
    _TermKind.EQUAL: _equal_match,
}


def match_query(query: Query, text: str) -> Match | None:
    """Score ``text`` against ``query``; return ``None`` when any term fails.

    An empty query matches everything with a zero score. Negated terms reject
    the candidate when the remainder matches and never contribute positions.
    """

    if query.is_empty:
        return Match(0, ())
    total = 0
    positions: set[int] = set()
    for term in query.terms:
        result = _MATCHERS[term.kind](text, term)
        if term.negate:
            if result is not None:
                return None
            continue
        if result is None:
            return None
        score, term_positions = result
        total += score
        positions.update(term_positions)
    return Match(total, tuple(sorted(positions)))


def match(query: str, text: str) -> Match | None:
    """Convenience wrapper: parse ``query`` and score ``text`` in one call."""

    return match_query(parse_query(query), text)


__all__ = ["Match", "Query", "match", "match_query", "parse_query"]
