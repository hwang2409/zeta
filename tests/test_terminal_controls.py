import random
import re
import sys

from zeta.tui.cards.base import UNSAFE_CONTROL_RE, safe_text, strip_terminal_controls

# Reference: the implementation before the clean-text fast path, kept verbatim.
_OSC_RE = re.compile(r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)")
_ESC_RE = re.compile(r"\x1b(?:[PX^_].*?\x1b\\|\][^\x07]*(?:\x07|\x1b\\))")
_C1_DCS_RE = re.compile(r"\x90.*?(?:\x9c|\x1b\\)", re.DOTALL)
_CSI_UNSUPPORTED_RE = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*(?!m)[@-~]")
_CSI_SGR_RE = re.compile(r"\x1b\[[0-?]*[ -/]*m")


def _reference(value: str) -> str:
    value = value.replace("\x9b", "\x1b[")
    value = _OSC_RE.sub("", value)
    value = _C1_DCS_RE.sub("", value)
    value = _ESC_RE.sub("", value)
    value = _CSI_UNSUPPORTED_RE.sub("", value)
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\x1b":
            match = _CSI_SGR_RE.match(value, index)
            if match is not None:
                result.append(match.group())
                index = match.end()
                continue
            index += 1
            continue
        character = value[index]
        if (
            character in {"\n", "\t"}
            or ord(character) >= 0x20
            and not 0x80 <= ord(character) <= 0x9F
        ):
            result.append(character)
        index += 1
    return "".join(result)


def test_unsafe_class_is_exactly_the_characters_the_function_changes() -> None:
    # Lone surrogates are included: decoded JSON can carry them.
    for code in range(sys.maxunicode + 1):
        character = chr(code)
        expected = _reference(character)
        assert strip_terminal_controls(character) == expected, hex(code)
        changed = expected != character
        assert (UNSAFE_CONTROL_RE.search(character) is not None) == changed, hex(code)


def test_mixed_controls_and_escape_sequences_match_reference() -> None:
    pieces = [
        "plain",
        " text ",
        "\n",
        "\t",
        "é",
        "漢",
        "\u200b",
        "\u202e",
        "\x7f",
        "\x1b[31m",
        "\x1b[0m",
        "\x1b[2J",
        "\x1b[?25l",
        "\x9b31m",
        "\x9b2J",
        "\x1b]0;title\x07",
        "\x1b]8;;https://x\x1b\\",
        "\x9d0;title\x9c",
        "\x1bPdata\x1b\\",
        "\x90dcs\x9c",
        "\x1b",
        "\x9b",
        "\x07",
        "\x00",
        "\r",
        "\x85",
        "[",
        "]",
        "\\",
        "m",
    ]
    rng = random.Random(1234)
    for _ in range(20_000):
        value = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 12)))
        assert strip_terminal_controls(value) == _reference(value), repr(value)


def test_safe_text_clean_card_lines_skip_ansi_parser(monkeypatch) -> None:
    calls = 0
    original = safe_text.__globals__["Text"].from_ansi

    def count_parser(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(safe_text.__globals__["Text"], "from_ansi", count_parser)
    clean = "stdout:\nplain output\nexit_code: 0"
    rendered = safe_text(clean, style="body")

    assert rendered.plain == clean
    assert calls == 0


def test_safe_text_preserves_ansi_card_output() -> None:
    rendered = safe_text("\x1b[31mfailed\x1b[0m", style="body")

    assert rendered.plain == strip_terminal_controls("\x1b[31mfailed\x1b[0m")
    assert rendered.style == "body"


def test_clean_text_skips_the_character_loop() -> None:
    value = "clean transcript text with no controls\n\t" * 250
    lines = 0
    code = strip_terminal_controls.__code__

    def trace(frame, event, arg):
        nonlocal lines
        if frame.f_code is not code:
            return None
        if event == "line":
            lines += 1
        return trace

    sys.settrace(trace)
    try:
        result = strip_terminal_controls(value)
    finally:
        sys.settrace(None)

    assert result == value
    assert lines < 10, f"{lines} lines executed for {len(value)} clean characters"
