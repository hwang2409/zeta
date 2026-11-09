from __future__ import annotations

import codecs

import pytest

from zeta.tools._shared import process_output


def _incremental_reference(data: bytes, limit: int) -> bytes:
    if limit == 0:
        return b""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    character_start = 0
    characters = 0
    for index, value in enumerate(data):
        emitted = decoder.decode(bytes((value,)), final=False)
        if not emitted:
            continue
        if characters + len(emitted) > limit:
            return data[:character_start]
        characters += len(emitted)
        character_start = index + 1
    emitted = decoder.decode(b"", final=True)
    if characters + len(emitted) > limit:
        return data[:character_start]
    return data


@pytest.mark.parametrize(
    ("data", "limit"),
    [
        (b"ascii", 3),
        (b"ascii", len(b"ascii")),
        (b"ascii", len(b"ascii") + 1),
        ("éclair".encode(), 2),
        (b"a\xc3\xa9", 2),
        (b"\xff", 1),
        (b"", 4),
        (b"ascii\xc3\xa9", 6),
    ],
    ids=[
        "ascii",
        "equal-limit",
        "over-limit",
        "multibyte-at-boundary",
        "mixed-ascii-then-multibyte",
        "invalid-utf8-fallback",
        "empty",
        "mixed-ascii-then-multibyte-limit",
    ],
)
def test_utf8_char_chunk_matches_incremental_decoder(data: bytes, limit: int) -> None:
    assert process_output.utf8_char_chunk(data, limit) == _incremental_reference(
        data, limit
    )


def test_utf8_char_chunk_does_not_create_decoder_for_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decoder_must_not_be_entered(_encoding: str):
        raise AssertionError("incremental decoder entered for pure ASCII")

    monkeypatch.setattr(
        process_output.codecs, "getincrementaldecoder", decoder_must_not_be_entered
    )
    assert process_output.utf8_char_chunk(b"pure ascii output", 9) == b"pure asci"
