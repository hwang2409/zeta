"""Pure output shaping for process tools."""

from __future__ import annotations

import codecs


def command_headline(command: str, limit: int = 80) -> str:
    headline = " ".join(command.split())
    if len(headline) <= limit:
        return headline
    return headline[: max(0, limit - 3)] + "..."


def task_exit_headline(task_id: str, exit_code: int | None, command: str, note: str) -> str:
    text = f"background task {task_id} exited ({exit_code}): {command_headline(command)}"
    return f"{text}; {note}" if note else text


def output_tail(output: bytearray | None) -> str:
    if not output:
        return ""
    return "\n".join(bytes(output)[-2_048:].decode(errors="replace").splitlines()[-20:])


def utf8_chunk(data: bytes, limit: int) -> bytes:
    """Cap UTF-8 output without ending inside a character."""
    candidate = data[:limit]
    try:
        candidate.decode()
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.end == len(candidate):
            candidate = candidate[: exc.start]
    if candidate:
        return candidate
    for end in range(1, len(data) + 1):
        try:
            data[:end].decode()
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data" or exc.end != end:
                return data[:end]
        else:
            return data[:end]
    return data


def utf8_char_chunk(data: bytes, limit: int) -> bytes:
    """Cap decoded UTF-8 characters without advancing past retained bytes."""
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
