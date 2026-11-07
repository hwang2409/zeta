from __future__ import annotations

import re
from pathlib import Path

import pytest

from zeta.protocol import types
from zeta.server import event_contract
from zeta.server.connections import ConnectionRegistry
from zeta.server.protocol import FrameCodec, ProtocolError

PROTOCOL_DOC = Path(__file__).parents[1] / "docs" / "serve-protocol.md"


def _section(document: str, heading: str, next_heading: str) -> str:
    return document.split(heading, 1)[1].split(next_heading, 1)[0]


def _documented_error_codes(document: str) -> dict[int, str]:
    section = _section(document, "## errors", "## normative wire schema")
    return {
        int(match.group(1)): match.group(2)
        for match in re.finditer(r"^\| `(-\d+)` \| ([^|]+?) \|$", section, re.MULTILINE)
    }


def _documented_event_names(document: str) -> set[str]:
    section = _section(document, "Event fields are:", "The `retry` data")
    names: set[str] = set()
    for row in section.splitlines():
        if not row.startswith("| `"):
            continue
        event_cell = row.split("|", 2)[1].split(" (feature", 1)[0]
        names.update(re.findall(r"`([a-z][a-z0-9_]*)`", event_cell))
    return names


def test_protocol_docs_match_every_emitted_error_code_and_meaning() -> None:
    documented = _documented_error_codes(PROTOCOL_DOC.read_text(encoding="utf-8"))
    registry = getattr(types, "PROTOCOL_ERROR_CODES", {})
    assert documented == registry, (
        "add every serve error code and its registry meaning to "
        "PROTOCOL_ERROR_CODES and docs/serve-protocol.md"
    )


def test_protocol_docs_match_every_runtime_wire_event() -> None:
    documented = _documented_event_names(PROTOCOL_DOC.read_text(encoding="utf-8"))
    registry = getattr(event_contract, "WIRE_EVENT_NAMES", frozenset())
    assert documented == registry, (
        "add every runtime wire event to its event_contract mapping and the "
        "event table in docs/serve-protocol.md"
    )


@pytest.mark.asyncio
async def test_unregistered_event_names_cannot_reach_the_wire() -> None:
    registry = object.__new__(ConnectionRegistry)
    registry._controller = None
    with pytest.raises(
        ValueError,
        match="add it to event_contract and docs/serve-protocol.md",
    ):
        await registry.publish_controller("undocumented", None)


def test_unregistered_error_codes_cannot_reach_the_wire() -> None:
    message = "add it to PROTOCOL_ERROR_CODES and docs/serve-protocol.md"
    with pytest.raises(ValueError, match=message):
        ProtocolError(-32999, "undocumented")
    with pytest.raises(ValueError, match=message):
        FrameCodec().error_response(1, -32999, "undocumented")


def test_protocol_connection_summary_includes_observers() -> None:
    document = PROTOCOL_DOC.read_text(encoding="utf-8")
    assert "The server accepts one client" not in document
    assert "one controller plus up to eight negotiated read-only\nobservers" in document
