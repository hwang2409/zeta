import json
from collections.abc import Iterable

from .schema import Event


def export_events(events: Iterable[Event]) -> str:
    return "".join(
        json.dumps(
            {"id": e.id, "type": e.type, "payload": e.payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for e in events
    )
