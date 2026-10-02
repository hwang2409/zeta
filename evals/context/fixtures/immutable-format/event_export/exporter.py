import json
from collections.abc import Iterable

from .schema import Event


def export_events(events: Iterable[Event]) -> str:
    return json.dumps([event.__dict__ for event in events], indent=2, sort_keys=True)
