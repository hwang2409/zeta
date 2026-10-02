from collections.abc import Iterable
from decimal import Decimal

from .parser import Event


def totals(events: Iterable[Event]) -> dict[str, Decimal]:
    # BUG: a later retry replaces the authoritative event.
    unique = {event.event_id: event for event in events}
    result: dict[str, Decimal] = {}
    for event in unique.values():
        result[event.parcel] = result.get(event.parcel, Decimal()) + event.delta
    return result
