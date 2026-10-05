from collections.abc import Iterable
from decimal import Decimal

from .parser import Event


def totals(events: Iterable[Event]) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        result[event.parcel] = result.get(event.parcel, Decimal()) + event.delta
    return result
