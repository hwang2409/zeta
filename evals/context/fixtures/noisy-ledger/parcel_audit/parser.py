from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class Event:
    event_id: str
    parcel: str
    delta: Decimal


def parse(line: str) -> Event | None:
    parts = line.strip().split("|")
    if len(parts) != 4:
        return None
    _timestamp, event_id, parcel, raw_delta = parts
    if not event_id or not parcel:
        return None
    try:
        delta = Decimal(raw_delta)
    except InvalidOperation:
        return None
    return Event(event_id, parcel, delta)
