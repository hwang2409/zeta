from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    payload: dict[str, object]
