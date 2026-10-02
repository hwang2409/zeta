from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    dependencies: tuple[str, ...]
    payload: str
