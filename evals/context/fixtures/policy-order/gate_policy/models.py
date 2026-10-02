from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    prefix: str
    effect: str
    reason: str
