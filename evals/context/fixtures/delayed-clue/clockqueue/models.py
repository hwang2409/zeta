from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    key: str
    created_at: int
    ttl: int
