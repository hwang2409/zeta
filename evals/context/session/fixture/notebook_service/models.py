from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    slug: str
    title: str
    body: str
