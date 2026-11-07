from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..client import MCPClient


@dataclass(frozen=True, slots=True)
class TransportFailure:
    operation: int | None
    client: MCPClient
    reason: str


@dataclass(frozen=True, slots=True)
class ChildFinished:
    task: asyncio.Task[object]


@dataclass(frozen=True, slots=True)
class CancelOperation:
    identifier: int
    acknowledged: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class Remove:
    result: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class RemoveWhenIdle:
    result: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class Close:
    result: asyncio.Future[None]
