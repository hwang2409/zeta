from __future__ import annotations

import asyncio
from typing import Any

from zeta.core.store import ConversationStore


async def open_test_store(*args: Any, **kwargs: Any) -> ConversationStore:
    """Open a writable test store without blocking the test event loop."""

    return await asyncio.to_thread(ConversationStore, *args, **kwargs)
