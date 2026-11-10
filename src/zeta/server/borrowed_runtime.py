"""A non-owning view of a TUI runtime for hosted observers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime.loop import AgentLoop


@dataclass(slots=True)
class BorrowedRuntime:
    """Expose the runtime values an observer host may inspect without closing them."""

    loop: AgentLoop
    home: Path
    session_id: str
    metadata: Any
    policy: Any

    @property
    def opened(self) -> Any:
        return self.metadata

    async def close(self) -> None:
        """Borrowers never own the loop, store, or session lease."""
        return
