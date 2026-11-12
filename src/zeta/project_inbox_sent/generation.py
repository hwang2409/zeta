"""The recipient inbox's monotone generation counter.

A scanner reads the counter to detect inbox changes without directory stat
fingerprints. Below the durable-intent epoch no counter file exists and the
scanner still uses fingerprints, so advancing it is a no-op.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..core.session_files import (
    atomic_publish_file,
    child_directory,
    read_session_file,
)

if TYPE_CHECKING:
    from ..project_registry import ProjectRegistry


class InboxGeneration:
    """Own the read and monotone advance of one home's inbox generation files."""

    def __init__(self, registry: ProjectRegistry, enabled: bool) -> None:
        self._registry = registry
        self.enabled = enabled

    def advance(self, inbox_fd: int) -> None:
        """Bump the generation before a scanner-visible mutation.

        Below the durable-intent epoch no generation file exists and the scanner
        still uses directory stat fingerprints, so this is a no-op.
        """
        if not self.enabled:
            return
        atomic_publish_file(
            inbox_fd,
            "generation",
            str(self.read(inbox_fd) + 1).encode("ascii"),
            sync_directory=True,
        )

    @staticmethod
    def read(inbox_fd: int) -> int:
        from ..project_inbox import InboxError

        try:
            raw = read_session_file(inbox_fd, "generation")
        except FileNotFoundError:
            return 0
        try:
            text = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise InboxError("invalid inbox generation") from exc
        if not text.isdigit():
            raise InboxError("invalid inbox generation")
        return int(text)

    def project_generation(self, project_id: str) -> int:
        """Read one inbox generation for the scanner without the inbox lock.

        Uses the O_NOFOLLOW directory helpers like every other inbox read, so a
        symlinked path component cannot redirect the read.
        """
        from ..project_inbox import InboxError

        try:
            root_fd = os.open(
                os.fspath(self._registry.root),
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except (FileNotFoundError, NotADirectoryError):
            return 0
        try:
            with (
                child_directory(root_fd, project_id, create=False) as project_fd,
                child_directory(project_fd, "inbox", create=False) as inbox_fd,
            ):
                return self.read(inbox_fd)
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise InboxError("invalid inbox generation") from exc
        finally:
            os.close(root_fd)
