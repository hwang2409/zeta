from __future__ import annotations

import os


def write_all(fd: int, payload: bytes) -> None:
    """Write the complete payload to a file descriptor."""

    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError("write returned zero bytes")
        view = view[written:]
