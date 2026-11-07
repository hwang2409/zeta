from __future__ import annotations

import asyncio
from typing import Any

_CLOSE_TIMEOUT = 10


async def close_serve(server: Any, *writers: Any) -> None:
    """Close a serve server before waiting for any client peers to close."""
    server_task = asyncio.create_task(server.close())
    await asyncio.sleep(0)
    first_error: BaseException | None = None
    try:
        for writer in writers:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), _CLOSE_TIMEOUT)
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
    finally:
        try:
            await asyncio.wait_for(asyncio.shield(server_task), _CLOSE_TIMEOUT)
        except BaseException as error:  # noqa: BLE001
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
