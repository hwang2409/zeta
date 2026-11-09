from __future__ import annotations

import asyncio
from typing import Any

from tests.support.server_client import read_sent_responses

_CLOSE_TIMEOUT = 10


async def close_serve(server: Any, *writers: Any) -> None:
    """Close a serve server before waiting for any client peers to close.

    First read every response still owed to a request the shared client sent, so a
    test cannot pass while it stopped reading before another actor's frame arrived.
    """
    first_error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None
    try:
        for writer in writers:
            await read_sent_responses(writer)
    except asyncio.CancelledError as error:
        cancellation = error
    except BaseException as error:  # noqa: BLE001
        first_error = error
    server_task = asyncio.create_task(server.close())
    try:
        await asyncio.sleep(0)
        for writer in writers:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), _CLOSE_TIMEOUT)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
    except asyncio.CancelledError as error:
        cancellation = cancellation or error
    finally:
        while not server_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(server_task), _CLOSE_TIMEOUT)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except TimeoutError as error:
                server_task.cancel()
                if first_error is None:
                    first_error = error
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        try:
            server_task.result()
        except BaseException as error:  # noqa: BLE001
            if first_error is None:
                first_error = error
    if cancellation is not None:
        if first_error is not None:
            cancellation.add_note(
                f"cleanup error: {type(first_error).__name__}: {first_error}"
            )
        raise cancellation
    if first_error is not None:
        raise first_error
