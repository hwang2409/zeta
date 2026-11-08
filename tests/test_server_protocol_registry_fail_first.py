from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract


@pytest.mark.asyncio
async def test_unnegotiated_clear_steering_uses_w4_method_not_found(
    tmp_path: Path,
) -> None:
    contract = await ServeContract.start(tmp_path, "unix")
    client = None
    try:
        client = await contract.connect("controller")
        await client.request(
            "hello", "hello", {"protocol_version": "1.1", "features": []}
        )
        reply = await client.request("clear", "clear_steering")
        assert reply["error"]["code"] == -32601
        assert (
            reply["error"]["message"]
            == "clear_steering requires the negotiated abort_scope feature"
        )
    finally:
        await contract.close()
