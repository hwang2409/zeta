import asyncio
import json
import math

import httpx
import pytest

from zeta.providers.codex_payload import _CodexImageURL
from zeta.providers.codex_request import prepare_large_image_request


def _payload(*values: str) -> dict[str, object]:
    return {"input": [{"type": "input_image", "image_url": value} for value in values]}


def _large_image() -> str:
    return "data:image/jpeg;base64," + "A" * (13 * 1024 * 1024)


@pytest.mark.asyncio
async def test_large_codex_body_is_exact_replayable_json() -> None:
    image = _large_image()
    payload = _payload(_CodexImageURL(image), _CodexImageURL(image))
    prepared = prepare_large_image_request(payload)
    assert prepared is not None
    expected = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    first = b"".join([part async for part in prepared.content])
    second = b"".join([part async for part in prepared.content])
    assert first == expected == second
    assert prepared.headers["Content-Length"] == str(len(expected))


@pytest.mark.asyncio
async def test_large_codex_body_uses_known_length_without_transfer_encoding() -> None:
    image = _large_image()
    payload = _payload(_CodexImageURL(image), _CodexImageURL(image))
    prepared = prepare_large_image_request(payload)
    assert prepared is not None
    request = httpx.Request("POST", "https://example.test", headers=prepared.headers, content=prepared.content)
    assert request.headers["content-length"] == prepared.headers["Content-Length"]
    assert "transfer-encoding" not in request.headers


def test_only_marked_large_images_use_special_path() -> None:
    image = _large_image()
    assert prepare_large_image_request(_payload(image, image)) is None
    assert prepare_large_image_request(_payload(_CodexImageURL(image))) is None
    assert prepare_large_image_request(_payload(_CodexImageURL(image), _CodexImageURL(image))) is not None


def test_small_images_and_nan_keep_standard_behavior() -> None:
    assert prepare_large_image_request(_payload(_CodexImageURL("data:image/jpeg;base64,A"))) is None
    payload = _payload(_CodexImageURL(_large_image()), _CodexImageURL(_large_image()))
    payload["bad"] = math.nan
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        prepared = prepare_large_image_request(payload)
        assert prepared is not None


@pytest.mark.asyncio
async def test_cancellation_does_not_corrupt_replay() -> None:
    image = _large_image()
    prepared = prepare_large_image_request(_payload(_CodexImageURL(image), _CodexImageURL(image)))
    assert prepared is not None

    async def consume() -> None:
        async for _ in prepared.content:
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    replay = b"".join([part async for part in prepared.content])
    assert len(replay) == int(prepared.headers["Content-Length"])
