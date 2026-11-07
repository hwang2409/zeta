from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract

PIN = Path(__file__).parent / "fixtures" / "serve_wire_pin.jsonl"


def _pin() -> list[dict[str, str]]:
    return [json.loads(line) for line in PIN.read_text().splitlines()[1:]]


def _w4_delta(raw: str, case: str) -> str:
    if case == "clear_steering":
        return raw.replace('"code":-32602', '"code":-32601', 1)
    if case == "delivery_status":
        return raw.replace('"code":-32602', '"code":-32601', 1)
    return raw


@pytest.mark.asyncio
async def test_real_unix_wire_pin_allows_only_w4_deltas(tmp_path: Path) -> None:
    pins = _pin()
    assert pins
    assert (
        PIN.read_text().splitlines()[0]
        == '{"fixture":"serve_wire_pin","base_sha":"e8e77882"}'
    )
    for index, pin in enumerate(pins[:8]):
        version = pin["case"].split()[2]
        features = json.loads(pin["case"].split(" ", 3)[3].replace("'", '"'))
        contract = await ServeContract.start(tmp_path / f"hello-{index}", "unix")
        try:
            role = "observer" if pin["case"].startswith("observer") else "controller"
            client = await contract.connect(role)
            actual = await client.request_raw(
                "o-hello" if role == "observer" else "c-hello",
                "hello",
                {
                    "protocol_version": version,
                    "features": features,
                    **({"role": "observer"} if role == "observer" else {}),
                },
            )
            assert actual == pin["response"]
        finally:
            await contract.close()

    contract = await ServeContract.start(tmp_path / "requests", "unix")
    try:
        controller = await contract.connect("controller")
        await controller.request(
            "c-hello", "hello", {"protocol_version": "1.1", "features": []}
        )
        for pin in pins[8:11]:
            case = pin["case"]
            if case == "observer unknown":
                observer = await contract.connect("observer")
                await observer.request(
                    "o-hello",
                    "hello",
                    {
                        "protocol_version": "1.2",
                        "role": "observer",
                        "features": ["serve_observers"],
                    },
                )
                actual = await observer.request_raw("ou", "not_a_method")
            else:
                actual = await controller.request_raw(
                    {
                        "not_a_method": "unknown",
                        "clear_steering": "clear",
                        "delivery_status": "delivery",
                    }[case],
                    case,
                )
            assert actual == _w4_delta(pin["response"], case)
    finally:
        await contract.close()
