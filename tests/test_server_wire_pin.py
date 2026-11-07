from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract

PIN = Path(__file__).parent / "fixtures" / "serve_wire_pin.jsonl"


def _pin() -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = [json.loads(line) for line in PIN.read_text().splitlines()[1:]]
    metadata = next(row for row in rows if row.get("kind") == "method_sets")
    return metadata, [row for row in rows if row.get("kind") != "method_sets"]


def _w4_delta(raw: str, method: str) -> str:
    if method in {"clear_steering", "delivery_status"} and (
        "requires the negotiated" in raw or "requires negotiated feature" in raw
    ):
        return raw.replace('"code":-32602', '"code":-32601', 1)
    return raw


@pytest.mark.asyncio
async def test_real_unix_wire_pin_allows_only_w4_deltas(tmp_path: Path) -> None:
    metadata, pins = _pin()
    assert pins
    assert PIN.read_text().splitlines()[0] == (
        '{"fixture":"serve_wire_pin","base_sha":"e8e77882"}'
    )
    assert {pin.get("kind") for pin in pins} >= {
        "mutation_observer",
        "mutation_unhealthy",
        "mutation_protocol",
        "delivery_observer",
        "delivery_unhealthy",
        "delivery_protocol",
        "observer_request",
    }
    for kind, methods in (
        ("observer_request", metadata["observer"]),
        ("mutation_observer", metadata["loaded"]),
        ("mutation_unhealthy", metadata["loaded"]),
        ("mutation_protocol", metadata["loaded"]),
        ("delivery_observer", metadata["delivery"]),
        ("delivery_unhealthy", metadata["delivery"]),
        ("delivery_protocol", metadata["delivery"]),
    ):
        rows = [pin for pin in pins if pin.get("kind") == kind]
        assert len(rows) == len(methods), kind
        assert {pin["method"] for pin in rows} == set(methods), kind

    contract = await ServeContract.start(tmp_path / "wire-pin", "unix")
    try:
        for index, pin in enumerate(pins):
            if pin.get("kind") is None:
                case = pin["case"]
                role, _hello, version, raw_features = case.split(" ", 3)
                features = ast.literal_eval(raw_features)
                request_id = "o-hello" if role == "observer" else "c-hello"
                client = await contract.connect(role)  # type: ignore[arg-type]
                actual = await client.request_raw(
                    request_id,
                    "hello",
                    {
                        "protocol_version": version,
                        "features": features,
                        **({"role": "observer"} if role == "observer" else {}),
                    },
                )
                assert actual == pin["response"]
                await client.close()
                continue
            role = pin.get("role", "controller")
            client = await contract.connect(role)  # type: ignore[arg-type]
            hello = await client.request_raw(
                f"hello-{index}",
                "hello",
                {
                    "protocol_version": pin.get("protocol_version", "1.2"),
                    "features": pin.get("features", []),
                    **({"role": "observer"} if role == "observer" else {}),
                },
            )
            if pin.get("kind") == "direct" and pin["case"] in {
                "not_a_method",
                "clear_steering",
                "delivery_status",
                "observer unknown",
            }:
                actual = await client.request_raw(
                    pin["request_id"], pin["method"], pin["params"]
                )
                assert actual == _w4_delta(pin["response"], pin["method"])
                await client.close()
                continue
            if pin.get("kind") is None:
                assert hello == pin["response"]
                continue
            actual = await client.request_raw(
                pin["request_id"], pin["method"], pin["params"]
            )
            if pin["method"] == "model_catalog":
                result = json.loads(actual)["result"]
                assert set(result) == {"models", "providers"}
                assert isinstance(result["models"], list)
                assert isinstance(result["providers"], dict)
            else:
                assert actual == _w4_delta(pin["response"], pin["method"])
            await client.close()
    finally:
        await contract.close()
