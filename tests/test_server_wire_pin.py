from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract

PIN = Path(__file__).parent / "fixtures" / "serve_wire_pin.jsonl"
_ERGONOMICS_METHODS = {
    "rename_session",
    "delete_session",
    "session_tree",
    "session_history",
    "content_read",
    "switch_branch",
    "fork_message",
    "model_catalog",
    "session_settings",
    "set_settings",
    "set_compaction",
    "send_images",
    "slash_list",
    "slash_run",
    "account_list",
    "account_logout",
    "account_refresh",
    "account_switch",
}


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
    ergonomics_rows = [pin for pin in pins if pin.get("kind") == "ergonomics_valid"]
    assert len(ergonomics_rows) == len(_ERGONOMICS_METHODS)
    assert {pin["method"] for pin in ergonomics_rows} == _ERGONOMICS_METHODS
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

    for index, pin in enumerate(pin for pin in pins if pin.get("kind") is None):
        case = pin["case"]
        role, _hello, version, raw_features = case.split(" ", 3)
        features = ast.literal_eval(raw_features)
        hello_contract = await ServeContract.start(tmp_path / f"hello-{index}", "unix")
        try:
            client = await hello_contract.connect(role)  # type: ignore[arg-type]
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
            await hello_contract.close()

    groups: dict[tuple[str, str, tuple[str, ...]], list[dict[str, object]]] = {}
    for pin in pins:
        if pin.get("kind") is None:
            continue
        role = pin.get("role", "controller")
        version = pin.get("protocol_version", "1.2")
        features = tuple(pin.get("features", []))
        groups.setdefault((role, version, features), []).append(pin)
    for group_index, ((role, version, features), group) in enumerate(groups.items()):
        contract = await ServeContract.start(
            tmp_path / f"wire-pin-{group_index}", "unix"
        )
        try:
            client = await contract.connect(role)  # type: ignore[arg-type]
            hello = await client.request_raw(
                f"hello-{group_index}",
                "hello",
                {
                    "protocol_version": version,
                    "features": list(features),
                    **({"role": "observer"} if role == "observer" else {}),
                },
            )
            assert json.loads(hello).get("error") is None
            for pin in group:
                actual = await client.request_raw(
                    pin["request_id"], pin["method"], pin["params"]
                )
                if pin.get("kind") == "ergonomics_valid":
                    assert json.loads(actual) == json.loads(pin["response"])
                elif pin["method"] == "model_catalog":
                    result = json.loads(actual)["result"]
                    assert set(result) == {"models", "providers"}
                    assert isinstance(result["models"], list)
                    assert isinstance(result["providers"], dict)
                else:
                    assert actual == _w4_delta(pin["response"], pin["method"])
        finally:
            await contract.close()
