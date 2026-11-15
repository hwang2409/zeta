from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract
from zeta.server.protocol import EventDescriptor, FrameCodec

PIN = Path(__file__).parent / "fixtures" / "serve_wire_pin.jsonl"


def _pin() -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = [json.loads(line) for line in PIN.read_text().splitlines()[1:]]
    metadata = next(row for row in rows if row.get("kind") == "method_sets")
    return metadata, [row for row in rows if row.get("kind") != "method_sets"]


def test_session_retention_event_codec_pin() -> None:
    row = next(json.loads(line) for line in PIN.read_text().splitlines() if 'session_retention_event' in line)
    projection = {
        "timestamp": "2026-01-01T00:00:00+00:00", "mode": "preview", "days": 30,
        "armed_by_zeta_version": None, "preview_id": "preview",
        "cutoff": "2025-12-02T00:00:00+00:00", "root_ids": ["root"],
        "count": 1, "descendant_count": 0, "bytes": 12, "skip_counts": {},
        "progress_counts": {}, "errors": [], "disable_pending": False,
    }
    frame = FrameCodec().notification(EventDescriptor.SESSION_RETENTION, "session", data=projection)
    assert frame is not None
    assert frame.decode() == row["frame"]


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
        '{"fixture":"serve_wire_pin","base_sha":"3e1f48a3"}'
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
        if pin.get("kind") is None or pin.get("kind") == "session_retention_event":
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
                assert actual == _w4_delta(pin["response"], pin["method"])
        finally:
            await contract.close()
