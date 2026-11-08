from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeContract

PIN = Path(__file__).parent / "fixtures" / "serve_wire_pin.jsonl"
PIN_BASE_SHA = "ded830c144474addcf17f950a55540912cb37c53"
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


def test_wire_pin_was_captured_from_base() -> None:
    header = json.loads(PIN.read_text().splitlines()[0])
    metadata, pins = _pin()
    _assert_fixture_shape(metadata, pins)
    command = header.get("command", "the base-worktree capture command")
    captured_at = header.get("captured_at")
    assert captured_at == PIN_BASE_SHA, (
        f"wire pin capture commit must be {PIN_BASE_SHA}; rerun {command}"
    )
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert captured_at != current, (
        f"wire pin must not be captured from the head under test; rerun {command}"
    )
    if shutil.which("git") is not None:
        assert (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", PIN_BASE_SHA, current],
                check=False,
            ).returncode
            == 0
        ), f"{PIN_BASE_SHA} must be an ancestor of HEAD; rerun {command}"


def _normalize_response(raw: str, active_session_id: str | None = None) -> str:
    if active_session_id is not None:
        raw = raw.replace(active_session_id, "$ACTIVE_SESSION")
    message = json.loads(raw)
    result = message.get("result")
    if isinstance(result, dict) and isinstance(result.get("providers"), dict):
        result["providers"] = dict(sorted(result["providers"].items()))
    return json.dumps(message, separators=(",", ":")) + "\n"


def _assert_fixture_shape(
    metadata: dict[str, object], pins: list[dict[str, object]]
) -> None:
    assert pins
    ergonomics_rows = [pin for pin in pins if pin.get("kind") == "ergonomics_valid"]
    assert len(ergonomics_rows) == len(_ERGONOMICS_METHODS) + 12
    assert {pin["method"] for pin in ergonomics_rows} == _ERGONOMICS_METHODS
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


@pytest.mark.asyncio
async def test_wire_pin_hello_rows(tmp_path: Path) -> None:
    _, pins = _pin()
    for index, pin in enumerate(pin for pin in pins if pin.get("kind") is None):
        role, _hello, version, raw_features = pin["case"].split(" ", 3)
        contract = await ServeContract.start(tmp_path / f"hello-{index}", "unix")
        try:
            client = await contract.connect(role)  # type: ignore[arg-type]
            actual = await client.request_raw(
                "o-hello" if role == "observer" else "c-hello",
                "hello",
                {
                    "protocol_version": version,
                    "features": ast.literal_eval(raw_features),
                    **({"role": "observer"} if role == "observer" else {}),
                },
            )
            assert actual == pin["response"]
        finally:
            await contract.close()


@pytest.mark.asyncio
async def test_real_unix_wire_pin_allows_only_w4_deltas(tmp_path: Path) -> None:
    metadata, pins = _pin()
    _assert_fixture_shape(metadata, pins)
    groups: dict[tuple[str, str, tuple[str, ...]], list[dict[str, object]]] = {}
    for pin in pins:
        if pin.get("kind") is None:
            continue
        key = (
            pin.get("role", "controller"),
            pin.get("protocol_version", "1.2"),
            tuple(pin.get("features", [])),
        )
        groups.setdefault(key, []).append(pin)
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
            active_session_id: str | None = None
            for pin in group:
                params = dict(pin["params"])
                if params.get("session_id") == "$ACTIVE_SESSION":
                    if active_session_id is None:
                        created = await client.request_raw(
                            "active-session", "new_session", {"provider": "codex"}
                        )
                        active_session_id = json.loads(created)["result"]["session"][
                            "session_id"
                        ]
                    params["session_id"] = active_session_id
                actual = await client.request_raw(
                    pin["request_id"], pin["method"], params
                )
                expected = _w4_delta(pin["response"], pin["method"])
                assert _normalize_response(
                    actual, active_session_id
                ) == _normalize_response(expected, active_session_id)
        finally:
            await contract.close()
