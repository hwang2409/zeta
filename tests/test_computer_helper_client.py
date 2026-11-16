"""Protocol framing + error mapping of the computer-use helper client against a fake helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zeta.tools.computer.helper_client import (
    HelperClient,
    HelperError,
    HelperProtocolError,
    HelperTimeout,
)

FAKE = Path(__file__).parent / "fixtures" / "computer_fake_helper.py"


def make_client(timeout: float = 5.0) -> HelperClient:
    return HelperClient([sys.executable, str(FAKE)], timeout=timeout)


def test_hello_round_trip() -> None:
    with make_client() as client:
        reply = client.request("hello")
    assert reply == {"version": "fake", "screen_recording": False, "accessibility": False}


def test_sequential_requests_match_ids() -> None:
    with make_client() as client:
        assert client.request("echo", value=1) == {"value": 1}
        assert client.request("echo", value="two") == {"value": "two"}
        assert client.request("state")["frontmost_bundle_id"] == "dev.zeta.fake"


@pytest.mark.parametrize(
    "code", ["locked", "secure_input", "not_permitted", "killed", "bad_request", "internal"]
)
def test_error_code_maps_to_exception(code: str) -> None:
    with make_client() as client, pytest.raises(HelperError) as excinfo:
        client.request("fail", code=code, message="nope")
    assert excinfo.value.code == code
    assert excinfo.value.message == "nope"


def test_large_payload_spanning_chunks() -> None:
    with make_client() as client:
        reply = client.request("big", size=500000)
    assert reply["blob"] == "x" * 500000


def test_timeout_does_not_hang() -> None:
    with make_client(timeout=0.2) as client, pytest.raises(HelperTimeout):
        client.request("block")


def test_crash_raises_protocol_error() -> None:
    with make_client() as client, pytest.raises(HelperProtocolError):
        client.request("crash")


def test_garbage_line_raises_protocol_error() -> None:
    with make_client() as client, pytest.raises(HelperProtocolError):
        client.request("garbage")


def test_kill_stays_killed_for_later_requests() -> None:
    with make_client() as client:
        assert client.request("kill") == {}
        with pytest.raises(HelperError, match="kill switch engaged") as excinfo:
            client.request("echo", value="later")
        assert excinfo.value.code == "killed"


def test_close_terminates_process() -> None:
    client = make_client()
    client.start()
    client.request("hello")
    client.close()
    assert client._proc is None


def test_request_before_start_errors() -> None:
    client = make_client()
    with pytest.raises(RuntimeError):
        client.request("hello")


def test_partial_line_at_eof_raises_protocol_error() -> None:
    with make_client() as client, pytest.raises(HelperProtocolError):
        client.request("partial")


def test_id_mismatch_is_protocol_error() -> None:
    with make_client() as client, pytest.raises(HelperProtocolError):
        client.request("wrong_id")


@pytest.mark.parametrize("op", ["bad_shape", "bad_error"])
def test_malformed_response_is_protocol_error(op: str) -> None:
    with make_client() as client, pytest.raises(HelperProtocolError):
        client.request(op)
