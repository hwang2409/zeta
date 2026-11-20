from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.serve_close import close_serve
from tests.test_server import _connect, _request
from tests.test_server_observers import _observer
from zeta.accounts import AccountStore
from zeta.server import ZetaServer


@pytest.mark.parametrize("negotiated", [False, True])
async def test_account_list_error_contract(tmp_path: Path, monkeypatch, negotiated: bool) -> None:
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(tmp_path))
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)

    async def request(method, params=None):
        return (await _request(reader, writer, method, method, params))[-1]

    features = ["account_list_idle_v1"] if negotiated else []
    writers = [writer]
    try:
        hello = (await request("hello", {"protocol_version": "1.2", "features": features}))["result"]
        assert ("account_list_idle_v1" in hello["capabilities"]["features"]) is negotiated
        idle = await request("account_list")
        if negotiated:
            assert idle["result"]["data"] == {"accounts": []}
            assert len(idle["result"]["revision"]) == 64
        else:
            assert idle["error"]["code"] == -32602
            missing = await request("account_list", {"session_id": "not-active"})
            assert missing["error"]["code"] == -32003

        observer, observer_writer = await _observer(server, features=["serve_observers", *features])
        writers.append(observer_writer)
        for method in ("account_list", "login_start"):
            error = (await _request(observer, observer_writer, method, method))[-1]["error"]
            data = {"code": "observer_read_only", "method": method}
            if negotiated:
                data["reason"] = "observer_read_only"
            assert error == {
                "code": -32008,
                "message": "method is unavailable to read-only observers",
                "data": data,
            }

        params = {}
        if not negotiated:
            created = await request("new_session", {"provider": "codex"})
            params["session_id"] = created["result"]["session"]["session_id"]
            loop = server.runtime.loop
            with patch.object(loop, "account_context", None):
                error = (await request("account_list", params))["error"]
                assert error == {"code": -32005, "message": "accounts unavailable for this session"}
            owner = patch.object(type(loop.account_context), "list_snapshot")
        else:
            owner = patch.object(AccountStore, "catalog_snapshot")

        with owner as read:
            for failure, code, message in (
                (RuntimeError, -32000, "unexpected server or storage failure" if negotiated else "owner failure"),
                (ValueError, -32005 if negotiated else -32602,
                 "stored state is not available to read" if negotiated else "owner failure"),
                (OSError, -32005 if negotiated else -32000,
                 "stored state is not available to read" if negotiated else "owner failure"),
            ):
                read.side_effect = failure("owner failure")
                error = (await request("account_list", params))["error"]
                expected = {"code": code, "message": message}
                if code == -32005:
                    expected["data"] = {"code": "state_unavailable"}
                assert error == expected
    finally:
        await close_serve(server, *writers)
