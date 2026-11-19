from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.credentials import named_account_credentials
from tests.support.serve_contract import ServeContract
from zeta.accounts import AccountId, AccountStore
from zeta.oauth import OAuthTokens


@pytest.mark.asyncio
async def test_scripted_reply_after_account_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep the turn offline even if the switch mistakenly builds a real backend.
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("no_proxy", "localhost")
    home = tmp_path / "home"
    for name in ("first", "second"):
        named_account_credentials(home, "codex", name).save(
            OAuthTokens(f"synthetic-{name}", "synthetic-refresh", 9_999_999_999)
        )
    await asyncio.to_thread(AccountStore(home).set_default, AccountId("codex", "first"))

    contract = await ServeContract.start(tmp_path, "unix")
    try:
        client = await contract.connect()
        hello = await client.request("hello", "hello", {"protocol_version": "1.2"})
        assert hello["result"]["server"] == "zeta"
        created = await client.request("new", "new_session", {"provider": "codex"})
        session = {"session_id": created["result"]["session"]["session_id"]}
        accounts = await client.request("accounts", "account_list", session)
        assert [
            item["account"] for item in accounts["result"]["accounts"] if item["current"]
        ] == ["codex:first"]
        switched = await client.request(
            "switch", "account_switch", {**session, "account": "codex:second"}
        )
        assert switched["result"] == {
            "account": "codex:second",
            "replay": False,
            "account_state": "ready",
        }
        prompt = "prompt after account switch"
        sent = await client.request("send", "send", {**session, "text": prompt})
        assert sent["result"]["accepted"] is True
        await client.event("agent_end")
        history = await client.request("history", "session_history", session)
        replies = [
            block["text"]
            for message in history["result"]["messages"]
            if message["role"] == "assistant"
            for block in message["content"]
            if block["type"] == "text"
        ]
        assert replies == [f"you said: {prompt}"]
    finally:
        await contract.close()
