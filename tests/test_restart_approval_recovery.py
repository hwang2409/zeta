from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.core.session import SessionManager
from zeta.protocol.types import ToolCall
from zeta.server.runtime import ServerRuntime
from zeta.storage_format import (
    FEATURE_EPOCHS,
    StorageFeature,
)

_RESTART_EPOCH = FEATURE_EPOCHS[StorageFeature.APPROVAL_RESTART_REASON]


def _runtime(home: Path, cwd: Path) -> ServerRuntime:
    return ServerRuntime(
        home,
        cwd=cwd,
        provider="codex",
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )


def _seed_foreground_approval(home: Path, cwd: Path, *, epoch: int | None = None) -> str:
    if epoch is not None:
        # Publish an older home manifest before the home is first opened, so the
        # writer lease observes that epoch for the whole run.
        home.mkdir(parents=True, exist_ok=True)
        (home / "storage-format.json").write_text(
            json.dumps({"writer_epoch": epoch}) + "\n"
        )
    opened = SessionManager(home).create(provider="codex", model="offline", cwd=cwd)
    store = opened.store
    store.append_approval_request("call-1", ToolCall("call-1", "bash", {"command": "true"}))
    session_id = store.session_id
    store.close()
    return session_id


async def _resume_and_close(home: Path, cwd: Path, session_id: str) -> int:
    runtime = _runtime(home, cwd)
    await runtime.resume_session(session_id)
    try:
        assert runtime.loop is not None
        return runtime.loop.manager.observed_storage_epoch
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_restart_recovery_high_epoch_records_server_restarted(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_foreground_approval, home, tmp_path)

    observed = await _resume_and_close(home, tmp_path, session_id)
    assert observed >= _RESTART_EPOCH

    def read_resolution() -> dict[str, str] | None:
        store = SessionManager(home).open(session_id).store
        try:
            return store.approval_resolution_data("call-1")
        finally:
            store.close()

    resolution = await asyncio.to_thread(read_resolution)
    # Fails on base: before this change startup never resolved the approval, so
    # approval_resolution_data stays None and this assertion raises.
    assert resolution is not None
    assert resolution["decision"] == "abort"
    assert resolution["reason"] == "server_restarted"


@pytest.mark.asyncio
async def test_restart_recovery_low_epoch_records_reasonless_abort(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    low = _RESTART_EPOCH - 1
    session_id = await asyncio.to_thread(
        _seed_foreground_approval, home, tmp_path, epoch=low
    )

    observed = await _resume_and_close(home, tmp_path, session_id)
    assert observed == low

    def read_resolution() -> dict[str, str] | None:
        store = SessionManager(home).open(session_id).store
        try:
            return store.approval_resolution_data("call-1")
        finally:
            store.close()

    resolution = await asyncio.to_thread(read_resolution)
    assert resolution is not None
    assert resolution["decision"] == "abort"
    # Old readers reject server_restarted below the feature epoch; a reasonless
    # abort is valid for them.
    assert "reason" not in resolution
