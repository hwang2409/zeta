from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeta.accounts import AccountId, AccountStore
from zeta.mcp.oauth_store import MCPOAuthToken, save_token, token_store_path
from zeta.memory.reconciliation_state import ReconciliationState
from zeta.project_inbox import SENT_HISTORY_LIMIT, SentMessageTracker
from zeta.project_registry import ProjectRegistry
from zeta.storage_format import (
    BROAD_OWNED_SUBTREES,
    CLOSED_VARIANTS,
    StorageFormatError,
    open_writer,
    require_closed_variant,
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_project_memory_pointer_preserves_unknown_fields(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("alpha", "test project")
    registry.update_memory(project.project_id, {"state.md": "one"})
    pointer = home / "projects" / project.project_id / "memory-current.json"
    value = _read(pointer)
    value["future-field"] = {"kept": True}
    _write(pointer, value)

    registry.update_memory(project.project_id, {"state.md": "two"})

    assert _read(pointer)["future-field"] == {"kept": True}


def test_sent_ledger_preserves_unknown_record_fields_and_prunes_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zeta.project_inbox as inbox

    assert SENT_HISTORY_LIMIT > 2
    monkeypatch.setattr(inbox, "SENT_HISTORY_LIMIT", 2)
    sessions = tmp_path / "sessions"
    session_id = "1" * 32
    messages = ("2" * 32, "3" * 32, "4" * 32)
    project_id = "5" * 32
    tracker = SentMessageTracker(sessions, session_id)
    tracker.record(messages[0], project_id)
    ledger = sessions / session_id / "project-inbox-sent.json"
    value = _read(ledger)
    records = value["records"]
    assert isinstance(records, list)
    records[0]["future-field"] = "kept"
    _write(ledger, value)

    tracker = SentMessageTracker(sessions, session_id)
    tracker.mark_reported(((messages[0], "done"),))
    tracker.record(messages[1], project_id)
    value = _read(ledger)
    records = value["records"]
    assert isinstance(records, list)
    records[1]["future-field"] = "kept"
    _write(ledger, value)
    tracker = SentMessageTracker(sessions, session_id)
    tracker.record(messages[2], project_id)

    records = _read(ledger)["records"]
    assert isinstance(records, list)
    assert [record["id"] for record in records] == [messages[1], messages[2]]
    assert records[0]["future-field"] == "kept"
    assert all(record["id"] != messages[0] for record in records)


def test_reconciliation_keyed_record_preserves_unknown_fields(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    path = home / "sessions" / "session-1" / "memory-reconcile.json"
    path.parent.mkdir(parents=True)
    with open_writer(home) as lease:
        state = ReconciliationState(
            path,
            project_id="p_alpha",
            session_id="session-1",
            diagnostics_path=home / "logs" / "memory.jsonl",
            storage_lease=lease,
        )
        state.record_failure(
            seq_start=1,
            seq_end=2,
            validation_summary="failed",
            reason="test",
            usage={},
            retry_backoff_seconds=1,
            now=1,
            occurred_at="2026-01-01T00:00:00Z",
        )
        value = _read(path)
        pending = value["pending_failures"]
        assert isinstance(pending, list)
        pending[0]["future-field"] = {"kept": True}
        _write(path, value)
        state = ReconciliationState(
            path,
            project_id="p_alpha",
            session_id="session-1",
            diagnostics_path=home / "logs" / "memory.jsonl",
            storage_lease=lease,
        )
        state.record_failure(
            seq_start=1,
            seq_end=2,
            validation_summary="failed again",
            reason="test",
            usage={},
            retry_backoff_seconds=1,
            now=2,
            occurred_at="2026-01-01T00:00:01Z",
        )

    pending = _read(path)["pending_failures"]
    assert isinstance(pending, list)
    assert pending[0]["future-field"] == {"kept": True}


def test_account_registry_preserves_unknown_root_fields(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    first = AccountId("anthropic", "first")
    second = AccountId("anthropic", "second")
    with store._home() as home_fd:
        store._register_account(home_fd, first)
    registry = tmp_path / "accounts.json"
    value = _read(registry)
    value["future-field"] = {"kept": True}
    _write(registry, value)

    with store._home() as home_fd:
        store._register_account(home_fd, second)

    assert _read(registry)["future-field"] == {"kept": True}


def test_mcp_oauth_token_preserves_unknown_fields(tmp_path: Path) -> None:
    token = MCPOAuthToken(
        access_token="first",
        refresh_token="refresh",
        expires_at=None,
        token_type="Bearer",
        scope=None,
        authorization_server="https://example.test",
        token_endpoint="https://example.test/token",
        authorization_endpoint="https://example.test/auth",
        client_id="client",
        client_secret=None,
        redirect_uri="http://127.0.0.1/callback",
        resource="https://example.test/resource",
    )
    save_token("server", token, home=tmp_path)
    path = token_store_path("server", tmp_path)
    value = _read(path)
    value["future-field"] = {"kept": True}
    _write(path, value)

    save_token("server", token, home=tmp_path)

    assert _read(path)["future-field"] == {"kept": True}


def test_slice_0b_registers_closed_variants_and_broad_subtrees() -> None:
    assert CLOSED_VARIANTS["project.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["inbox.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["sent_message_ledger.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["account_registry.version"] == frozenset({1})
    assert CLOSED_VARIANTS["memory.format"] == frozenset({1, 2})
    assert BROAD_OWNED_SUBTREES["memory-current.json"] == frozenset({("history",)})
    with pytest.raises(StorageFormatError, match="unsupported stored variant"):
        require_closed_variant("inbox.schema_version", 2)
