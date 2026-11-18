from __future__ import annotations

import sqlite3

import pytest

from zeta.telemetry.viewer.store import ViewerStore


def test_store_deduplicates_and_projects(tmp_path):
    store = ViewerStore(tmp_path / "telemetry.sqlite3")
    row = {"trace_id": b"1" * 16, "span_id": b"2" * 8, "parent_span_id": None, "operation": "turn", "kind": 1, "start_ns": 1, "end_ns": 2, "duration_ns": 1, "status": "ok", "error_type": None, "session_id": "a" * 32, "root_session_id": "a" * 32, "service_version": "0.1.0", "scope_version": "0.1.0"}
    assert store.write([row, row]) == 1
    assert store.sessions()["sessions"][0]["span_count"] == 1
    assert store.trace("31" * 16)["trace"]["span_count"] == 1


def test_schema_rejects_invalid_span_identity(tmp_path):
    store = ViewerStore(tmp_path / "telemetry.sqlite3")
    row = {"trace_id": b"1", "span_id": b"2" * 8, "parent_span_id": None, "operation": "turn", "kind": 1, "start_ns": 1, "end_ns": 2, "duration_ns": 1, "status": "ok", "service_version": "0.1.0", "scope_version": "0.1.0"}
    with pytest.raises(sqlite3.IntegrityError):
        store.write([row])
