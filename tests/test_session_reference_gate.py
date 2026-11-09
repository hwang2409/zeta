from __future__ import annotations

import pytest

from zeta.core.session_files import (
    SessionReferenceGate,
    SessionReferenceUnavailable,
)
from zeta.storage_format import open_writer


def test_shared_reference_gate_publishes_generation_and_rejects_reserved_id(tmp_path):
    lease = open_writer(tmp_path)
    try:
        gate = SessionReferenceGate(tmp_path, lease)
        with gate.shared(installed_ids=("root",), referenced_ids=("parent",)):
            generation = int((tmp_path / "session-reference-generation").read_text())
            assert generation == 1

        reservation_dir = tmp_path / "session-trash" / "reservations"
        reservation_dir.mkdir(parents=True)
        (reservation_dir / "root.json").write_text("{}")
        with pytest.raises(
            SessionReferenceUnavailable,
            match=r"session reference unavailable: root is being deleted",
        ), gate.shared(installed_ids=("root",)):
            pass
    finally:
        lease.close()


def test_reference_gate_is_dormant_below_feature_epoch(tmp_path):
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":6}\n')
    lease = open_writer(tmp_path)
    try:
        gate = SessionReferenceGate(tmp_path, lease)
        with gate.shared(installed_ids=("bad/id",)):
            pass
        assert not (tmp_path / ".domain-locks").exists()
        assert not (tmp_path / "session-reference-generation").exists()
    finally:
        lease.close()
