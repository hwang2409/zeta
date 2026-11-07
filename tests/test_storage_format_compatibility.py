from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.storage_format_compatibility import (
    _assert_mutation_delta,
    _digest_tree,
    _run_roundtrips,
)
from zeta.storage_format import WRITER_EPOCH


def test_selected_writer_round_trips_and_reopens_candidate_data() -> None:
    _run_roundtrips(Path(__file__).parents[1], WRITER_EPOCH)


def test_mutation_delta_rejects_non_declared_path(tmp_path: Path) -> None:
    (tmp_path / "owned.json").write_text("before")
    before = _digest_tree(tmp_path)
    (tmp_path / "owned.json").write_text("after")
    (tmp_path / "undeclared.json").write_text("changed")
    after = _digest_tree(tmp_path)
    with pytest.raises(RuntimeError, match="undeclared paths"):
        _assert_mutation_delta(before, after, ("owned.json",))


def test_digest_changes_when_higher_epoch_home_is_mutated(tmp_path: Path) -> None:
    path = tmp_path / "storage-format.json"
    path.write_text('{"writer_epoch":4}\n')
    before = _digest_tree(tmp_path)
    path.write_text('{"writer_epoch":3}\n')
    assert _digest_tree(tmp_path)["storage-format.json"] != before["storage-format.json"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _digest_tree(tmp_path)["storage-format.json"]
