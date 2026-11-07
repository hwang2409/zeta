from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.test_attention import _session
from zeta.attention_decisions import open_decisions
from zeta.attention_records import AttentionStore
from zeta.skills import SkillCatalog
from zeta.storage_format import (
    StorageFormatError,
    advance,
    require_closed_variant,
)
from zeta.tools.registry import ToolRegistry
from zeta.tui.cards.decisions import DecisionsPanel


def _record(store: AttentionStore, *, kind: str = "decision", **kwargs):
    values = dict(
        session_id=kwargs.pop("session_id", "a" * 32),
        project_id=None,
        entry_id=None,
        entry_seq=None,
        title=kwargs.pop("title", "topic"),
        why="context",
        **kwargs,
    )
    try:
        record = store.request(**values, kind=kind)
    except TypeError:
        record = store.request(**values)
        if kind != "decision":
            path = store.directory / f"{record.id}.json"
            value = json.loads(path.read_text())
            value["kind"] = kind
            path.write_text(json.dumps(value))
            record = store.get(record.id)
    return record


def test_old_attention_bytes_read_as_decision_and_remain_unchanged(
    tmp_path: Path,
) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    record = _record(store, session_id=opened.store.session_id)
    path = store.directory / f"{record.id}.json"
    value = json.loads(path.read_text())
    assert "kind" not in value
    assert store.get(record.id).kind == "decision"
    assert (
        path.read_bytes()
        == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    opened.store.close()


def test_attention_kinds_sort_decisions_first_and_render_kind(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    answer = _record(store, kind="answer", session_id=opened.store.session_id)
    decision = _record(store, session_id=opened.store.session_id)
    items = open_decisions(tmp_path, opened.store.session_id)
    assert [item.record.kind for item in items] == ["decision", "answer"]
    panel = DecisionsPanel()
    panel.set_items(items)
    panel.move(1)
    rendered = "\n".join(text for line in panel.render_lines() for _style, text in line)
    assert "answer" in rendered and "Kind" in rendered and "Topic" in rendered
    assert decision.id != answer.id
    opened.store.close()


def test_information_item_dismisses_without_reply(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    record = _record(store, kind="answer", session_id=opened.store.session_id)
    dismissed, already = store.dismiss(record.id, session_id=opened.store.session_id)
    assert not already
    assert dismissed.status == "resolved" and dismissed.decision is None
    assert store.dismiss(record.id, session_id=opened.store.session_id)[1]
    opened.store.close()


def test_decision_cannot_be_dismissed(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    record = _record(store, session_id=opened.store.session_id)
    with pytest.raises(ValueError, match="require a reply"):
        store.dismiss(record.id, session_id=opened.store.session_id)
    opened.store.close()


def test_request_attention_kind_rules_use_registered_handler(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    registry = ToolRegistry(
        tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
    )
    handler = registry.definitions_by_name["request_attention"].handler
    result = asyncio.run(
        handler({"title": "answer", "why": "context", "kind": "answer"})
    )
    assert result["structuredContent"]["kind"] == "answer"
    with pytest.raises(ValueError, match="cannot have options"):
        asyncio.run(
            handler(
                {"title": "bad", "why": "context", "kind": "deferred", "options": ["x"]}
            )
        )
    asyncio.run(registry.close())
    opened.store.close()


def test_non_decision_creation_waits_for_shared_cutover(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    session = _session(tmp_path)
    try:
        store = AttentionStore(session.store.session_dir, writer_epoch=2)
    except TypeError:
        store = AttentionStore(session.store.session_dir)
    decision = _record(store, session_id=session.store.session_id)
    assert store.get(decision.id).kind == "decision"
    with pytest.raises(
        ValueError,
        match="zeta storage-format advance --expected-epoch 2 --target-epoch 3",
    ):
        _record(store, kind="answer", session_id=session.store.session_id)
    session.store.close()


def test_attention_kind_closed_variant_is_fail_closed(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    record = _record(store, session_id=opened.store.session_id)
    path = store.directory / f"{record.id}.json"
    value = json.loads(path.read_text())
    value["kind"] = "future"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, StorageFormatError)):
        store.get(record.id)
    with pytest.raises(StorageFormatError):
        require_closed_variant("attention.kind", "future")
    opened.store.close()


def test_information_items_reject_choice_affordances(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    store = AttentionStore(opened.store.session_dir)
    with pytest.raises(ValueError, match="cannot have options"):
        _record(
            store, kind="decided", session_id=opened.store.session_id, options=["x"]
        )
    with pytest.raises(ValueError, match="cannot have options"):
        _record(
            store,
            kind="deferred",
            session_id=opened.store.session_id,
            recommendation="x",
        )
    opened.store.close()
