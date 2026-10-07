from __future__ import annotations

import json
from pathlib import Path

from zeta.transcript_search.eval import evaluate_manifest
from zeta.transcript_search.units import MAX_UNIT_BYTES, render_transcript_units


def _message(seq: int, role: str, text: str, *, metadata: dict | None = None) -> dict:
    return {
        "seq": seq,
        "id": f"row-{seq}",
        "parent_id": f"row-{seq - 1}" if seq > 1 else None,
        "lane": "main",
        "type": "message",
        "data": {
            "created_at": f"2026-01-01T00:00:{seq:02d}Z",
            "message": {
                "role": role,
                "content": [{"type": "text", "text": text}],
                "metadata": metadata or {},
            },
        },
    }


def test_turn_boundaries_authorship_tools_reports_and_secrets() -> None:
    rows = [
        _message(1, "user", "harness-created input", metadata={"origin": "harness"}),
        _message(2, "assistant", "first reply", metadata={"response_state": "completed"}),
        _message(3, "user", "real question password=hunter2", metadata={"origin": "human"}),
        {
            "seq": 4,
            "id": "row-4",
            "parent_id": "row-3",
            "lane": "main",
            "type": "message",
            "data": {"message": {"role": "assistant", "content": [{"type": "tool_use", "tool_call": {"id": "t", "name": "bash", "arguments": {"command": "pytest tests/test_x.py", "secret": "ghp_abcdefghijklmnopqrstuvwxyz"}}}], "metadata": {"response_state": "synthetic"}}},
        },
        {
            "seq": 5,
            "id": "row-5",
            "parent_id": "row-4",
            "lane": "main",
            "type": "message",
            "data": {"message": {"role": "tool_result", "content": [], "tool_result": {"tool_call_id": "t", "content": "first diagnostic\n" + "x" * 5000 + "\nlast diagnostic", "is_error": True}, "metadata": {}}},
        },
        _message(6, "assistant", "second reply", metadata={"response_state": "completed"}),
        {
            "seq": 7,
            "id": "row-7",
            "parent_id": "row-6",
            "lane": "main",
            "type": "notification",
            "data": {"kind": "agent_completion", "status": "completed", "description": "review", "text": "child found a race", "created_at": "2026-01-01T00:00:07Z"},
        },
    ]
    units = render_transcript_units("p_" + "a" * 32, "session", rows)
    turns = [unit for unit in units if unit.kind == "turn"]
    reports = [unit for unit in units if unit.kind == "child_report"]

    assert len(turns) == 2
    assert "Human:" not in turns[0].text
    assert "Input: harness-created input" in turns[0].text
    assert "Human: real question" in turns[1].text
    assert "hunter2" not in turns[1].text
    assert "Tool bash" in turns[1].text
    assert "pytest tests/test_x.py" in turns[1].text
    assert "first diagnostic" in turns[1].text
    assert "last diagnostic" in turns[1].text
    assert len(turns[1].text.encode()) <= MAX_UNIT_BYTES
    assert len(reports) == 1
    assert reports[0].seq_start == reports[0].seq_end == 7
    assert "child found a race" in reports[0].text


def test_oversized_turn_splits_deterministically() -> None:
    rows = [
        _message(1, "user", "question", metadata={"origin": "human"}),
        _message(2, "assistant", "word " * 10000, metadata={"response_state": "completed"}),
    ]
    first = render_transcript_units("p_" + "a" * 32, "session", rows)
    second = render_transcript_units("p_" + "a" * 32, "session", rows)
    assert first == second
    assert len(first) > 1
    assert {unit.turn_id for unit in first} == {first[0].turn_id}
    assert [unit.chunk_index for unit in first] == list(range(len(first)))
    assert {unit.chunk_count for unit in first} == {len(first)}
    assert all(len(unit.text.encode()) <= MAX_UNIT_BYTES for unit in first)


def test_synthetic_eval_fixture_meets_thresholds() -> None:
    root = Path(__file__).parent / "fixtures" / "transcript_search"
    result = evaluate_manifest(root / "manifest.json", root / "units.jsonl")
    assert result.query_count == 4
    assert result.recall_at[1] >= 0.75
    assert result.recall_at[5] == 1.0
    assert result.mrr >= 0.8


def test_eval_fixture_has_no_embedded_external_content() -> None:
    root = Path(__file__).parent / "fixtures" / "transcript_search"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["fixture"] == "synthetic-v1"
    assert all(item["target_unit_ids"] for item in manifest["queries"])
