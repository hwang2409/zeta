from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _row(seq: int, row_type: str, data: dict[str, object], *, entry_id: str) -> dict[str, object]:
    return {
        "seq": seq,
        "id": entry_id,
        "parent_id": None if seq == 1 else f"row-{seq - 1}",
        "lane": "main",
        "type": row_type,
        "data": data,
    }


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/scan_step_delivery_sessions.py",
            "--root",
            str(root),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_scan_reports_valid_legacy_and_version_one_records(tmp_path: Path) -> None:
    session = tmp_path / "root" / "agents" / "1"
    session.mkdir(parents=True)
    notification = {
        "notification_id": "notice-1",
        "kind": "task_exited",
        "task_id": "task-1",
        "headline": "printf done",
        "exit_code": 0,
        "output_tail": "done",
        "origin": "notification",
        "background_owner": "run_background",
        "background_phase": "natural_exit",
    }
    payload = [notification]
    from zeta.agent._delivery_notifications import notification_batch_id

    identity = {
        "version": 1,
        "batch_id": notification_batch_id(payload, ["notice-1"]),
        "notification_ids": ["notice-1"],
    }
    prompt = _row(1, "pending_prompt", {"text": "follow up", "origin": "agent_send"}, entry_id="prompt-1")
    tagged = _row(2, "message", {"message": {"role": "user", "content": [{"type": "text", "text": "follow up"}], "metadata": {"zeta.origin": "agent_send", "zeta.pending_prompt_id": "prompt-1"}}}, entry_id="row-2")
    tagged["parent_id"] = "prompt-1"
    source = _row(3, "notification", {key: value for key, value in notification.items() if key != "notification_id"}, entry_id="notice-1")
    source["parent_id"] = "row-2"
    staged = _row(4, "message", {"message": {"role": "system", "content": [{"type": "text", "text": "background context"}], "metadata": {"zeta_event": "agent_notifications", "notifications": payload, "zeta.step_delivery": identity}}}, entry_id="row-4")
    staged["parent_id"] = "notice-1"
    rows = [{"type": "header", "data": {}}, prompt, tagged, source, staged]
    (session / "conversation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["schema"] == "zeta.step-delivery-scan.v1"
    assert summary["files_scanned"] == 1
    assert summary["child_session_files"] == 1
    assert summary["counts"]["prompt_tagged"] == 1
    assert summary["counts"]["notification_batches_unacknowledged"] == 1
    assert summary["anomalies"] == []
    assert "follow up" not in result.stdout
    assert "printf done" not in result.stdout


def test_scan_rejects_each_integrity_anomaly(tmp_path: Path) -> None:
    cases = {
        "invalid-json": ("not-json\n", 2, "invalid_json"),
        "dangling-prompt": (
            json.dumps(_row(1, "message", {"message": {"role": "user", "content": [], "metadata": {"zeta.pending_prompt_id": "missing"}}}, entry_id="row-1")) + "\n",
            1,
            "dangling_prompt_tag",
        ),
        "unknown-version": (
            json.dumps(_row(1, "message", {"message": {"role": "system", "content": [], "metadata": {"zeta_event": "agent_notifications", "notifications": [], "zeta.step_delivery": {"version": 2, "batch_id": "sha256:" + "0" * 64, "notification_ids": []}}}}, entry_id="row-1")) + "\n",
            1,
            "unknown_step_delivery_version",
        ),
    }
    for name, (text, exit_code, diagnostic) in cases.items():
        root = tmp_path / name
        session = root / "session"
        session.mkdir(parents=True)
        (session / "conversation.jsonl").write_text(text)
        result = _run(root)
        assert result.returncode == exit_code
        assert diagnostic in result.stderr
        assert len(result.stderr) < 1000
