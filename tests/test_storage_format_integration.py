from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeta.cli.main import main
from zeta.core.checkpoints import ConversationEntry, ConversationIntegrityError
from zeta.core.session import SessionManager


def _inject(path: Path) -> None:
    value = json.loads(path.read_text())
    value["future"] = {"nested": {"value": 7}}
    path.write_text(json.dumps(value))


def _assert_future(path: Path) -> None:
    assert json.loads(path.read_text())["future"] == {"nested": {"value": 7}}


def test_every_session_mapping_writer_preserves_unknown_nested_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    session_dir = store.session_dir

    metadata_path = session_dir / "meta.json"
    state_path = session_dir / "session_state.json"
    _inject(metadata_path)
    _inject(state_path)
    manager.touch(opened.metadata)
    store.set_bash_cwd(tmp_path / "next")
    _assert_future(metadata_path)
    _assert_future(state_path)

    store.start_agent_lifecycle(
        handle="agent-1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="test",
    )
    lifecycle_path = session_dir / "agent_lifecycle.json"
    _inject(lifecycle_path)
    store.finish_agent_lifecycle("completed", final_result="done")
    _assert_future(lifecycle_path)
    store.close()


def test_too_new_refusal_is_visible_in_tui_and_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":999}\n')
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    assert main(["--provider", "codex", "--model", "offline"]) == 1
    assert "newer than writer capability" in capsys.readouterr().err
    assert main(["serve", "--provider", "codex", "--model", "offline"]) == 2
    assert "newer than writer capability" in capsys.readouterr().err


def test_unknown_conversation_variant_fails_closed() -> None:
    with pytest.raises(
        ConversationIntegrityError, match="unsupported conversation entry type"
    ):
        ConversationEntry.from_dict(
            {
                "seq": 1,
                "id": "entry-1",
                "parent_id": None,
                "lane": "main",
                "type": "future_variant",
                "data": {},
            }
        )
