from pathlib import Path

import pytest

from zeta.core.project_context import load_project_context
from zeta.core.slash import SlashStatus, _format_status
from zeta.prompts import load_identity


def test_packaged_identity_loads_outside_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    identity = load_identity()

    assert identity.startswith("You are zeta, a coding agent")
    assert "Honesty:" in identity


def test_project_context_loads_in_fixed_order_and_labels_sources(tmp_path: Path) -> None:
    zeta_home = tmp_path / "zeta-home"
    zeta_home.mkdir()
    (zeta_home / "AGENTS.md").write_text("home rules", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("repo rules", encoding="utf-8")

    context = load_project_context(repo_root=tmp_path, zeta_home=zeta_home)

    assert context.files == (
        (zeta_home / "AGENTS.md").resolve(),
        (tmp_path / "AGENTS.md").resolve(),
    )
    assert context.system_prompt.index("home rules") < context.system_prompt.index(
        "repo rules"
    )
    assert f"Instructions from {(zeta_home / 'AGENTS.md').resolve()}:" in context.system_prompt
    assert f"Instructions from {(tmp_path / 'AGENTS.md').resolve()}:" in context.system_prompt


def test_project_context_uses_repo_claude_only_without_agents(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("claude rules", encoding="utf-8")

    context = load_project_context(repo_root=tmp_path, zeta_home=tmp_path / "home")

    assert context.files == ((tmp_path / "CLAUDE.md").resolve(),)
    assert "claude rules" in context.system_prompt


def test_project_context_skips_missing_files_without_walking(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested rules", encoding="utf-8")

    context = load_project_context(repo_root=tmp_path, zeta_home=tmp_path / "home")

    assert context.files == ()
    assert "nested rules" not in context.system_prompt


def test_project_context_propagates_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("rules", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == path:
            raise PermissionError("unreadable")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(PermissionError, match="unreadable"):
        load_project_context(repo_root=tmp_path, zeta_home=tmp_path / "home")


def test_status_lists_loaded_context_files() -> None:
    output = _format_status(
        SlashStatus(
            session_id="session",
            provider="fake",
            model="offline",
            retained_tail=8,
            tokens_used_this_session=0,
            tokens_in_current_context=None,
            compaction_marker_count=0,
            pending_approvals=(),
            context_files=("/home/henry/.zeta/AGENTS.md",),
        )
    )

    assert "context_files: /home/henry/.zeta/AGENTS.md" in output
