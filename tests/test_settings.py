"""Tests for the layered settings module (ZETA-73)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.settings import (
    SETTINGS_FILENAME,
    LoadedSettings,
    ResolvedConfig,
    Settings,
    load_settings,
    resolve,
)


def _write(base: Path, body: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / SETTINGS_FILENAME
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    return path


def test_missing_files_are_silent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings == Settings()
    assert loaded.notices == ()


def test_global_and_project_merge_with_project_wins(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        provider = "claude"
        model = "opus-4.7"
        token_budget = 100000

        [approval]
        allow = ["read"]
        deny = ["bash"]
        """,
    )
    _write(
        project,
        """
        model = "sonnet-4.7"

        [approval]
        allow = ["read", "write"]
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.notices == ()
    settings = loaded.settings
    assert settings.provider == "claude"
    assert settings.model == "sonnet-4.7"
    assert settings.token_budget == 100000
    assert settings.approval_allow == ("read", "write")
    assert settings.approval_deny == ("bash",)
    assert settings.approval_ask == ()


def test_deep_merge_preserves_unrelated_table_keys(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        [approval]
        allow = ["read"]
        deny = ["bash"]
        ask = ["edit"]
        """,
    )
    _write(
        project,
        """
        [approval]
        allow = ["fetch"]
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings.approval_allow == ("fetch",)
    assert loaded.settings.approval_deny == ("bash",)
    assert loaded.settings.approval_ask == ("edit",)


def test_cli_flags_override_settings(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        provider = "claude"
        model = "opus-4.7"
        yolo = true
        token_budget = 100000
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    config = resolve(
        loaded.settings,
        cli_provider="codex",
        cli_model="gpt-5",
        cli_yolo=False,
        cli_token_budget=42,
    )
    assert config.provider == "codex"
    assert config.model == "gpt-5"
    assert config.token_budget == 42
    assert config.yolo is True  # settings.yolo still wins because CLI has no --no-yolo


def test_resolve_falls_back_to_defaults_when_nothing_configured(tmp_path: Path) -> None:
    loaded = load_settings(home=tmp_path / "home", project_dir=None)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=False,
        cli_token_budget=None,
    )
    assert config.provider == "fake"
    assert config.model is None
    assert config.yolo is False
    assert config.token_budget is None
    assert config.approval_allow == ()


def test_malformed_toml_fails_open_with_notice(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (home / SETTINGS_FILENAME).write_text("not = valid = toml", encoding="utf-8")
    _write(
        project,
        """
        provider = "codex"
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings.provider == "codex"
    assert len(loaded.notices) == 1
    assert "ignored" in loaded.notices[0]
    assert str(home / SETTINGS_FILENAME) in loaded.notices[0]


def test_malformed_project_does_not_erase_global(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        provider = "claude"
        model = "opus-4.7"
        """,
    )
    project.mkdir()
    (project / SETTINGS_FILENAME).write_text("garbage[", encoding="utf-8")
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings.provider == "claude"
    assert loaded.settings.model == "opus-4.7"
    assert any("ignored" in notice for notice in loaded.notices)


def test_invalid_types_are_dropped_with_notices(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        provider = 5
        model = "opus-4.7"
        yolo = "on"
        token_budget = -1

        [approval]
        allow = "read"
        deny = ["ok", 5]
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings.provider is None
    assert loaded.settings.model == "opus-4.7"
    assert loaded.settings.yolo is None
    assert loaded.settings.token_budget is None
    assert loaded.settings.approval_allow == ()
    assert loaded.settings.approval_deny == ()
    notice_targets = " ".join(loaded.notices)
    assert "provider" in notice_targets
    assert "yolo" in notice_targets
    assert "token_budget" in notice_targets
    assert "approval.allow" in notice_targets
    assert "approval.deny" in notice_targets


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        provider = "gpt"
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings.provider is None
    assert any("provider" in notice for notice in loaded.notices)


def test_unknown_top_level_key_is_dropped_with_notice(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        unknown = "value"
        provider = "claude"
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings.provider == "claude"
    assert any("unknown" in notice for notice in loaded.notices)


def test_keybindings_table_is_loaded_but_reserved(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        [keybindings]
        ctrl_p = "previous"
        ctrl_n = "next"
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.notices == ()
    assert dict(loaded.settings.keybindings) == {
        "ctrl_p": "previous",
        "ctrl_n": "next",
    }


def test_keybindings_must_be_a_table(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        keybindings = "invalid"
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert dict(loaded.settings.keybindings) == {}
    assert any("keybindings" in notice for notice in loaded.notices)


def test_approval_lists_round_trip_into_policy(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        [approval]
        allow = ["read", "write"]
        deny = ["bash"]
        ask = ["edit"]
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(
        store=store,
        always_allow=loaded.settings.approval_allow,
        always_deny=loaded.settings.approval_deny,
        always_ask=loaded.settings.approval_ask,
        default=ApprovalDecision.ASK,
    )
    assert policy.decide("read", {}) is ApprovalDecision.ALLOW
    assert policy.decide("write", {}) is ApprovalDecision.ALLOW
    assert policy.decide("bash", {}) is ApprovalDecision.DENY
    assert policy.decide("edit", {}) is ApprovalDecision.ASK
    assert policy.decide("other", {}) is ApprovalDecision.ASK


def test_yolo_from_settings_flows_into_approval_default(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        yolo = true
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=False,
        cli_token_budget=None,
    )
    assert config.yolo is True
    default = ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK
    assert default is ApprovalDecision.ALLOW


def test_load_returns_loaded_settings_dataclass(tmp_path: Path) -> None:
    loaded = load_settings(home=tmp_path, project_dir=None)
    assert isinstance(loaded, LoadedSettings)
    assert isinstance(loaded.settings, Settings)


def test_resolve_returns_resolved_config(tmp_path: Path) -> None:
    loaded = load_settings(home=tmp_path, project_dir=None)
    config = resolve(
        loaded.settings,
        cli_provider="fake",
        cli_model=None,
        cli_yolo=False,
        cli_token_budget=None,
    )
    assert isinstance(config, ResolvedConfig)
    assert config.provider == "fake"


def test_project_and_global_pointing_to_same_dir_is_read_once(tmp_path: Path) -> None:
    _write(
        tmp_path,
        """
        provider = "claude"
        """,
    )
    loaded = load_settings(home=tmp_path, project_dir=tmp_path)
    assert loaded.settings.provider == "claude"
    assert loaded.notices == ()


@pytest.mark.parametrize("value", ["", "\n", "   \n"])
def test_empty_settings_file_is_silent(tmp_path: Path, value: str) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / SETTINGS_FILENAME).write_text(value, encoding="utf-8")
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings == Settings()
    assert loaded.notices == ()
