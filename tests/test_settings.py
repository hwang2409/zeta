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
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.notices == ()
    settings = loaded.settings
    assert settings.provider == "claude"
    assert settings.model == "sonnet-4.7"
    assert settings.token_budget == 100000
    # Approval lists come from the global layer only.
    assert settings.approval_allow == ("read",)
    assert settings.approval_deny == ("bash",)
    assert settings.approval_ask == ()


def test_deep_merge_preserves_unrelated_table_keys(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        [keybindings]
        ctrl_p = "previous"
        ctrl_n = "next"
        ctrl_r = "refresh"
        """,
    )
    _write(
        project,
        """
        [keybindings]
        ctrl_p = "back"
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert dict(loaded.settings.keybindings) == {
        "ctrl_p": "back",
        "ctrl_n": "next",
        "ctrl_r": "refresh",
    }


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
    # Omitted --yolo (None) inherits settings.yolo=true.
    inherited = resolve(
        loaded.settings,
        cli_provider="codex",
        cli_model="gpt-5",
        cli_yolo=None,
        cli_token_budget=42,
    )
    assert inherited.provider == "codex"
    assert inherited.model == "gpt-5"
    assert inherited.token_budget == 42
    assert inherited.yolo is True
    # Explicit --no-yolo (False) beats settings.yolo=true.
    overridden = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=False,
        cli_token_budget=None,
    )
    assert overridden.yolo is False
    # Explicit --yolo (True) beats settings.yolo=false too.
    settings_off_loaded = load_settings(home=tmp_path / "empty", project_dir=None)
    forced_on = resolve(
        settings_off_loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=True,
        cli_token_budget=None,
    )
    assert forced_on.yolo is True


def test_resolve_falls_back_to_defaults_when_nothing_configured(tmp_path: Path) -> None:
    loaded = load_settings(home=tmp_path / "home", project_dir=None)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
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
        cli_yolo=None,
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
        cli_yolo=None,
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


def test_hostile_project_cannot_grant_yolo_or_approvals(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        provider = "claude"
        """,
    )
    _write(
        project,
        """
        yolo = true
        model = "sonnet-4.7"

        [approval]
        allow = ["bash", "write"]
        deny = ["read"]
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    # yolo and approval from the project layer are dropped completely.
    assert loaded.settings.yolo is None
    assert loaded.settings.approval_allow == ()
    assert loaded.settings.approval_deny == ()
    assert loaded.settings.approval_ask == ()
    # Safe keys from the project layer still apply.
    assert loaded.settings.model == "sonnet-4.7"
    assert loaded.settings.provider == "claude"
    # A loud warning names the file and the ignored keys.
    assert len(loaded.warnings) == 1
    warning = loaded.warnings[0]
    assert "cannot grant approvals" in warning
    assert "approval" in warning
    assert "yolo" in warning
    assert str(project / SETTINGS_FILENAME) in warning or "~/" in warning
    assert loaded.notices == ()


def test_global_layer_may_still_set_yolo_and_approvals(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        yolo = true

        [approval]
        allow = ["read"]
        deny = ["bash"]
        """,
    )
    _write(
        project,
        """
        model = "sonnet-4.7"
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.warnings == ()
    assert loaded.settings.yolo is True
    assert loaded.settings.approval_allow == ("read",)
    assert loaded.settings.approval_deny == ("bash",)
    assert loaded.settings.model == "sonnet-4.7"


def test_keybindings_reject_non_string_values(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        [keybindings]
        ctrl_p = "previous"
        ctrl_n = 42
        ctrl_r = true
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert dict(loaded.settings.keybindings) == {"ctrl_p": "previous"}
    joined = " ".join(loaded.notices)
    assert "keybindings.ctrl_n" in joined
    assert "keybindings.ctrl_r" in joined


def test_notices_collapse_home_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    home_dir = fake_home / ".zeta"
    (home_dir).mkdir()
    (home_dir / SETTINGS_FILENAME).write_text("not = valid = toml", encoding="utf-8")
    loaded = load_settings(home=home_dir, project_dir=None)
    assert loaded.notices
    combined = " ".join(loaded.notices)
    assert str(fake_home) not in combined
    assert "~/" in combined


def test_project_settings_read_from_dot_zeta(tmp_path: Path) -> None:
    """End-to-end: <project>/.zeta/settings.toml is what the project layer reads."""

    home = tmp_path / "home"
    project = tmp_path / "project"
    dot_zeta = project / ".zeta"
    _write(dot_zeta, 'model = "sonnet-4.7"\n')
    loaded = load_settings(home=home, project_dir=dot_zeta)
    assert loaded.settings.model == "sonnet-4.7"
    # A non-dot-zeta project directory sees no settings.
    plain = load_settings(home=home, project_dir=project)
    assert plain.settings.model is None


# --- ZETA-86: argument-scoped rules ---------------------------------------


def test_scoped_approval_rules_round_trip_into_policy(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        [approval]
        allow = ["bash(git status*)", "read"]
        deny = ["bash(rm *)"]
        ask = ["write(/etc/*)"]
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.warnings == ()
    assert loaded.notices == ()
    assert loaded.settings.approval_allow == ("bash(git status*)", "read")
    policy = ApprovalPolicy(
        store=ConversationStore(tmp_path / "sessions"),
        always_allow=loaded.settings.approval_allow,
        always_deny=loaded.settings.approval_deny,
        always_ask=loaded.settings.approval_ask,
    )
    policy.declare_subjects({"bash": "command", "read": "path", "write": "path"})
    assert policy.decide("bash", {"command": "git status -s"}) is ApprovalDecision.ALLOW
    assert policy.decide("bash", {"command": "rm -rf /"}) is ApprovalDecision.DENY
    assert policy.decide("bash", {"command": "ls"}) is ApprovalDecision.ASK
    assert policy.decide("write", {"path": "/etc/hosts"}) is ApprovalDecision.ASK
    assert policy.decide("read", {"path": "/etc/hosts"}) is ApprovalDecision.ALLOW


def test_malformed_approval_rule_is_dropped_with_loud_warning(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(
        home,
        """
        [approval]
        allow = ["bash(git status", "read"]
        deny = ["bash()"]
        """,
    )
    loaded = load_settings(home=home, project_dir=None)
    # Only the malformed entry is dropped; the rest of the list survives.
    assert loaded.settings.approval_allow == ("read",)
    assert loaded.settings.approval_deny == ()
    assert loaded.notices == ()
    assert len(loaded.warnings) == 2
    assert "approval.allow" in loaded.warnings[0]
    assert "'bash(git status'" in loaded.warnings[0]
    assert "approval.deny" in loaded.warnings[1]
    assert "'bash()'" in loaded.warnings[1]


def test_hostile_project_cannot_grant_scoped_approvals(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write(
        home,
        """
        [approval]
        deny = ["bash(rm *)"]
        """,
    )
    _write(
        project,
        """
        [approval]
        allow = ["bash(*)", "write(*)"]
        deny = []
        """,
    )
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings.approval_allow == ()
    assert loaded.settings.approval_deny == ("bash(rm *)",)
    assert len(loaded.warnings) == 1
    assert "cannot grant approvals" in loaded.warnings[0]
    policy = ApprovalPolicy(
        always_allow=loaded.settings.approval_allow,
        always_deny=loaded.settings.approval_deny,
        always_ask=loaded.settings.approval_ask,
    )
    policy.declare_subjects({"bash": "command", "write": "path"})
    assert policy.decide("bash", {"command": "git status"}) is ApprovalDecision.ASK
    assert policy.decide("write", {"path": "x"}) is ApprovalDecision.ASK
    assert policy.decide("bash", {"command": "rm -rf /"}) is ApprovalDecision.DENY


def test_scoped_rules_reach_the_live_policy_through_create_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: settings -> policy -> registry subject declaration."""

    from zeta.cli import build_parser
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    _write(
        home,
        """
        [approval]
        allow = ["bash(git status*)", "todo(*)"]
        """,
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    app = create_app(build_parser().parse_args(["--provider", "fake"]))

    policy = app.approval_policy
    assert policy is not None
    assert policy.decide("bash", {"command": "git status --short"}) is ApprovalDecision.ALLOW
    assert policy.decide("bash", {"command": "git push"}) is ApprovalDecision.ASK
    assert policy.decide("bash", {"cmd": "git status"}) is ApprovalDecision.ASK
    # todo declares no subject, so the scoped rule is dropped and reported.
    assert policy.decide("todo", {}) is ApprovalDecision.ASK
    assert len(policy.notices) == 1
    assert "todo(*)" in policy.notices[0]
