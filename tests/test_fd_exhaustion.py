"""Open-file exhaustion: a raised limit, fewer pinned directories, honest errors."""

from __future__ import annotations

import errno
import os
import resource
from pathlib import Path

import pytest

import zeta.accounts.files as account_files
import zeta.cli.main as cli_main
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountError, AccountStore
from zeta.codex import CodexCredentialDriver
from zeta.core import session_files
from zeta.core.store import ConversationStore
from zeta.oauth import OAuthTokens
from zeta.providers.anthropic import AnthropicCredentialDriver
from zeta.runtime.loop._completion import _error_info
from zeta.tools._shared.process import background_tasks_for_session

_INF = resource.RLIM_INFINITY


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ((256, _INF), (10240, _INF)),
        ((256, 4096), (4096, 4096)),
        ((20000, _INF), None),
        ((_INF, _INF), None),
    ],
)
def test_cli_start_raises_soft_open_file_limit_without_lowering(
    monkeypatch: pytest.MonkeyPatch,
    current: tuple[int, int],
    expected: tuple[int, int] | None,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _which: current)
    monkeypatch.setattr(
        resource, "setrlimit", lambda _which, limits: calls.append(limits)
    )

    with pytest.raises(SystemExit):
        cli_main.main(["--help"])

    assert calls == ([] if expected is None else [expected])


def test_cli_start_ignores_a_refused_open_file_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_which: int, _limits: tuple[int, int]) -> None:
        raise ValueError("not allowed to raise maximum limit")

    monkeypatch.setattr(resource, "getrlimit", lambda _which: (256, _INF))
    monkeypatch.setattr(resource, "setrlimit", refuse)

    with pytest.raises(SystemExit) as exited:
        cli_main.main(["--help"])

    assert exited.value.code == 0


def _descriptors_on(path: Path) -> int:
    target = os.stat(path)
    count = 0
    for name in os.listdir("/dev/fd"):
        try:
            info = os.fstat(int(name))
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino):
            count += 1
    return count


@pytest.mark.asyncio
async def test_live_child_pins_only_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    sessions = tmp_path / "sessions"
    parent = ConversationStore(sessions, session_id="parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    tasks = background_tasks_for_session(child, command_niceness=10)
    try:
        held = {
            path.relative_to(tmp_path).as_posix(): _descriptors_on(path)
            for path in (
                sessions,
                parent.session_dir,
                child.session_dir.parent,
                child.session_dir,
            )
        }
        # One pin per open store plus the background registry's lease; no parent pins.
        assert held == {
            "sessions": 0,
            "sessions/parent": 1,
            "sessions/parent/agents": 0,
            "sessions/parent/agents/1": 2,
        }
    finally:
        await tasks.close()
        child.close()
        parent.close()


def _exhausted(*_args: object, **_kwargs: object) -> int:
    raise OSError(errno.EMFILE, os.strerror(errno.EMFILE))


def _assert_descriptor_error(error: BaseException) -> None:
    assert not isinstance(error, AccountError), str(error)
    assert "too many open files (soft limit" in str(error)
    assert "login" not in str(error)
    assert _error_info(error, provider_error=True).provider_auth_failure is False


@pytest.mark.parametrize(
    ("module", "attribute", "name"),
    [
        # opening accounts/codex/default (_read_account_tokens)
        (session_files, "_open_child_directory", "default"),
        # opening oauth.json inside it (_read_tokens)
        (account_files, "open_session_file", "oauth.json"),
    ],
)
def test_exhausted_account_token_read_is_not_a_login_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    attribute: str,
    name: str,
) -> None:
    credentials = named_account_credentials(tmp_path, "codex", "default")
    credentials.save(OAuthTokens("access", "refresh", 9_999_999_999))
    real_open = getattr(module, attribute)

    def open_file(
        directory_fd: int, opened: str, *args: object, **kwargs: object
    ) -> int:
        if opened == name:
            return _exhausted()
        return real_open(directory_fd, opened, *args, **kwargs)

    monkeypatch.setattr(module, attribute, open_file)

    with pytest.raises(Exception) as caught:
        credentials.read()

    _assert_descriptor_error(caught.value)


def test_exhausted_registry_read_is_not_a_broken_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named_account_credentials(tmp_path, "codex", "default").save(
        OAuthTokens("access", "refresh", 9_999_999_999)
    )
    monkeypatch.setattr(account_files, "open_session_file", _exhausted)

    with pytest.raises(Exception) as caught:
        AccountStore(tmp_path).list_display()

    _assert_descriptor_error(caught.value)
    assert "unreadable" not in str(caught.value)


@pytest.mark.parametrize(
    ("driver", "variable", "name"),
    [
        (CodexCredentialDriver, "CODEX_HOME", "auth.json"),
        (AnthropicCredentialDriver, "CLAUDE_CONFIG_DIR", ".credentials.json"),
    ],
)
def test_exhausted_provider_bootstrap_is_not_a_login_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: type[CodexCredentialDriver | AnthropicCredentialDriver],
    variable: str,
    name: str,
) -> None:
    (tmp_path / name).write_text("{}")
    monkeypatch.setenv(variable, str(tmp_path))
    monkeypatch.setattr(Path, "open", _exhausted)

    with pytest.raises(Exception) as caught:
        driver().bootstrap()

    _assert_descriptor_error(caught.value)


def test_exhausted_accounts_status_reports_unknown_not_invalid_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    named_account_credentials(tmp_path, "codex", "default").save(
        OAuthTokens("access", "refresh", 9_999_999_999)
    )
    real_open = account_files.open_session_file

    def open_file(directory_fd: int, name: str, flags: int) -> int:
        if name == "oauth.json":
            return _exhausted()
        return real_open(directory_fd, name, flags)

    monkeypatch.setattr(account_files, "open_session_file", open_file)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert cli_main.main(["accounts", "status"]) == 1

    assert capsys.readouterr().out.strip() == (
        "codex:default  login unknown (too_many_open_files)  expires unknown  default"
    )
