from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import zeta.accounts.store as account_store_module
import zeta.cli.main as cli_main
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, AccountStore
from zeta.core.session_files import SessionFileReplaced
from zeta.oauth import OAuthTokens

_ACCESS_SECRET = "access-secret-marker"
_REFRESH_SECRET = "refresh-secret-marker"


def _save_account(
    home: Path,
    provider: str = "codex",
    name: str = "default",
    *,
    expires_at: float = 9_999_999_999,
) -> Path:
    credential = named_account_credentials(home, provider, name)
    credential.save(OAuthTokens(_ACCESS_SECRET, _REFRESH_SECRET, expires_at))
    return credential.path


def _use_token_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    response: httpx.Response,
) -> None:
    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: response)

    def credentials(provider: str, *, home: Path, account: AccountId):
        assert provider == account.provider == "codex"
        return named_account_credentials(
            home, account.provider, account.name, token_url="https://refresh.test/token"
        )

    monkeypatch.setattr(cli_main, "credential_store", credentials)
    monkeypatch.setattr(
        cli_main.httpx,
        "AsyncClient",
        lambda: original_async_client(transport=transport),
    )


def _run_json(home: Path, monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setenv("ZETA_HOME", str(home))
    exit_code = cli_main.main(["accounts", "status", "--json", *args])
    return exit_code


def test_accounts_status_reports_login_expiry_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path)
    backend_calls = 0

    def fail_backend(*_args: object, **_kwargs: object) -> None:
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("status must not create a provider backend")

    monkeypatch.setattr("zeta.providers.factory.build_backend", fail_backend)

    assert _run_json(tmp_path, monkeypatch) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == [
        {
            "provider": "codex",
            "account": "default",
            "is_default": True,
            "logged_in": True,
            "access_token_expires_at": "2286-11-20T17:46:39Z",
        }
    ]
    assert _ACCESS_SECRET not in output
    assert _REFRESH_SECRET not in output
    assert backend_calls == 0
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize(
    ("response", "expected_code", "expected_retryable"),
    [
        (httpx.Response(400, json={"error": "invalid_grant"}), "auth_error", None),
        (httpx.Response(503, json={"error": "unavailable"}), "http_error", True),
    ],
)
def test_accounts_status_refresh_classifies_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    response: httpx.Response,
    expected_code: str,
    expected_retryable: bool | None,
) -> None:
    _save_account(tmp_path, expires_at=1)
    _use_token_endpoint(monkeypatch, tmp_path, response)

    assert _run_json(tmp_path, monkeypatch, "--refresh") != 0
    [status] = json.loads(capsys.readouterr().out)
    assert status["logged_in"] is True
    assert status["refresh"] == {
        "ok": False,
        "code": expected_code,
        "retryable": expected_retryable,
    }


def test_accounts_status_refresh_rotates_persisted_token_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_path = _save_account(tmp_path, expires_at=1)
    before = credential_path.read_bytes()
    _use_token_endpoint(
        monkeypatch,
        tmp_path,
        httpx.Response(
            200,
            json={
                "access_token": "rotated-access-secret",
                "refresh_token": "rotated-refresh-secret",
                "expires_in": 3600,
            },
        ),
    )

    assert _run_json(tmp_path, monkeypatch, "--refresh") == 0
    output = capsys.readouterr().out
    [status] = json.loads(output)
    assert status["refresh"] == {"ok": True, "code": None, "retryable": None}
    assert credential_path.read_bytes() != before
    persisted = named_account_credentials(tmp_path, "codex").read()
    assert persisted is not None
    assert persisted.access_token == "rotated-access-secret"
    assert "rotated-access-secret" not in output
    assert "rotated-refresh-secret" not in output
    assert _ACCESS_SECRET not in output
    assert _REFRESH_SECRET not in output


def test_accounts_status_missing_login_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path)
    AccountStore(tmp_path).credential_path(AccountId("codex", "default")).unlink()

    assert _run_json(tmp_path, monkeypatch) != 0
    [status] = json.loads(capsys.readouterr().out)
    assert status["logged_in"] is False
    assert status["access_token_expires_at"] is None


def test_accounts_status_retries_registry_replacement_and_keeps_json_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path)
    store = AccountStore(tmp_path)
    real_open = account_store_module.open_session_file
    replace_next_read = True

    def replacing_open(directory_fd: int, name: str, flags: int) -> int:
        nonlocal replace_next_read
        if name == "accounts.json" and replace_next_read:
            replace_next_read = False
            raise SessionFileReplaced("registry replaced during validation")
        replace_next_read = True
        return real_open(directory_fd, name, flags)

    monkeypatch.setattr(account_store_module, "open_session_file", replacing_open)

    for _ in range(20):
        assert [item.account for item in store.list_catalog()] == [
            AccountId("codex", "default")
        ]
        assert _run_json(tmp_path, monkeypatch) == 0
        assert json.loads(capsys.readouterr().out)[0]["logged_in"] is True


def test_accounts_status_keeps_migrated_account_when_legacy_file_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path)
    (tmp_path / "codex-oauth.json").write_text(
        json.dumps(OAuthTokens("legacy-access", "legacy-refresh", 1).to_dict())
    )

    assert _run_json(tmp_path, monkeypatch) == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "provider": "codex",
            "account": "default",
            "is_default": True,
            "logged_in": True,
            "access_token_expires_at": "2286-11-20T17:46:39Z",
        }
    ]


def test_accounts_status_reports_pending_legacy_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "codex-oauth.json").write_text(
        json.dumps(OAuthTokens(_ACCESS_SECRET, _REFRESH_SECRET, 1).to_dict())
    )

    assert _run_json(tmp_path, monkeypatch, "--refresh") == 1
    output = capsys.readouterr().out
    assert json.loads(output) == [
        {
            "provider": "codex",
            "account": "default",
            "is_default": True,
            "logged_in": None,
            "access_token_expires_at": None,
            "error": "migration_pending",
            "retryable": False,
            "refresh": {
                "ok": False,
                "code": "migration_pending",
                "retryable": False,
            },
        }
    ]
    assert _ACCESS_SECRET not in output
    assert _REFRESH_SECRET not in output


def test_accounts_status_is_bounded_while_every_account_lock_is_held(
    tmp_path: Path,
) -> None:
    accounts = [AccountId(provider, "default") for provider in ("anthropic", "codex")]
    for account in accounts:
        _save_account(tmp_path, account.provider)
    store = AccountStore(tmp_path)
    store.migrate()
    for provider in ("anthropic", "codex"):
        with store.provider_login_lock(provider):
            pass
    for account in accounts:
        with store.usage_guard(account):
            pass

    lock_paths = sorted(tmp_path.rglob("*.lock"))
    assert {path.relative_to(tmp_path).as_posix() for path in lock_paths} == {
        ".accounts-migration.lock",
        ".accounts.lock",
        ".anthropic-oauth.json.lock",
        ".codex-oauth.json.lock",
        "account-locks/anthropic/default/lifecycle.lock",
        "account-locks/anthropic/default/usage.lock",
        "account-locks/anthropic/login.lock",
        "account-locks/codex/default/lifecycle.lock",
        "account-locks/codex/default/usage.lock",
        "account-locks/codex/login.lock",
    }

    ready = tmp_path / "all-locks-ready"
    release_lifecycle = tmp_path / "release-lifecycle"
    lifecycle_released = tmp_path / "lifecycle-released"
    reacquire_lifecycle = tmp_path / "reacquire-lifecycle"
    lifecycle_reacquired = tmp_path / "lifecycle-reacquired"
    done = tmp_path / "all-locks-done"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys,time; from pathlib import Path; "
                "ready,release,released,reacquire,reacquired,done="
                "map(Path,sys.argv[1:7]); paths=list(map(Path,sys.argv[7:])); "
                "handles=[path.open('a+') for path in paths]; "
                "lifecycle=[handle for handle in handles "
                "if handle.name.endswith('/lifecycle.lock')]; "
                "[fcntl.flock(handle,fcntl.LOCK_EX) for handle in handles]; "
                "ready.touch(); "
                "\nwhile not release.exists(): time.sleep(0.01)\n"
                "[fcntl.flock(handle,fcntl.LOCK_UN) for handle in lifecycle]; "
                "released.touch(); "
                "\nwhile not reacquire.exists(): time.sleep(0.01)\n"
                "[fcntl.flock(handle,fcntl.LOCK_EX) for handle in lifecycle]; "
                "reacquired.touch(); "
                "\nwhile not done.exists(): time.sleep(0.01)\n"
            ),
            str(ready),
            str(release_lifecycle),
            str(lifecycle_released),
            str(reacquire_lifecycle),
            str(lifecycle_reacquired),
            str(done),
            *(str(path) for path in lock_paths),
        ]
    )

    def wait_for(path: Path) -> None:
        deadline = time.monotonic() + 10
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists()

    command = (
        "import zeta.accounts.store as store, zeta.cli.main as cli; "
        "store.OAUTH_REFRESH_TIMEOUT_SECONDS=cli.OAUTH_REFRESH_TIMEOUT_SECONDS=0.05; "
        "raise SystemExit(cli.main(['accounts','status','--json']))"
    )
    env = {**os.environ, "ZETA_HOME": str(tmp_path)}
    try:
        wait_for(ready)
        plain = subprocess.run(
            [sys.executable, "-c", command],
            env=env,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        assert plain.returncode == 1
        assert {row["error"] for row in json.loads(plain.stdout)} == {"lock_timeout"}

        release_lifecycle.touch()
        wait_for(lifecycle_released)
        refresh_command = (
            "import asyncio; from pathlib import Path; "
            "import zeta.accounts.store as store, zeta.cli.main as cli; "
            "store.OAUTH_REFRESH_TIMEOUT_SECONDS=cli.OAUTH_REFRESH_TIMEOUT_SECONDS=0.05; "
            "real=cli._read_safe_account_status; first=True; "
            "\nasync def read(credentials):\n"
            " global first\n result=await real(credentials)\n"
            " if first:\n  first=False\n  Path(sys.argv[1]).touch()\n"
            "  while not Path(sys.argv[2]).exists(): await asyncio.sleep(0.01)\n"
            " return result\n"
            "\nimport sys; cli._read_safe_account_status=read; "
            "raise SystemExit(cli.main(['accounts','status','--refresh','--json']))"
        )
        refreshed = subprocess.run(
            [
                sys.executable,
                "-c",
                refresh_command,
                str(reacquire_lifecycle),
                str(lifecycle_reacquired),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert refreshed.returncode == 1
        refresh_rows = json.loads(refreshed.stdout)
        assert {row["refresh"]["code"] for row in refresh_rows} == {"lock_timeout"}
    finally:
        done.touch()
        holder.wait(timeout=10)
    assert holder.returncode == 0


def test_accounts_status_lock_timeout_keeps_json_valid_and_reports_other_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = AccountId("codex", "locked")
    _save_account(tmp_path, name=locked.name)
    _save_account(tmp_path, name="healthy")
    ready = tmp_path / "lock-ready"
    release = tmp_path / "lock-release"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; from pathlib import Path; "
                "from zeta.accounts import AccountId,AccountStore; "
                "home,ready,release=map(Path,sys.argv[1:]); "
                "account=AccountId('codex','locked'); "
                "guard=AccountStore(home).lifecycle_guard(account); "
                "guard.__enter__(); ready.touch(); "
                "\nwhile not release.exists(): time.sleep(0.01)\n"
                "guard.__exit__(None,None,None)"
            ),
            str(tmp_path),
            str(ready),
            str(release),
        ]
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    monkeypatch.setattr(
        cli_main, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.1, raising=False
    )
    fallback_release = threading.Timer(1, release.touch)
    fallback_release.start()

    started = time.monotonic()
    try:
        assert _run_json(tmp_path, monkeypatch) == 1
    finally:
        release.touch()
        fallback_release.cancel()
        fallback_release.join(timeout=2)
        process.wait(timeout=10)
    assert process.returncode == 0
    assert time.monotonic() - started < 0.5
    assert json.loads(capsys.readouterr().out) == [
        {
            "provider": "codex",
            "account": "healthy",
            "is_default": False,
            "logged_in": True,
            "access_token_expires_at": "2286-11-20T17:46:39Z",
        },
        {
            "provider": "codex",
            "account": "locked",
            "is_default": True,
            "logged_in": None,
            "access_token_expires_at": None,
            "error": "lock_timeout",
            "retryable": True,
        },
    ]


def test_accounts_status_refresh_lock_timeout_after_initial_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path, expires_at=1)
    _use_token_endpoint(
        monkeypatch,
        tmp_path,
        httpx.Response(
            200,
            json={
                "access_token": "rotated-access-secret",
                "refresh_token": "rotated-refresh-secret",
                "expires_in": 3600,
            },
        ),
    )
    ready = tmp_path / "refresh-race-lock-ready"
    release = tmp_path / "refresh-race-lock-release"
    process: subprocess.Popen[bytes] | None = None
    real_read = cli_main._read_safe_account_status

    async def read_then_lock(credentials):
        nonlocal process
        result = await real_read(credentials)
        if process is None:
            process = await asyncio.to_thread(
                subprocess.Popen,
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; from pathlib import Path; "
                        "from zeta.accounts import AccountId,AccountStore; "
                        "home,ready,release=map(Path,sys.argv[1:]); "
                        "guard=AccountStore(home).lifecycle_guard("
                        "AccountId('codex','default')); "
                        "guard.__enter__(); ready.touch(); "
                        "\nwhile not release.exists(): time.sleep(0.01)\n"
                        "guard.__exit__(None,None,None)"
                    ),
                    str(tmp_path),
                    str(ready),
                    str(release),
                ]
            )
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert ready.exists()
        return result

    monkeypatch.setattr(cli_main, "_read_safe_account_status", read_then_lock)
    monkeypatch.setattr(
        account_store_module, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.01
    )

    try:
        assert _run_json(tmp_path, monkeypatch, "--refresh") == 1
    finally:
        release.touch()
        assert process is not None
        process.wait(timeout=10)

    assert process.returncode == 0
    [status] = json.loads(capsys.readouterr().out)
    assert status["refresh"] == {
        "ok": False,
        "code": "lock_timeout",
        "retryable": True,
    }


def test_accounts_status_refresh_lock_timeout_has_refresh_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path, name="locked")
    ready = tmp_path / "refresh-lock-ready"
    release = tmp_path / "refresh-lock-release"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; from pathlib import Path; "
                "from zeta.accounts import AccountId,AccountStore; "
                "home,ready,release=map(Path,sys.argv[1:]); "
                "guard=AccountStore(home).lifecycle_guard(AccountId('codex','locked')); "
                "guard.__enter__(); ready.touch(); "
                "\nwhile not release.exists(): time.sleep(0.01)\n"
                "guard.__exit__(None,None,None)"
            ),
            str(tmp_path),
            str(ready),
            str(release),
        ]
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    monkeypatch.setattr(
        cli_main, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.01, raising=False
    )

    try:
        assert _run_json(tmp_path, monkeypatch, "--refresh") == 1
    finally:
        release.touch()
        process.wait(timeout=10)

    assert process.returncode == 0
    [status] = json.loads(capsys.readouterr().out)
    assert status["refresh"] == {
        "ok": False,
        "code": "lock_timeout",
        "retryable": True,
    }


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), 1e300])
def test_accounts_status_non_renderable_expiry_is_invalid_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expires_at: float,
) -> None:
    credential_path = _save_account(tmp_path)
    value = json.loads(credential_path.read_text())
    value["expires_at"] = expires_at
    credential_path.write_text(json.dumps(value))

    assert _run_json(tmp_path, monkeypatch) == 1
    assert json.loads(capsys.readouterr().out) == [
        {
            "provider": "codex",
            "account": "default",
            "is_default": True,
            "logged_in": False,
            "access_token_expires_at": None,
            "error": "invalid_credentials",
        }
    ]


def test_accounts_status_corrupt_login_keeps_json_valid_and_reports_other_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corrupt_path = _save_account(tmp_path, name="corrupt")
    _save_account(tmp_path, name="healthy")
    corrupt_path.write_text("{broken secret-file-content")

    assert _run_json(tmp_path, monkeypatch) == 1
    captured = capsys.readouterr()
    statuses = json.loads(captured.out)
    assert statuses == [
        {
            "provider": "codex",
            "account": "corrupt",
            "is_default": True,
            "logged_in": False,
            "access_token_expires_at": None,
            "error": "invalid_credentials",
        },
        {
            "provider": "codex",
            "account": "healthy",
            "is_default": False,
            "logged_in": True,
            "access_token_expires_at": "2286-11-20T17:46:39Z",
        },
    ]
    assert "secret-file-content" not in captured.out + captured.err


def test_accounts_status_corrupt_login_human_output_has_one_line_per_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corrupt_path = _save_account(tmp_path, name="corrupt")
    _save_account(tmp_path, name="healthy")
    corrupt_path.write_text("{broken secret-file-content")
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert cli_main.main(["accounts", "status"]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "codex:corrupt  login invalid (invalid_credentials)  expires unknown  default",
        "codex:healthy  logged in  expires 2286-11-20T17:46:39Z",
    ]
    assert "Traceback" not in captured.out + captured.err
    assert "secret-file-content" not in captured.out + captured.err


def test_accounts_status_provider_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path, "anthropic", "work")
    _save_account(tmp_path, "codex", "personal")
    AccountStore(tmp_path).rename(AccountId("anthropic", "work"), "team")

    assert _run_json(tmp_path, monkeypatch, "anthropic") == 0

    statuses = json.loads(capsys.readouterr().out)
    assert [(item["provider"], item["account"]) for item in statuses] == [
        ("anthropic", "team")
    ]

    assert _run_json(tmp_path, monkeypatch, "codex") == 0
    assert [item["provider"] for item in json.loads(capsys.readouterr().out)] == [
        "codex"
    ]

    empty_home = tmp_path / "empty"
    assert _run_json(empty_home, monkeypatch, "codex") == 1
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize("refresh", [False, True])
def test_accounts_status_never_blocks_on_any_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh: bool,
) -> None:
    # The invariant, checked at the one seam every zeta lock goes through:
    # nothing the status command reaches may take a lock without LOCK_NB,
    # so a lock added later anywhere is caught here too.
    import fcntl

    _save_account(tmp_path, expires_at=1)
    _use_token_endpoint(
        monkeypatch,
        tmp_path,
        httpx.Response(
            200,
            json={
                "access_token": "rotated-access-secret",
                "refresh_token": "rotated-refresh-secret",
                "expires_in": 3600,
            },
        ),
    )
    real_flock = fcntl.flock
    blocking: list[int] = []

    def nonblocking_only(fd: int, operation: int) -> None:
        if operation & (fcntl.LOCK_EX | fcntl.LOCK_SH) and not operation & fcntl.LOCK_NB:
            blocking.append(operation)
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", nonblocking_only)

    args = ("--refresh",) if refresh else ()
    _run_json(tmp_path, monkeypatch, *args)
    [row] = json.loads(capsys.readouterr().out)
    if refresh:
        assert row["refresh"] == {"ok": True, "code": None, "retryable": None}
    assert blocking == []
