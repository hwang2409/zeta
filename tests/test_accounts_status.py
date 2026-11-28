from __future__ import annotations

import asyncio
import errno
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

import zeta.accounts.store as account_store_module
import zeta.cli.main as cli_main
import zeta.server.accounts as server_accounts
from tests.support.credentials import named_account_credentials
from tests.support.storage_homes import initialize_storage_home
from zeta.accounts import AccountError, AccountId, AccountStore
from zeta.accounts.session import SessionAccountContext
from zeta.accounts.usage import AccountUsage
from zeta.core.session import SessionManager
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


def test_accounts_status_reports_recovery_for_malformed_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialize_storage_home(tmp_path)
    registry = tmp_path / "accounts.json"
    original = b"not json\n"
    registry.write_bytes(original)

    assert _run_json(tmp_path, monkeypatch) == 1
    assert capsys.readouterr().err == (
        f"accounts.json is unreadable: {registry}; kept it and every account directory; "
        "move it aside to a backup, then run login again or restore it from a backup, "
        "then restart zeta\n"
    )
    assert registry.read_bytes() == original


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
    initialize_storage_home(tmp_path)
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


@pytest.mark.parametrize("refresh", [False, True])
def test_accounts_status_is_bounded_while_every_account_lock_is_held(
    tmp_path: Path,
    refresh: bool,
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
        ".storage-format.lock",
        ".anthropic-oauth.json.lock",
        ".codex-oauth.json.lock",
        "account-locks/anthropic/default/lifecycle.lock",
        "account-locks/anthropic/default/usage.lock",
        "account-locks/anthropic/login.lock",
        "account-locks/codex/default/lifecycle.lock",
        "account-locks/codex/default/usage.lock",
        "account-locks/codex/login.lock",
    }
    lock_paths = [path for path in lock_paths if path.name != ".storage-format.lock"]

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys; from pathlib import Path; "
                "paths=list(map(Path,sys.argv[1:])); "
                "handles=[path.open('a+') for path in paths]; "
                "lifecycle=[handle for handle in handles "
                "if handle.name.endswith('/lifecycle.lock')]; "
                "[fcntl.flock(handle,fcntl.LOCK_EX) for handle in handles]; "
                "print('held',flush=True); "
                "\nfor command in sys.stdin:\n"
                " if command.strip() == 'release':\n"
                "  [fcntl.flock(handle,fcntl.LOCK_UN) for handle in lifecycle]\n"
                "  print('released',flush=True)\n"
                " elif command.strip() == 'reacquire':\n"
                "  [fcntl.flock(handle,fcntl.LOCK_EX) for handle in lifecycle]\n"
                "  print('reacquired',flush=True)\n"
                " elif command.strip() == 'done': break\n"
            ),
            *(str(path) for path in lock_paths),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def send(process: subprocess.Popen[str], event: str) -> None:
        assert process.stdin is not None
        process.stdin.write(event + "\n")
        process.stdin.flush()

    def expect(stream, event: str) -> None:
        readable, _, _ = select.select([stream], [], [], 10)
        assert readable, f"timed out waiting for {event}"
        assert stream.readline() == event + "\n"

    status_process: subprocess.Popen[str] | None = None
    try:
        assert holder.stdout is not None
        expect(holder.stdout, "held")
        if refresh:
            send(holder, "release")
            expect(holder.stdout, "released")

        command = (
            "import sys; import zeta.accounts.store as store, zeta.cli.main as cli; "
            "store.OAUTH_REFRESH_TIMEOUT_SECONDS=cli.OAUTH_REFRESH_TIMEOUT_SECONDS=0.05; "
            + (
                "real=cli._read_safe_account_status; first=True; "
                "\nasync def read(credentials):\n"
                " global first\n result=await real(credentials)\n"
                " if first:\n  first=False\n"
                "  print('initial-read',file=sys.stderr,flush=True)\n"
                "  sys.stdin.readline()\n"
                " return result\n"
                "\ncli._read_safe_account_status=read; "
                if refresh
                else ""
            )
            + "print('started',file=sys.stderr,flush=True); sys.stdin.readline(); "
            "raise SystemExit(cli.main(['accounts','status',"
            + ("'--refresh'," if refresh else "")
            + "'--json']))"
        )
        status_process = subprocess.Popen(
            [sys.executable, "-c", command],
            env={**os.environ, "ZETA_HOME": str(tmp_path)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert status_process.stderr is not None
        expect(status_process.stderr, "started")
        send(status_process, "run")
        if refresh:
            expect(status_process.stderr, "initial-read")
            send(holder, "reacquire")
            expect(holder.stdout, "reacquired")
            send(status_process, "continue")

        stdout, stderr = status_process.communicate(timeout=2)
        assert stderr == ""
        assert status_process.returncode == 1
        rows = json.loads(stdout)
        if refresh:
            assert {row["refresh"]["code"] for row in rows} == {"lock_timeout"}
        else:
            assert {row["error"] for row in rows} == {"lock_timeout"}
    finally:
        if status_process is not None and status_process.poll() is None:
            status_process.kill()
            status_process.wait(timeout=10)
        send(holder, "done")
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
    monkeypatch.setattr(cli_main, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.1, raising=False)
    fallback_release = threading.Timer(1, release.touch)
    fallback_release.start()

    try:
        assert _run_json(tmp_path, monkeypatch) == 1
        # Status must answer while the other process still holds the lock.
        # Only the 1 s fallback timer can release it before this point.
        lock_was_still_held = not release.exists()
    finally:
        release.touch()
        fallback_release.cancel()
        fallback_release.join(timeout=2)
        process.wait(timeout=10)
    assert process.returncode == 0
    assert lock_was_still_held
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
                ],
            )
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert ready.exists()
        return result

    monkeypatch.setattr(cli_main, "_read_safe_account_status", read_then_lock)
    monkeypatch.setattr(account_store_module, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.01)

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
    monkeypatch.setattr(cli_main, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.01, raising=False)

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


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
def test_accounts_status_unreadable_login_is_unknown_not_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_path = _save_account(tmp_path)
    os.chmod(credential_path, 0)
    try:
        with pytest.raises(OSError, match=r"could not be read \(EACCES") as caught:
            named_account_credentials(tmp_path, "codex", "default").read()
        assert "login" not in str(caught.value)

        assert _run_json(tmp_path, monkeypatch, "--refresh") == 1
        assert json.loads(capsys.readouterr().out) == [
            {
                "provider": "codex",
                "account": "default",
                "is_default": True,
                "logged_in": None,
                "access_token_expires_at": None,
                "error": "unreadable_credentials",
                "retryable": False,
                "refresh": {
                    "ok": False,
                    "code": "unreadable_credentials",
                    "retryable": False,
                },
            }
        ]
        assert cli_main.main(["accounts", "status"]) == 1
        assert capsys.readouterr().out.strip() == (
            "codex:default  login unknown (unreadable_credentials)"
            "  expires unknown  default"
        )
    finally:
        os.chmod(credential_path, 0o600)


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
def test_accounts_status_never_blocks_on_account_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh: bool,
) -> None:
    # Account locks must stay nonblocking. Stored-format lease and shape locks
    # are ordered outside the account lock domain and may block off the event loop.
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
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
            path = os.fsdecode(raw).rstrip("\0")
        if (
            ".storage-format" not in path
            and operation & (fcntl.LOCK_EX | fcntl.LOCK_SH)
            and not operation & fcntl.LOCK_NB
        ):
            blocking.append(operation)
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", nonblocking_only)

    args = ("--refresh",) if refresh else ()
    _run_json(tmp_path, monkeypatch, *args)
    [row] = json.loads(capsys.readouterr().out)
    if refresh:
        assert row["refresh"] == {"ok": True, "code": None, "retryable": None}
    assert blocking == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
def test_startup_with_unreadable_registry_warns_and_keeps_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_account(tmp_path)
    registry = tmp_path / "accounts.json"
    original = registry.read_bytes()
    warning = (
        f"accounts.json is unreadable: {registry}; kept it and every account directory; "
        "move it aside to a backup, then run login again or restore it from a backup, "
        "then restart zeta"
    )
    os.chmod(registry, 0)
    try:
        startup = ("accounts.json is unreadable; kept it and every account directory",)
        assert AccountStore(tmp_path).migrate() == startup
        assert asyncio.run(server_accounts.migrate(tmp_path)) == startup

        monkeypatch.setenv("ZETA_HOME", str(tmp_path))
        assert cli_main.main(["accounts", "status"]) == 1
        assert capsys.readouterr().err == warning + "\n"
    finally:
        os.chmod(registry, 0o600)
    assert registry.read_bytes() == original


def _unreadable_row(home: Path, row: str) -> tuple[Path, str]:
    """Build one startup home whose `row` path cannot be read; return it and the warning."""
    if row == "legacy":
        initialize_storage_home(home)
        path = home / "codex-oauth.json"
        path.write_text(json.dumps(OAuthTokens("a", "r", 1).to_dict()))
    else:
        credential = _save_account(home)
        path = {
            "accounts-root": home / "accounts",
            "provider-dir": credential.parent.parent,
            "account-dir": credential.parent,
            "orphan-dir": credential.parent.with_name("work"),
            "locks-dir": home / "account-locks" / "codex",
            "oauth": credential,
            "pending": credential.with_name("publish-pending.json"),
            "registry": home / "accounts.json",
        }[row]
        if row == "pending":
            path.write_text("{}")
        if row == "orphan-dir":
            path.mkdir()
    if row == "registry":
        return path, "accounts.json is unreadable; kept it and every account directory"
    reason = "could not be read or written (EACCES: Permission denied)"
    return path, f"kept {path}: {reason}; fix its permissions, then restart zeta"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
@pytest.mark.parametrize(
    "row",
    [
        "accounts-root",
        "provider-dir",
        "account-dir",
        "orphan-dir",
        "locks-dir",
        "oauth",
        "legacy",
        "pending",
        "registry",
    ],
)
def test_startup_with_an_unreadable_account_path_warns_and_keeps_every_file(
    tmp_path: Path, row: str
) -> None:
    path, warning = _unreadable_row(tmp_path, row)
    files = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    mode = path.stat().st_mode
    os.chmod(path, 0)
    try:
        assert AccountStore(tmp_path).migrate() == (warning,)
        assert asyncio.run(server_accounts.migrate(tmp_path)) == (warning,)
    finally:
        os.chmod(path, mode)
    assert {item: item.read_bytes() for item in files} == files
    assert path.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
@pytest.mark.parametrize("row", ["oauth", "account-dir", "out-of-files"])
def test_one_unreadable_account_is_listed_unavailable_after_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    default = AccountId("codex", "default")
    work = AccountId("codex", "work")
    _save_account(tmp_path)
    credential = _save_account(tmp_path, name="work")
    store = AccountStore(tmp_path)
    assert store.migrate() == ()
    store.set_default(default)
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="gpt-5.5", cwd=tmp_path)
    try:
        reads = {
            "session start": lambda: SessionAccountContext.load(
                tmp_path, manager, opened.metadata
            ),
            "account_list": store.catalog_snapshot,
            "state accounts": store.state_rows,
        }
        if row == "out-of-files":
            open_directory = account_store_module._open_account_directory

            def exhausted(home_fd: int, account: AccountId):
                if account == work:
                    raise OSError(errno.EMFILE, "Too many open files", "work")
                return open_directory(home_fd, account)

            # test-contract: seam real EMFILE needs the whole process out of descriptors
            monkeypatch.setattr(
                account_store_module, "_open_account_directory", exhausted
            )
            for name, read in reads.items():
                with pytest.raises(OSError) as caught:
                    read()
                assert caught.value.errno == errno.EMFILE, name
            return
        path = credential if row == "oauth" else credential.parent
        mode = path.stat().st_mode
        os.chmod(path, 0)
        try:
            context = reads["session start"]()
            _, displays, _ = reads["account_list"]()
            rows, warnings = reads["state accounts"]()
        finally:
            os.chmod(path, mode)
        assert context.resolve("codex") == default
        assert context.unavailable_accounts == (str(work),)
        assert {item.account: item.available for item in displays} == {
            default: True,
            work: False,
        }
        assert {item["account_id"]: item["status"] for item in rows} == {
            str(default): "selected",
            str(work): "error",
        }
        assert f"{work}: credentials unavailable or expired" in warnings
    finally:
        opened.store.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            "unreadable",
            "OAuth credentials file could not be read (EACCES: Permission denied)",
        ),
        ("out-of-files", "OAuth credentials file was not read"),
    ],
)
def test_unreadable_default_account_ends_cli_start_with_a_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    row: str,
    message: str,
) -> None:
    credential = _save_account(tmp_path)
    assert AccountStore(tmp_path).migrate() == ()
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    monkeypatch.delenv("ZETA_ACCOUNT_HOME", raising=False)
    argv = ["--provider", "codex", "--no-session", "--print", "hi"]
    if row == "out-of-files":

        def exhausted(*_args: object) -> None:
            raise OSError(errno.EMFILE, "Too many open files", "oauth.json")

        # test-contract: seam real EMFILE needs the whole process out of descriptors
        monkeypatch.setattr(account_store_module, "_read_optional_bytes", exhausted)
        assert cli_main.main(argv) == 2
    else:
        mode = credential.stat().st_mode
        os.chmod(credential, 0)
        try:
            assert cli_main.main(argv) == 2
        finally:
            os.chmod(credential, mode)
    err = capsys.readouterr().err
    assert err.startswith("zeta: ") and message in err
    assert "Traceback" not in err


def _outcome(call: Callable[[], object]) -> object:
    """Return the result, or the escaped error, so a crash fails at an assertion."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - the table asserts on what escaped
        return exc


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
@pytest.mark.parametrize(
    "row", ["usage refresh", "accounts refresh", "cli start", "logout", "out-of-files"]
)
def test_one_unreadable_account_fails_alone_with_a_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    row: str,
) -> None:
    work = AccountId("codex", "work")
    credential = _save_account(tmp_path, name="work")
    store = AccountStore(tmp_path)
    assert store.migrate() == ()
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    monkeypatch.delenv("ZETA_ACCOUNT_HOME", raising=False)
    start = ["--provider", "codex", "--no-session", "--print", "hi"]
    if row == "usage refresh":
        good = AccountId("anthropic", "default")
        broken = AccountId("anthropic", "aaa")  # listed first, so it runs first
        _save_account(tmp_path, "anthropic")
        path = _save_account(tmp_path, "anthropic", "aaa")
        fixture = Path(__file__).parent / "fixtures/usage/anthropic_oauth_usage.json"
        reply = json.loads(fixture.read_text())["response"]
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=reply))
        usage = AccountUsage(store, transport=transport)
        os.chmod(path, 0)
        try:
            results = _outcome(lambda: asyncio.run(usage.refresh_stale()))
        finally:
            os.chmod(path, 0o600)
        assert isinstance(results, dict), results
        assert results[good].error is None and results[good].freshness == "now"
        failed = "internal refresh error; retry or restart zeta"
        assert results[broken].error == failed
        saved = json.loads((path.parent / "usage.json").read_text())
        assert (saved["outcome"], saved["last_error"]) == ("failed", failed)
        return
    if row == "out-of-files":
        open_directory = account_store_module.child_directory

        def exhausted(parent_fd: int, name: str, **kwargs: bool):
            if name == work.name:
                raise OSError(errno.EMFILE, "Too many open files", name)
            return open_directory(parent_fd, name, **kwargs)

        # test-contract: seam real EMFILE needs the whole process out of descriptors
        monkeypatch.setattr(account_store_module, "child_directory", exhausted)
        assert _outcome(lambda: cli_main.main(start)) == 2
        err = capsys.readouterr().err
        assert err.startswith("zeta: ") and "Traceback" not in err
        assert f"account {work} directory was not read" in err
        return
    argv = {
        "accounts refresh": ["accounts", "refresh"],
        "cli start": start,
        "logout": ["accounts", "logout", str(work)],
    }[row]
    files = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    os.chmod(credential.parent, 0)
    try:
        code = _outcome(lambda: cli_main.main(argv))
    finally:
        os.chmod(credential.parent, 0o700)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    if row == "accounts refresh":
        assert code == 0, code
        assert f"{work}" in captured.out
    else:
        assert code == 2, code
        message = f"zeta: [Errno 13] account {work} directory could not be read"
        assert message in captured.err
    assert {item: item.read_bytes() for item in files if item.name != "usage.json"} == {
        item: data for item, data in files.items() if item.name != "usage.json"
    }
    assert str(work) in json.loads((tmp_path / "accounts.json").read_text())["accounts"]


def test_startup_with_a_symlinked_login_is_a_bad_login_not_a_read_error(
    tmp_path: Path,
) -> None:
    credential = _save_account(tmp_path)
    target = tmp_path / "elsewhere-oauth.json"
    credential.rename(target)
    credential.symlink_to(target)
    original = target.read_bytes()

    for migrate in (
        AccountStore(tmp_path).migrate,
        lambda: asyncio.run(server_accounts.migrate(tmp_path)),
    ):
        with pytest.raises(AccountError, match="OAuth credentials are invalid; login required"):
            migrate()
    assert credential.is_symlink()
    assert target.read_bytes() == original


def test_startup_out_of_files_is_not_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_account(tmp_path)

    def exhausted(*_args: object) -> None:
        raise OSError(errno.EMFILE, "Too many open files", "default")

    # test-contract: seam real EMFILE needs the whole process out of descriptors
    monkeypatch.setattr(account_store_module, "_open_account_directory", exhausted)
    with pytest.raises(OSError) as caught:
        AccountStore(tmp_path).migrate()
    assert caught.value.errno == errno.EMFILE


def test_startup_failure_without_a_path_names_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = _save_account(tmp_path)
    original = credential.read_bytes()

    def full_disk(*_args: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    # test-contract: seam os.write/fsync ENOSPC carries no file name and needs a full disk
    monkeypatch.setattr(account_store_module, "_open_account_directory", full_disk)
    warning = (
        "kept every account file: account migration failed "
        "(ENOSPC: No space left on device); fix it, then restart zeta"
    )
    assert AccountStore(tmp_path).migrate() == (warning,)
    assert credential.read_bytes() == original


def test_accounts_commands_say_how_to_log_in_when_home_has_no_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialize_storage_home(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    hint = "no accounts; run zeta login <provider> --account <name>\n"

    assert cli_main.main(["accounts", "status"]) == 1
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", hint)

    assert cli_main.main(["accounts"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (hint, "")

    assert cli_main.main(["accounts", "status", "--json"]) == 1
    assert capsys.readouterr().out == "[]\n"


def test_accounts_status_filter_names_the_provider_that_has_no_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initialize_storage_home(tmp_path)
    _save_account(tmp_path, "codex")
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert cli_main.main(["accounts", "status", "anthropic"]) == 1
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (
        "",
        "no anthropic accounts; run zeta login anthropic --account <name>\n",
    )
