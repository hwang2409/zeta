from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import zeta.cli.main as cli_main
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, AccountStore
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
