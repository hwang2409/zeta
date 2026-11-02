from __future__ import annotations

import multiprocessing
import os
import signal

from zeta.accounts import AccountId, AccountStore


def crash_logout(home: str, account: AccountId, point: str) -> None:
    store = AccountStore(home)
    if point in {"before-commit", "after-commit"}:
        original = AccountStore._write_registry

        def crash(self: AccountStore, home_fd: int, value: dict[str, object]) -> None:
            if point == "after-commit":
                original(self, home_fd, value)
            os.kill(os.getpid(), signal.SIGKILL)

        AccountStore._write_registry = crash
    elif point == "partway-cleanup":
        import zeta.accounts.logout as logout_module

        def crash(directory_fd: int, allowed: set[str]) -> None:
            if "oauth.json" in allowed:
                os.unlink("oauth.json", dir_fd=directory_fd)
                os.kill(os.getpid(), signal.SIGKILL)
            logout_module._remove_directory_files(directory_fd, allowed)

        logout_module._remove_directory_files = crash
    store.logout(account, sessions_root=store.home / "sessions")


def hold_account_credential_lock(
    home: str,
    account: AccountId,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with AccountStore(home).credential_guard(account):
        ready.set()
        release.wait()
