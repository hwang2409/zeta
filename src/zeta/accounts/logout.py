"""Transactional account logout filesystem operations."""

from __future__ import annotations

import os
import stat

from ..core.session_files import child_directory
from .store import AccountError, AccountId


def _fsync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)


def _validate_directory(directory_fd: int, allowed: set[str]) -> None:
    if stat.S_IMODE(os.fstat(directory_fd).st_mode) != 0o700:
        raise AccountError("account directory has unsafe permissions")
    for name in os.listdir(directory_fd):
        if name not in allowed:
            raise AccountError("account directory contains an unexpected file")
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AccountError("account directory contains an unsafe file")


def _remove_directory_files(directory_fd: int, allowed: set[str]) -> None:
    _validate_directory(directory_fd, allowed)
    names = os.listdir(directory_fd)
    for name in names:
        os.unlink(name, dir_fd=directory_fd)
    _fsync_directory(directory_fd)


def validate_account_directory(
    home_fd: int, root: str, account: AccountId, *, allowed: set[str]
) -> None:
    try:
        with (
            child_directory(home_fd, root) as root_fd,
            child_directory(root_fd, account.provider) as provider_fd,
            child_directory(provider_fd, account.name) as account_fd,
        ):
            _validate_directory(account_fd, allowed)
    except FileNotFoundError:
        return


def remove_account_directory(
    home_fd: int, root: str, account: AccountId | tuple[str, str], *, allowed: set[str]
) -> None:
    provider, name = (
        (account.provider, account.name)
        if isinstance(account, AccountId)
        else account
    )
    try:
        with (
            child_directory(home_fd, root) as root_fd,
            child_directory(root_fd, provider) as provider_fd,
            child_directory(provider_fd, name) as account_fd,
        ):
            _remove_directory_files(account_fd, allowed)
            os.rmdir(name, dir_fd=provider_fd)
            _fsync_directory(provider_fd)
    except FileNotFoundError:
        return


def remove_account_lock_directory(
    home_fd: int, account: AccountId, *, remove: bool
) -> None:
    try:
        with (
            child_directory(home_fd, "account-locks") as locks_fd,
            child_directory(locks_fd, account.provider) as provider_fd,
            child_directory(provider_fd, account.name) as account_fd,
        ):
            allowed = {"lifecycle.lock", "usage.lock"}
            if remove:
                _remove_directory_files(account_fd, allowed)
                os.rmdir(account.name, dir_fd=provider_fd)
                _fsync_directory(provider_fd)
            else:
                _validate_directory(account_fd, allowed)
    except FileNotFoundError:
        return
