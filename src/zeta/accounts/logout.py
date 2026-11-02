"""Transactional account logout filesystem operations."""

from __future__ import annotations

import os
import stat
import uuid
from typing import TYPE_CHECKING

from ..core.session_files import child_directory
from .store import AccountError

if TYPE_CHECKING:
    from .store import AccountId


def _fsync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)


def _validate_directory_files(directory_fd: int, allowed: set[str]) -> None:
    for name in os.listdir(directory_fd):
        if name not in allowed:
            raise AccountError("account directory contains an unexpected file")
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AccountError("account directory contains an unsafe file")


def _remove_directory_files(directory_fd: int, allowed: set[str]) -> None:
    names = os.listdir(directory_fd)
    _validate_directory_files(directory_fd, allowed)
    for name in names:
        os.unlink(name, dir_fd=directory_fd)
    _fsync_directory(directory_fd)


def _quarantine_account_directory(
    home_fd: int, account: AccountId
) -> str | None:
    try:
        with (
            child_directory(home_fd, "accounts") as accounts_fd,
            child_directory(accounts_fd, account.provider) as provider_fd,
            child_directory(provider_fd, account.name) as account_fd,
        ):
            allowed = {"oauth.json", "usage.json", "publish-pending.json"}
            _validate_directory_files(account_fd, allowed)
            tombstone = f".logout-{uuid.uuid4().hex}"
            os.rename(account.name, tombstone, src_dir_fd=provider_fd, dst_dir_fd=provider_fd)
            _fsync_directory(provider_fd)
            try:
                with child_directory(provider_fd, tombstone) as tombstone_fd:
                    _validate_directory_files(tombstone_fd, allowed)
            except BaseException:
                os.rename(tombstone, account.name, src_dir_fd=provider_fd, dst_dir_fd=provider_fd)
                _fsync_directory(provider_fd)
                raise
            return tombstone
    except FileNotFoundError:
        return None


def restore_account_directory(
    home_fd: int, account: AccountId, tombstone: str
) -> None:
    with (
        child_directory(home_fd, "accounts") as accounts_fd,
        child_directory(accounts_fd, account.provider) as provider_fd,
    ):
        os.rename(tombstone, account.name, src_dir_fd=provider_fd, dst_dir_fd=provider_fd)
        _fsync_directory(provider_fd)


def remove_account_tombstone(
    home_fd: int, account: AccountId, tombstone: str
) -> None:
    with (
        child_directory(home_fd, "accounts") as accounts_fd,
        child_directory(accounts_fd, account.provider) as provider_fd,
        child_directory(provider_fd, tombstone) as account_fd,
    ):
        _remove_directory_files(
            account_fd, {"oauth.json", "usage.json", "publish-pending.json"}
        )
        os.rmdir(tombstone, dir_fd=provider_fd)
        _fsync_directory(provider_fd)


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
            else:
                _validate_directory_files(account_fd, allowed)
            if remove:
                os.rmdir(account.name, dir_fd=provider_fd)
                _fsync_directory(provider_fd)
    except FileNotFoundError:
        return
