"""Pytest cleanup for short Unix socket paths."""

from __future__ import annotations

import pytest

from tests.support.sockets import cleanup_short_socket_paths


@pytest.fixture(autouse=True)
def cleanup_test_sockets():
    try:
        yield
    finally:
        cleanup_short_socket_paths()
