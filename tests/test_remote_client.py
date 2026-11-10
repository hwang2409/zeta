from __future__ import annotations

import pytest

from zeta.remote.discovery import choose_session, split_target, validate_host
from zeta.remote.transport import remote_bridge_command
from zeta.remote.wire import RemoteWireError, decode_lines


def test_remote_target_and_shell_validation() -> None:
    assert split_target("home:abc") == ("home", "abc")
    with pytest.raises(ValueError):
        validate_host("bad host")
    with pytest.raises(ValueError):
        split_target("home:a/b")
    command = remote_bridge_command("attach", "abc")
    assert 'command -v zeta' in command
    assert '"$HOME/.local/bin/zeta"' in command
    with pytest.raises(ValueError):
        remote_bridge_command("attach", "a;b")


def test_remote_wire_rejects_unbounded_or_unterminated_frames() -> None:
    with pytest.raises(RemoteWireError):
        tuple(decode_lines([b'{"method":"event"}']))
    with pytest.raises(RemoteWireError):
        tuple(decode_lines([b"x" * (256 * 1024 + 1)]))


def test_session_prefix_must_be_unique() -> None:
    from zeta.remote.discovery import RemoteSession

    rows = (RemoteSession("home", "abc1", "first", "idle"), RemoteSession("home", "abc2", "second", "idle"))
    with pytest.raises(ValueError):
        choose_session(rows, "abc")
