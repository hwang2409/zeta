from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from zeta.tui.app import TUIApp
from zeta.tui.slash_handlers.fork_view import ForkContext, ForkViewMixin

# -- request_open_fork / request_return_to_main keep the main turn alive -----


def test_request_open_fork_suspends_without_aborting_the_turn() -> None:
    aborts: list[str] = []
    app = SimpleNamespace(
        _open_fork_target=None,
        _exit_requested=False,
        _active_session=None,
        abort_active=lambda *a, **k: aborts.append("abort"),
    )
    TUIApp.request_open_fork(app, "fork-123")
    assert app._open_fork_target == "fork-123"
    assert app._exit_requested is True
    # The in-flight main turn is never aborted when a discussion opens.
    assert aborts == []
    assert TUIApp.take_open_fork(app) == "fork-123"
    assert app._open_fork_target is None


def test_request_return_to_main_does_not_abort() -> None:
    aborts: list[str] = []
    app = SimpleNamespace(
        _exit_requested=False,
        _active_session=None,
        abort_active=lambda *a, **k: aborts.append("abort"),
    )
    TUIApp.request_return_to_main(app)
    assert app._exit_requested is True
    assert aborts == []


# -- return arming (review minor #2) ----------------------------------------


class _FakeForkApp(ForkViewMixin):
    def __init__(self, *, item_open: bool) -> None:
        self._fork_context = ForkContext(
            "Pick DB", "aid", "sid", "orchestrator", Path("/tmp/x")
        )
        self._fork_return_armed = False
        self._fork_returning = False
        self._item_open = item_open
        self.printed: list[str] = []
        self.returned = False

    def _fork_item_open(self) -> bool:
        return self._item_open

    def _print_system(self, message: str) -> None:
        self.printed.append(message)

    def request_return_to_main(self) -> None:
        self.returned = True


def test_open_item_arms_once_then_returns() -> None:
    app = _FakeForkApp(item_open=True)
    app._return_to_main()
    assert app._fork_return_armed is True
    assert app.returned is False
    app._return_to_main()
    assert app.returned is True


def test_esc_clears_arm_so_return_asks_again() -> None:
    app = _FakeForkApp(item_open=True)
    app._return_to_main()  # Enter on (main): arms.
    app.clear_fork_return_arm()  # Esc exits navigation.
    assert app._fork_return_armed is False
    app._return_to_main()  # Enter on (main) again: must ask, not leave.
    assert app.returned is False
    assert app._fork_return_armed is True


def test_resolved_item_returns_immediately() -> None:
    app = _FakeForkApp(item_open=False)
    app._return_to_main()
    assert app.returned is True


# -- status-bar main-activity indicator -------------------------------------


class _FakeMain:
    def __init__(self, count: int) -> None:
        self._count = count

    def _notification_count(self) -> int:
        return self._count


class _FakeForkStatus(ForkViewMixin):
    def __init__(self) -> None:
        self._main_app = None
        self._main_notification_baseline = 0


def test_main_activity_pending_tracks_new_notifications() -> None:
    fork = _FakeForkStatus()
    assert fork.main_activity_pending is False  # not attached to a main runtime
    main = _FakeMain(3)
    fork.attach_main(main)
    assert fork.main_activity_pending is False
    main._count = 4  # a main notification arrives while the fork is shown
    assert fork.main_activity_pending is True
