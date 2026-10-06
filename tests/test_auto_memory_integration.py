from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from zeta.config.settings import load_settings, resolve
from zeta.core.slash import create_slash_registry
from zeta.server.server import _Client
from zeta.skills import SkillCatalog


def _resolve(home: Path, *, cli_auto_memory: bool | None = None):
    settings = load_settings(home=home).settings
    return resolve(
        settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
        cli_auto_memory=cli_auto_memory,
    )


def test_memory_settings_defaults_and_overrides(tmp_path: Path) -> None:
    defaults = _resolve(tmp_path)
    assert defaults.memory_auto is True
    assert defaults.memory_model == "gpt-5.6-luna"
    assert defaults.memory_token_threshold == 50_000
    assert defaults.memory_idle_minutes == 10

    (tmp_path / "settings.toml").write_text(
        """[memory]
auto = false
model = "gpt-5.6-terra"
token_threshold = 1234
idle_minutes = 2
""",
        encoding="utf-8",
    )
    configured = _resolve(tmp_path)
    assert configured.memory_auto is False
    assert configured.memory_model == "gpt-5.6-terra"
    assert configured.memory_token_threshold == 1234
    assert configured.memory_idle_minutes == 2
    assert _resolve(tmp_path, cli_auto_memory=True).memory_auto is True


def test_memory_slash_command_dispatches_log_and_undo() -> None:
    calls: list[str] = []
    session = SimpleNamespace(
        slash_memory=lambda args: calls.append(args) or f"memory:{args}"
    )
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert registry.dispatch(session, "/memory log") == "memory:log"
    assert registry.dispatch(session, "/memory undo") == "memory:undo"
    assert calls == ["log", "undo"]


async def test_serve_memory_event_requires_negotiated_feature() -> None:
    client = object.__new__(_Client)
    client._closed = False
    client.features = frozenset({"memory_updated"})
    events: list[tuple[str, str | None, dict[str, object]]] = []

    async def notify(
        event: str, session_id: str | None = None, **fields: object
    ) -> None:
        events.append((event, session_id, fields))

    client._notify = notify
    client._publish_memory_notice("session", "memory updated: decisions.md (+1)")
    await asyncio.sleep(0)
    assert events == [
        (
            "memory_updated",
            "session",
            {"message": "memory updated: decisions.md (+1)"},
        )
    ]

    client.features = frozenset()
    client._publish_memory_notice("session", "hidden")
    await asyncio.sleep(0)
    assert len(events) == 1
