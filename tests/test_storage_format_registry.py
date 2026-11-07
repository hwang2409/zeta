from __future__ import annotations

import ast
from pathlib import Path

import pytest

from zeta.core.store._client_delivery import validate_client_delivery_data
from zeta.core.store._validation import (
    validate_child_idle_notification_data,
    validate_terminal_transition_data,
)
from zeta.core.todo import parse_todo_items
from zeta.storage_format import ACCEPTED_KEY_SETS, compare_registry_dumps


def _assert_exact(validator, payload: dict[str, object], declared: str, *, required: bool = True) -> None:
    validator(payload)
    assert set(payload) == set(ACCEPTED_KEY_SETS[declared])
    extra = {**payload, "_registry_probe_extra": True}
    with pytest.raises((ValueError, TypeError)):
        validator(extra)
    if required:
        for key in payload:
            missing = dict(payload)
            del missing[key]
            with pytest.raises((ValueError, TypeError, KeyError)):
                validator(missing)


def test_every_registry_name_has_one_production_owner() -> None:
    names: dict[str, int] = {name: 0 for name in ACCEPTED_KEY_SETS}
    for path in Path("src").rglob("*.py"):
        if path.name in {"storage_format.py", "ssh.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in names
            ):
                names[node.value] += 1
    assert all(count == 1 for count in names.values()), {
        name: count for name, count in names.items() if count != 1
    }


def test_strict_production_validators_are_bound_to_registry() -> None:
    todo = [{"content": "ship", "status": "pending"}]
    parse_todo_items(todo)
    assert set(todo[0]) == set(ACCEPTED_KEY_SETS["todo_item"])
    with pytest.raises(ValueError):
        parse_todo_items([{**todo[0], "_registry_probe_extra": True}])
    with pytest.raises(ValueError):
        parse_todo_items([{"content": "ship"}])
    _assert_exact(
        validate_child_idle_notification_data,
        {
            "origin": "notification", "kind": "child_idle", "child_instance_id": "child",
            "child_session_path": "/sessions/child", "description": "waiting",
            "status": "child_idle", "text": "waiting", "ask_parent_question_id": None,
            "display_only": True,
        },
        "child_idle_notification", required=False,
    )
    _assert_exact(validate_terminal_transition_data, {"action": "cancel"}, "response_action")
    _assert_exact(
        validate_terminal_transition_data,
        {"action": "finish", "report": {"status": "done", "summary": "ok"}},
        "response_action_report",
    )
    validate_client_delivery_data(
        {"delivery_id": "d" * 32, "method": "steer", "status": "queued", "outcome": {"accepted": True}},
        "session",
    )
    with pytest.raises(ValueError):
        validate_client_delivery_data(
            {"delivery_id": "d" * 32, "method": "steer", "status": "queued", "outcome": {"accepted": True}, "_registry_probe_extra": True},
            "session",
        )


def test_registry_delta_requires_epoch_and_names_the_fix() -> None:
    predecessor = {"writer_epoch": 3, "accepted_key_sets": {"todo_item": ["content", "status"]}}
    candidate = {"writer_epoch": 3, "accepted_key_sets": {"todo_item": ["content", "status", "extra"]}}
    with pytest.raises(ValueError, match="bump the epoch in src/zeta/storage_format.py"):
        compare_registry_dumps(predecessor, candidate)
