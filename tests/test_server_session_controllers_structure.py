from __future__ import annotations

import ast
from pathlib import Path

import pytest

from zeta.server.protocol import ProtocolError
from zeta.server.session_controllers import SessionControllers

ROOT = Path(__file__).parents[1] / "src" / "zeta" / "server"
# the only request entries that resolve a record; everything below them takes it
RESOLVER_ENTRIES = {
    "handlers/registry.py:dispatch",
    "handlers/registry.py:session_history_and_subscribe",
    "handlers/registry.py:child_run_events",
    "handlers/registry.py:path_complete",
    "handlers/registry.py:child_run_list",
    "connections.py:ConnectionRegistry.read_tree",
    "connections.py:ConnectionRegistry.read_history",
    "connections.py:ConnectionRegistry.read_child_run_history",
    "server.py:_Client._route_request",
}


def _resolver_calls() -> dict[str, list[int]]:
    """``path:function`` -> lines of its ``controllers.owned/selected`` calls,
    counting lambdas as their enclosing function."""
    calls: dict[str, list[int]] = {}

    def visit(node: ast.AST, path: str, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, path, f"{scope}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{path}:{scope}{child.name}"
                calls.setdefault(name, [])
                visit_body(child, path, name)

    def visit_body(function: ast.AST, path: str, name: str) -> None:
        for child in ast.iter_child_nodes(function):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(ast.Module(body=[child], type_ignores=[]), path, name.split(":", 1)[1] + ".")
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"owned", "selected"}
                and isinstance(child.func.value, ast.Attribute)
                and child.func.value.attr in {"controllers", "_controllers"}
            ):
                calls[name].append(child.lineno)
            visit_body(child, path, name)

    for file in sorted(ROOT.rglob("*.py")):
        visit(ast.parse(file.read_text()), file.relative_to(ROOT).as_posix(), "")
    return calls


def test_record_indexes_have_one_owner() -> None:
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                attrs = [child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)]
                text = ast.unparse(node)
                if (
                    any(name in attrs for name in ("_parked", "_attached", "_records"))
                    and ("SessionController" in text or "ServeControl" in text or "_parked" in attrs)
                ):
                    assert path.name == "session_controllers.py", f"{path}:{node.lineno}: {text}"


def test_server_has_only_private_test_aliases() -> None:
    tree = ast.parse((ROOT / "server.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZetaServer")
    names = {node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "control" not in names
    assert "runtime" not in names
    assert "__getattr__" not in names
    assert "_control" + "_for_test" not in names
    assert "_runtime" + "_for_test" not in names


def test_records_are_resolved_once_at_request_entries() -> None:
    calls = _resolver_calls()
    assert RESOLVER_ENTRIES <= calls.keys(), sorted(RESOLVER_ENTRIES - calls.keys())
    misplaced = [
        f"{name}:{line}"
        for name, lines in calls.items()
        if name not in RESOLVER_ENTRIES
        for line in lines
    ]
    repeated = [
        f"{name}:{line}"
        for name, lines in calls.items()
        if len(lines) > 1
        for line in lines
    ]
    assert misplaced == [], "resolver below a request entry: " + ", ".join(misplaced)
    assert repeated == [], "more than one resolver in one entry: " + ", ".join(repeated)


def test_record_identity_survives_transitions() -> None:
    class Runtime:
        session_id = "s"
        opened = object()

    class Control:
        runtime = Runtime()

    control = Control()
    owner = SessionControllers(control)
    record = owner.attached()
    assert owner.attach(record) is record
    assert owner.park(record) is record
    assert owner.unpark(record) is record
    assert owner.attach(record) is record
    owner.park(record)
    owner.remove(record)
    assert owner.records() == ()
    with pytest.raises(ProtocolError):
        owner.selected(None, "s")
