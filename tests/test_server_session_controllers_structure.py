from __future__ import annotations

import ast
from pathlib import Path

from zeta.server.session_controllers import SessionControllers

ROOT = Path(__file__).parents[1] / "src" / "zeta" / "server"


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
    assert {"_control_for_test", "_runtime_for_test"} <= names


def test_record_identity_survives_transitions() -> None:
    class Runtime:
        session_id = "s"

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
