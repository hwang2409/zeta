from __future__ import annotations

import ast
from pathlib import Path


def test_feature_gates_do_not_compare_against_writer_epoch() -> None:
    source_root = Path.cwd() / "src" / "zeta"
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        if path.name == "storage_format.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if any(
                isinstance(child, ast.Name) and child.id == "WRITER_EPOCH"
                for child in ast.walk(node)
            ):
                violations.append(f"{path.relative_to(Path.cwd())}:{node.lineno}")

    assert not violations, (
        "use FEATURE_EPOCHS[<feature>] from storage_format.py: "
        + ", ".join(violations)
    )
