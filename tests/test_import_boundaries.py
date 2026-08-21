import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "zeta"
FORBIDDEN = {
    "core": {"providers", "tools", "skills", "tui", "cli"},
    "providers": {"tools", "skills", "tui", "cli"},
    "tools": {"providers", "skills", "tui", "cli"},
    "skills": {"providers", "tui", "cli"},
}


def _target_parts(file_path: Path, node: ast.ImportFrom) -> tuple[str, ...]:
    relative = file_path.relative_to(ROOT).with_suffix("")
    package = ("zeta", *relative.parts[:-1])
    if node.level:
        package = package[: len(package) - node.level + 1]
    if node.module:
        package = (*package, *node.module.split("."))
    return package


def test_import_boundaries() -> None:
    violations: list[str] = []
    for file_path in ROOT.rglob("*.py"):
        relative = file_path.relative_to(ROOT)
        bucket = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        forbidden = FORBIDDEN.get(bucket, set())
        if not forbidden:
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name.split(".")
                    if (
                        len(target) > 1
                        and target[0] == "zeta"
                        and target[1] in forbidden
                    ):
                        violations.append(
                            f"{relative}:{node.lineno}: import {alias.name}"
                        )
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _target_parts(file_path, node)
            if len(target) > 1 and target[0] == "zeta" and target[1] in forbidden:
                imported = f"from {'.' * node.level}{node.module or ''} import ..."
                violations.append(f"{relative}:{node.lineno}: {imported}")
            for alias in node.names:
                if alias.name not in forbidden:
                    continue
                if node.module is None:
                    imported = f"from {'.' * node.level} import {alias.name}"
                    violations.append(f"{relative}:{node.lineno}: {imported}")
    assert not violations, "\n".join(violations)
