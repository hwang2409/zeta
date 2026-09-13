import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).parents[1] / "src" / "zeta"
FORBIDDEN = {
    "automations": {"tui", "cli"},
    "runtime": {"tui", "cli"},
    "core": {"providers", "tools", "skills", "tui", "cli"},
    "providers": {"tools", "skills", "tui", "cli"},
    "tools": {"providers", "skills", "tui", "cli"},
    "skills": {"providers", "tui", "cli"},
    "server": {"tui"},
    "tui": {"server"},
}


def _module_name(file_path: Path, root: Path = ROOT) -> str:
    relative = file_path.relative_to(root).with_suffix("")
    parts = ("zeta", *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_from_target(
    file_path: Path,
    node: ast.ImportFrom,
    root: Path = ROOT,
) -> tuple[str, ...]:
    relative = file_path.relative_to(root).with_suffix("")
    package = ("zeta", *relative.parts[:-1])
    if node.level == 0:
        return tuple((node.module or "").split("."))
    package = package[: len(package) - node.level + 1]
    if node.module:
        return (*package, *node.module.split("."))
    return package


def _forbidden_imports(file_path: Path, root: Path = ROOT) -> list[str]:
    relative = file_path.relative_to(root)
    bucket = relative.parts[0] if len(relative.parts) > 1 else relative.stem
    forbidden = FORBIDDEN.get(bucket, set())
    if not forbidden:
        return []

    violations: list[str] = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in ast.walk(tree):
        targets: list[tuple[str, ...]] = []
        if isinstance(node, ast.Import):
            targets = [tuple(alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = _import_from_target(file_path, node, root)
            targets.append(target)
            if node.module is None:
                targets.extend((*target, alias.name) for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            targets.append(tuple(node.args[0].value.split(".")))

        for target in targets:
            if len(target) > 1 and target[0] == "zeta" and target[1] in forbidden:
                violations.append(f"{relative}:{node.lineno}: {'.'.join(target)}")
    return violations


def test_layer_modules_import_in_fresh_processes() -> None:
    source_root = str(ROOT.parent.parent)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    modules = sorted(
        _module_name(path)
        for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).parts[0] in FORBIDDEN
    )

    failures: list[str] = []
    for module in modules:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; importlib.import_module(sys.argv[1])",
                module,
            ],
            cwd=ROOT.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            failures.append(f"{module}:\n{result.stderr}")
    assert not failures, "\n".join(failures)


def test_absolute_import_boundary_has_teeth(tmp_path: Path) -> None:
    mutated_root = tmp_path / "zeta"
    shutil.copytree(ROOT, mutated_root)
    loop_path = mutated_root / "core" / "loop.py"
    loop_path.write_text(
        "from zeta.tools import ToolRegistry\n" + loop_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    violations = [
        violation
        for file_path in mutated_root.rglob("*.py")
        for violation in _forbidden_imports(file_path, mutated_root)
    ]

    assert any(
        "core/loop.py" in violation and "zeta.tools" in violation
        for violation in violations
    ), "\n".join(violations)


def test_import_boundaries() -> None:
    violations = [
        violation
        for file_path in ROOT.rglob("*.py")
        for violation in _forbidden_imports(file_path)
    ]
    assert not violations, "\n".join(violations)
