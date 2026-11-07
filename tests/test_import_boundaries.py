import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1] / "src" / "zeta"
FORBIDDEN = {
    "codex": {"providers", "tools"},
    "oauth": {"providers", "tools"},
    "runtime": {"tui", "cli"},
    "core": {"providers", "tools", "tui", "cli"},
    "providers": {"tools", "tui", "cli"},
    "tools": {"providers", "tui", "cli"},
    "skills": {"providers", "tui", "cli"},
    "server": {"tui"},
    "tui": {"server"},
}


# These imports are intentionally computed: each name is discovered within its own package.
DYNAMIC_IMPORT_ALLOWLIST = {
    "runtime/__init__.py:21": "lazy-load one of runtime's declared exports",
    "tools/registry.py:108": "load the package being scanned",
    "tools/registry.py:122": "load a discovered tool module",
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
    if relative.as_posix() == "runtime/headless.py":
        # This moved driver owns the existing TUI composition path.
        forbidden = forbidden - {"tui"}
    if not forbidden:
        return []

    violations: list[str] = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    importlib_names = {"importlib"}
    import_module_names = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(statement, ast.ImportFrom) and statement.module == "importlib":
            for alias in statement.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)

    file_package = ("zeta", *relative.with_suffix("").parts)
    if file_package[-1] == "__init__":
        file_package = file_package[:-1]

    def dynamic_target(node: ast.Call) -> tuple[str, ...] | None:
        function = node.func
        is_import_module = (
            isinstance(function, ast.Name) and function.id in import_module_names
        ) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_names
            and function.attr == "import_module"
        )
        is_dunder_import = (
            isinstance(function, ast.Name) and function.id == "__import__"
        )
        if not (is_import_module or is_dunder_import) or not node.args:
            return None
        module = node.args[0]
        if not isinstance(module, ast.Constant) or not isinstance(module.value, str):
            return ()
        name = module.value
        if name.startswith("."):
            package = file_package
            if is_import_module:
                for keyword in node.keywords:
                    if (
                        keyword.arg == "package"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        package = tuple(keyword.value.value.split("."))
            level = len(name) - len(name.lstrip("."))
            package = package[: len(package) - level + 1]
            name = ".".join((*package, name.lstrip(".")))
        target = tuple(name.split("."))
        if is_dunder_import:
            for keyword in node.keywords:
                if keyword.arg == "fromlist" and isinstance(
                    keyword.value, (ast.List, ast.Tuple)
                ):
                    target += tuple(
                        item.value
                        for item in keyword.value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    )
        return target

    for node in ast.walk(tree):
        targets: list[tuple[str, ...]] = []
        if isinstance(node, ast.Import):
            targets = [tuple(alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = _import_from_target(file_path, node, root)
            targets.append(target)
            if node.module is None:
                targets.extend((*target, alias.name) for alias in node.names)
        elif isinstance(node, ast.Call):
            target = dynamic_target(node)
            if target == ():
                marker = f"{relative}:{node.lineno}"
                if marker not in DYNAMIC_IMPORT_ALLOWLIST:
                    violations.append(f"{marker}: non-literal dynamic import")
                continue
            if target is not None:
                targets.append(target)

        for target in targets:
            if len(target) > 1 and target[0] == "zeta" and target[1] in forbidden:
                violations.append(f"{relative}:{node.lineno}: {'.'.join(target)}")
    return violations


def _layer_modules() -> list[str]:
    return sorted(
        _module_name(path)
        for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).parts[0] in FORBIDDEN
        and "tests" not in path.relative_to(ROOT).parts
    )


LAYER_MODULES = _layer_modules()


@pytest.mark.parametrize("module", LAYER_MODULES)
def test_layer_modules_import_in_fresh_processes(module: str) -> None:
    source_root = str(ROOT.parent.parent)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
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
    assert result.returncode == 0, f"{module}:\n{result.stderr}"


def test_layer_module_parameters_cover_discovered_modules() -> None:
    assert LAYER_MODULES == _layer_modules()


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
        if "tests" not in file_path.relative_to(mutated_root).parts
        for violation in _forbidden_imports(file_path, mutated_root)
    ]

    assert any(
        "core/loop.py" in violation and "zeta.tools" in violation
        for violation in violations
    ), "\n".join(violations)


def test_dynamic_import_boundaries_have_teeth(tmp_path: Path) -> None:
    root = tmp_path / "zeta"
    (root / "core").mkdir(parents=True)
    path = root / "core" / "fixture.py"
    path.write_text(
        """__import__("zeta.tools")
__import__("zeta", fromlist=["tools"])
import importlib as il; il.import_module("zeta.tools")
from importlib import import_module as load; load("zeta.tools")
name = "zeta.tools"; load(name)
value = "tools"; __import__(f"zeta.{value}")
""",
        encoding="utf-8",
    )

    violations = _forbidden_imports(path, root)

    assert sum("zeta.tools" in violation for violation in violations) == 4
    assert (
        sum("non-literal dynamic import" in violation for violation in violations) == 2
    )


def test_websearch_import_does_not_load_providers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zeta.tools.websearch; "
                "print([m for m in sys.modules if m.startswith('zeta.providers')])"
            ),
        ],
        cwd=ROOT.parent.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_production_does_not_import_test_backends() -> None:
    violations: list[str] = []
    for file_path in ROOT.rglob("*.py"):
        if "tests" in file_path.relative_to(ROOT).parts:
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "tests" or name.startswith("tests.") for name in names):
                violations.append(f"{file_path.relative_to(ROOT)}:{node.lineno}")

    assert not violations, "production imports test code:\n" + "\n".join(violations)


def test_import_boundaries() -> None:
    violations = [
        violation
        for file_path in ROOT.rglob("*.py")
        if "tests" not in file_path.relative_to(ROOT).parts
        for violation in _forbidden_imports(file_path)
    ]
    assert not violations, "\n".join(violations)
