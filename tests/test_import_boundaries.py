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


def _import_statement_aliases(statement: ast.Import) -> set[str]:
    return {
        alias.asname or alias.name
        for alias in statement.names
        if alias.name == "importlib"
    }


def _import_from_aliases(statement: ast.ImportFrom) -> set[str]:
    if statement.module != "importlib":
        return set()
    return {
        alias.asname or alias.name
        for alias in statement.names
        if alias.name == "import_module"
    }


def _import_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            importlib_names.update(_import_statement_aliases(statement))
        elif isinstance(statement, ast.ImportFrom):
            import_module_names.update(_import_from_aliases(statement))
    return importlib_names, import_module_names


def _relative_target(name: str, package: tuple[str, ...]) -> tuple[str, ...]:
    level = len(name) - len(name.lstrip("."))
    package = package[: len(package) - level + 1]
    return (*package, *name.lstrip(".").split("."))


def _package_argument(node: ast.Call) -> ast.AST | None:
    if len(node.args) > 1:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg == "package":
            return keyword.value
    return None


def _is_name_derived_package(argument: ast.AST) -> bool:
    if isinstance(argument, ast.Name):
        return argument.id in {"__package__", "__name__"}
    if isinstance(argument, ast.Subscript):
        argument = argument.value
    return (
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Attribute)
        and isinstance(argument.func.value, ast.Name)
        and argument.func.value.id == "__name__"
        and argument.func.attr in {"partition", "rpartition", "rsplit"}
    )


def _dynamic_kind(
    node: ast.Call, importlib_names: set[str], import_module_names: set[str]
) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        if function.id == "__import__":
            return "__import__"
        if function.id in import_module_names:
            return "import_module"
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in importlib_names
        and function.attr == "import_module"
    ):
        return "import_module"
    return None


def _literal_module(node: ast.Call) -> str | None:
    if not node.args:
        return None
    module = node.args[0]
    if isinstance(module, ast.Constant) and isinstance(module.value, str):
        return module.value
    return ""


def _resolve_relative_name(
    name: str, node: ast.Call, kind: str, file_package: tuple[str, ...]
) -> str | None:
    if not name.startswith("."):
        return name
    package = file_package
    if kind == "import_module":
        package_argument = _package_argument(node)
        if isinstance(package_argument, ast.Constant) and isinstance(
            package_argument.value, str
        ):
            package = tuple(package_argument.value.split("."))
        elif not _is_name_derived_package(package_argument):
            return None
    if not package:
        return None
    return ".".join(_relative_target(name, package))


def _fromlist_argument(node: ast.Call) -> ast.AST | None:
    if len(node.args) > 3:
        return node.args[3]
    for keyword in node.keywords:
        if keyword.arg == "fromlist":
            return keyword.value
    return None


def _fromlist_names(node: ast.Call) -> tuple[str, ...] | None:
    argument = _fromlist_argument(node)
    if argument is None:
        return ()
    if not isinstance(argument, (ast.List, ast.Tuple)):
        return None
    return tuple(
        item.value
        for item in argument.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )


def _dynamic_target(
    node: ast.Call,
    importlib_names: set[str],
    import_module_names: set[str],
    file_package: tuple[str, ...],
) -> tuple[str, ...] | None:
    kind = _dynamic_kind(node, importlib_names, import_module_names)
    if kind is None or not node.args:
        return None
    name = _literal_module(node)
    if name is None or name == "":
        return ()
    resolved = _resolve_relative_name(name, node, kind, file_package)
    if resolved is None:
        return ()
    target = tuple(resolved.split("."))
    if kind == "__import__":
        fromlist = _fromlist_names(node)
        if fromlist is None:
            return ()
        target += fromlist
    return target


def _has_reason_marker(source_line: str) -> bool:
    marker = "# import-boundary:"
    reason = source_line.split(marker, 1)[1].strip() if marker in source_line else ""
    return bool(reason) and not reason.startswith("#")


def _node_targets(
    node: ast.AST,
    file_path: Path,
    root: Path,
    importlib_names: set[str],
    import_module_names: set[str],
    file_package: tuple[str, ...],
) -> tuple[list[tuple[str, ...]], bool]:
    if isinstance(node, ast.Import):
        return [tuple(alias.name.split(".")) for alias in node.names], False
    if isinstance(node, ast.ImportFrom):
        target = _import_from_target(file_path, node, root)
        names = [target]
        if node.module is None:
            names.extend((*target, alias.name) for alias in node.names)
        return names, False
    if not isinstance(node, ast.Call):
        return [], False
    target = _dynamic_target(node, importlib_names, import_module_names, file_package)
    if target == ():
        return [], True
    return ([target] if target is not None else []), False


def _forbidden_target(
    relative: Path, node: ast.AST, target: tuple[str, ...], forbidden: set[str]
) -> str | None:
    if len(target) > 1 and target[0] == "zeta" and target[1] in forbidden:
        return f"{relative}:{node.lineno}: {'.'.join(target)}"
    return None


def _forbidden_imports(file_path: Path, root: Path = ROOT) -> list[str]:
    relative = file_path.relative_to(root)
    bucket = relative.parts[0] if len(relative.parts) > 1 else relative.stem
    forbidden = FORBIDDEN.get(bucket, set())
    if relative.as_posix() == "runtime/headless.py":
        # This moved driver owns the existing TUI composition path.
        forbidden = forbidden - {"tui"}
    if not forbidden:
        return []

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    importlib_names, import_module_names = _import_aliases(tree)
    file_package = ("zeta", *relative.with_suffix("").parts)
    if relative.stem != "__init__":
        file_package = file_package[:-1]
    lines = source.splitlines()
    violations: list[str] = []
    for node in ast.walk(tree):
        targets, dynamic = _node_targets(
            node, file_path, root, importlib_names, import_module_names, file_package
        )
        if dynamic:
            if not _has_reason_marker(lines[node.lineno - 1]):
                violations.append(
                    f"{relative}:{node.lineno}: non-literal dynamic import"
                )
            continue
        violations.extend(
            violation
            for target in targets
            if (violation := _forbidden_target(relative, node, target, forbidden))
        )
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
    core_path = mutated_root / "core" / "session.py"
    core_path.write_text(
        "from zeta.tools import ToolRegistry\n" + core_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    violations = [
        violation
        for file_path in mutated_root.rglob("*.py")
        if "tests" not in file_path.relative_to(mutated_root).parts
        for violation in _forbidden_imports(file_path, mutated_root)
    ]

    assert any(
        "core/session.py" in violation and "zeta.tools" in violation
        for violation in violations
    ), "\n".join(violations)


def test_dynamic_import_boundaries_have_teeth(tmp_path: Path) -> None:
    root = tmp_path / "zeta"
    (root / "core").mkdir(parents=True)
    path = root / "core" / "fixture.py"
    path.write_text(
        """__import__("zeta.tools")
__import__("zeta", fromlist=["tools"])
__import__("zeta", None, None, ["tools"])
parts = ["tools"]; __import__("zeta", fromlist=parts)
import importlib as il; il.import_module("zeta.tools")
from importlib import import_module as load; load("zeta.tools")
name = "zeta.tools"; load(name)
value = "tools"; __import__(f"zeta.{value}")
load("..tools", __package__)
load("..tools", package=__package__)
""",
        encoding="utf-8",
    )

    violations = _forbidden_imports(path, root)

    assert sum("zeta.tools" in violation for violation in violations) == 7
    assert (
        sum("non-literal dynamic import" in violation for violation in violations) == 3
    )


def test_dynamic_import_reason_markers_are_line_stable(tmp_path: Path) -> None:
    root = tmp_path / "zeta"
    (root / "core").mkdir(parents=True)
    path = root / "core" / "fixture.py"
    path.write_text(
        "from importlib import import_module as load\nload(name)  # import-boundary: discovered by the fixture\n",
        encoding="utf-8",
    )

    assert _forbidden_imports(path, root) == []

    path.write_text(
        "from importlib import import_module as load\nload(name)  # import-boundary:\n",
        encoding="utf-8",
    )
    assert len(_forbidden_imports(path, root)) == 1

    path.write_text(
        "from importlib import import_module as load\n"
        "load(name)  # import-boundary:  # fmt: skip\n",
        encoding="utf-8",
    )
    assert len(_forbidden_imports(path, root)) == 1

    path.write_text("\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(_forbidden_imports(path, root)) == 1


def test_name_derived_relative_import_is_checked(tmp_path: Path) -> None:
    root = tmp_path / "zeta"
    (root / "core").mkdir(parents=True)
    path = root / "core" / "fixture.py"
    path.write_text(
        "from importlib import import_module as load\n"
        'load("..tools", package=__name__.rpartition(".")[0])\n',
        encoding="utf-8",
    )

    violations = _forbidden_imports(path, root)

    assert any("zeta.tools" in violation for violation in violations)


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
