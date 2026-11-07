"""Reject new low-signal test patterns found on added diff lines."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIME_ERROR = (
    "wait on a readiness event or inject a clock; timeouts are hang guards only "
    "(use '# test-contract: latency <reason>' on this line or the line above only "
    "when real product latency is the contract)"
)
SHORT_TIMEOUT_ERROR = (
    "use a generous hang guard (>= 10 s) and wait on the event; use an injected "
    "clock for timeout behavior"
)
DIGEST_ERROR = "assert semantic fields, not a hash"
CONTRACT_ERROR = (
    "assert through <public interface> or mark the contract with "
    "'# test-contract: complexity <reason>', 'durability <reason>', or "
    "'seam <reason>' on this line or the line above"
)
CONTRACT_KINDS = {"bootstrap", "complexity", "durability", "latency", "seam"}
ASYNC_BLOCKING_CODE = "TQ007"
ASYNC_BLOCKING_ERROR = (
    "sync filesystem call {call} in async def {function} blocks the event loop; "
    "move it off the loop with `await asyncio.to_thread(...)`, or mark a reviewed "
    "exception with `# async-blocking: <reason>`"
)
ASYNC_PATH_METHODS = {
    "is_dir",
    "is_file",
    "exists",
    "stat",
    "lstat",
    "open",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "mkdir",
    "unlink",
    "rename",
    "replace",
    "iterdir",
    "glob",
    "rglob",
    "resolve",
    "touch",
}
ASYNC_OS_FUNCTIONS = {
    "stat",
    "lstat",
    "open",
    "scandir",
    "listdir",
    "makedirs",
    "remove",
    "unlink",
    "rename",
    "replace",
}
ASYNC_SHUTIL_FUNCTIONS = {
    "copy",
    "copy2",
    "copyfile",
    "copytree",
    "move",
    "rmtree",
    "make_archive",
    "unpack_archive",
}
ASYNC_PATH_NAME_HINTS = {"path", "directory", "root", "cwd", "file", "dir"}
GLOBAL_BOOTSTRAP_ERROR = (
    "global test bootstrap {name} hides production startup; use a local fixture "
    "and a real subprocess test; use '# test-contract: bootstrap <reason>' on "
    "this line or the line above for an intended session-wide harness setting"
)
MIN_HANG_GUARD_SECONDS = 10
AWAIT_DEADLINE_ERROR = (
    "await without a deadline: use `await await_or_fail(x, what=...)` "
    "(tests/support/owned.py), or asyncio.wait_for(x, 10)"
)
SYNC_DEADLINE_ERROR = "blocking wait without a deadline: pass timeout=10"
AWAIT_DEADLINE_CODE = "TQ004"
RATCHET_PATH = Path("tests/test_quality_ratchet.txt")
GLOBAL_BOOTSTRAP_ALLOWLIST = {
    "ZETA_TEST_PARENT_PID",  # harness identity for child-process ownership checks
    "ZETA_STALL_WATCHDOG",  # disables an unrelated watchdog during test setup
    "ZETA_TEST_AUDIT_LEDGER",  # passes the fixture's audit destination to children
    "ZETA_TEST_LIVE_HOME",  # identifies the guarded live-home fixture
    "ZETA_TEST_OWNERSHIP_TOKEN",  # authenticates writes from guarded children
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


def _normalized_diff_line(line: str) -> str:
    return re.sub(r"\s+", "", line)


def _diff_header_path(value: str) -> Path | None:
    if value == "/dev/null":
        return None
    return Path(value[2:] if value.startswith(("a/", "b/")) else value)


def _added_lines(diff: str) -> dict[Path, set[int]]:
    added_by_path: dict[Path, list[tuple[int, str]]] = {}
    deleted_by_path: dict[Path, list[str]] = {}
    source_path: Path | None = None
    destination_path: Path | None = None
    new_line = 0
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            source_path = None
            destination_path = None
            new_line = 0
            header = line.removeprefix("diff --git ")
            if header.startswith("a/") and " b/" in header:
                source, destination = header.split(" b/", 1)
                source_path = _diff_header_path(source)
                destination_path = _diff_header_path(f"b/{destination}")
            continue
        if line.startswith("--- "):
            source_path = _diff_header_path(line[4:])
            continue
        if line.startswith("+++ "):
            destination_path = _diff_header_path(line[4:])
            if destination_path is not None:
                added_by_path.setdefault(destination_path, [])
                deleted_by_path.setdefault(destination_path, [])
            continue
        match = hunk.match(line)
        if match:
            new_line = int(match.group(1))
            continue
        if destination_path is None:
            continue
        if line.startswith("+"):
            added_by_path[destination_path].append(
                (new_line, _normalized_diff_line(line[1:]))
            )
            new_line += 1
        elif line.startswith("-") and source_path is not None:
            deleted_by_path[destination_path].append(_normalized_diff_line(line[1:]))
        elif line.startswith(" "):
            new_line += 1

    changed: dict[Path, set[int]] = {}
    for changed_path, additions in added_by_path.items():
        unmatched_deletions = deleted_by_path[changed_path].copy()
        changed[changed_path] = set()
        for line_number, normalized in additions:
            if normalized in unmatched_deletions:
                unmatched_deletions.remove(normalized)
            else:
                changed[changed_path].add(line_number)
    return changed


def _is_test_path(path: Path) -> bool:
    return path.suffix == ".py" and (
        path.name.startswith("test_") or "tests" in path.parts
    )


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _marker(lines: list[str], line: int) -> tuple[str, str] | None:
    for candidate in (line, line - 1):
        if candidate < 1 or candidate > len(lines):
            continue
        marker_line = lines[candidate - 1]
        if candidate == line - 1 and not marker_line.lstrip().startswith("#"):
            continue
        match = re.search(
            r"#\s*test-contract:\s*([a-z-]+)(?:\s+([^#\s].*?))?\s*$",
            marker_line,
        )
        if match and match.group(1) in CONTRACT_KINDS and match.group(2):
            return match.group(1), match.group(2).strip()
        match = re.search(r"#\s*async-blocking:\s*([^#\s].*?)\s*$", marker_line)
        if match:
            return "async-blocking", match.group(1).strip()
    return None


def _has_marker(lines: list[str], line: int, contracts: set[str]) -> bool:
    marker = _marker(lines, line)
    return marker is not None and marker[0] in contracts


def _touches_added_line(node: ast.AST, added: set[int]) -> bool:
    end = getattr(node, "end_lineno", node.lineno)
    return any(node.lineno <= line <= end for line in added)


def _contains_timing_reference(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _name(child.func) in {
            "time.monotonic",
            "time.perf_counter",
            "monotonic",
            "perf_counter",
        }:
            return True
        if isinstance(child, (ast.Name, ast.Attribute)):
            identifier = child.id if isinstance(child, ast.Name) else child.attr
            if "elapsed" in identifier.lower():
                return True
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == "elapsed"
        ):
            return True
    return False


def _serialized_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not any(
            isinstance(part, ast.Call)
            and _name(part.func) in {"json.dumps", "pickle.dumps"}
            for part in ast.walk(node.value)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _is_serialized_digest(node: ast.AST, serialized_names: set[str]) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or _name(child.func) not in {
            "hashlib.sha256",
            "sha256",
        }:
            continue
        if not child.args:
            continue
        digest_input = child.args[0]
        if (
            isinstance(digest_input, ast.Name) and digest_input.id in serialized_names
        ) or any(
            isinstance(part, ast.Call)
            and _name(part.func) in {"json.dumps", "pickle.dumps"}
            for part in ast.walk(digest_input)
        ):
            return True
    return False


def _literal_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _is_literal_reference(node: ast.AST, literal_names: set[str]) -> bool:
    return isinstance(node, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)) or (
        isinstance(node, ast.Name) and node.id in literal_names
    )


def _private_patch(node: ast.Call) -> bool:
    function = _name(node.func)
    if function.endswith(".setattr") and len(node.args) >= 2:
        target, name = node.args[:2]
        return (isinstance(name, ast.Constant) and str(name.value).startswith("_")) or (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value.rsplit(".", 1)[-1].startswith("_")
        )
    if function in {"patch", "unittest.mock.patch"} and node.args:
        target = node.args[0]
        return (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value.rsplit(".", 1)[-1].startswith("_")
        )
    if function.endswith("patch.object") and len(node.args) >= 2:
        name = node.args[1]
        return isinstance(name, ast.Constant) and str(name.value).startswith("_")
    return False


def _mock_contract_assertion(node: ast.AST) -> bool:
    count_names = {"call_count", "await_count", "mock_calls", "call_args_list"}
    assert_methods = (
        "assert_called",
        "assert_not_called",
        "assert_awaited",
        "assert_not_awaited",
        "assert_has_calls",
        "assert_has_awaits",
    )
    if isinstance(node, ast.Assert):
        return any(
            isinstance(child, ast.Attribute) and child.attr in count_names
            for child in ast.walk(node.test)
        )
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        method = _name(node.value.func).rsplit(".", 1)[-1]
        return method.startswith(assert_methods)
    return False


def _wait_timeout(node: ast.Call) -> float | None:
    function = _name(node.func)
    if function not in {"asyncio.wait_for", "wait_for", "await_or_fail"}:
        return None
    if function in {"asyncio.wait_for", "wait_for"} and len(node.args) > 1:
        return _literal_number(node.args[1])
    for keyword in node.keywords:
        if keyword.arg == "timeout":
            return _literal_number(keyword.value)
    return None


def _timing_findings(path: Path, node: ast.AST, lines: list[str]) -> list[Finding]:
    if isinstance(node, ast.Call):
        timeout = _wait_timeout(node)
        if (
            timeout is not None
            and 0 < timeout < MIN_HANG_GUARD_SECONDS
            and not _has_marker(lines, node.lineno, {"latency"})
        ):
            return [
                Finding(
                    path, node.lineno, node.col_offset + 1, "TQ001", SHORT_TIMEOUT_ERROR
                )
            ]
        if _name(node.func) in {"time.sleep", "asyncio.sleep"}:
            delay = _literal_number(node.args[0]) if node.args else None
            if (
                delay is not None
                and delay > 0
                and not _has_marker(lines, node.lineno, {"latency"})
            ):
                return [
                    Finding(path, node.lineno, node.col_offset + 1, "TQ001", TIME_ERROR)
                ]
    if (
        isinstance(node, ast.Assert)
        and any(isinstance(child, ast.Compare) for child in ast.walk(node.test))
        and _contains_timing_reference(node.test)
        and not _has_marker(lines, node.lineno, {"latency"})
    ):
        return [Finding(path, node.lineno, node.col_offset + 1, "TQ001", TIME_ERROR)]
    return []


def _digest_findings(
    path: Path, node: ast.AST, literal_names: set[str], serialized_names: set[str]
) -> list[Finding]:
    if not isinstance(node, ast.Assert):
        return []
    comparisons = [
        child for child in ast.walk(node.test) if isinstance(child, ast.Compare)
    ]
    violates = any(
        (
            _is_serialized_digest(compare.left, serialized_names)
            and any(
                _is_literal_reference(value, literal_names)
                for value in compare.comparators
            )
        )
        or (
            any(
                _is_serialized_digest(value, serialized_names)
                for value in compare.comparators
            )
            and _is_literal_reference(compare.left, literal_names)
        )
        for compare in comparisons
    )
    if not violates:
        return []
    return [Finding(path, node.lineno, node.col_offset + 1, "TQ002", DIGEST_ERROR)]


def _contract_findings(path: Path, node: ast.AST, lines: list[str]) -> list[Finding]:
    violates = (isinstance(node, ast.Call) and _private_patch(node)) or (
        _mock_contract_assertion(node)
    )
    if not violates or _has_marker(
        lines, node.lineno, {"complexity", "durability", "seam"}
    ):
        return []
    return [Finding(path, node.lineno, node.col_offset + 1, "TQ003", CONTRACT_ERROR)]


def _enclosing_nodes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    ancestors: list[ast.AST] = []
    while node in parents:
        node = parents[node]
        ancestors.append(node)
    return ancestors


def _inside_timeout_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    return any(
        isinstance(ancestor, ast.AsyncWith)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _name(item.context_expr.func) in {"asyncio.timeout", "timeout"}
            for item in ancestor.items
        )
        for ancestor in _enclosing_nodes(node, parents)
    )


def _inside_deadline_argument(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    child = node
    for ancestor in _enclosing_nodes(node, parents):
        if (
            isinstance(ancestor, ast.Call)
            and _name(ancestor.func)
            in {
                "asyncio.wait_for",
                "wait_for",
                "asyncio.timeout",
                "timeout",
                "await_or_fail",
            }
            and any(child is argument for argument in ancestor.args)
        ):
            return True
        if isinstance(ancestor, (ast.stmt, ast.comprehension)):
            return False
        child = ancestor
    return False


def _enclosing_function_touches_added_line(
    node: ast.AST, added: set[int], parents: dict[ast.AST, ast.AST]
) -> bool:
    return any(
        _touches_added_line(ancestor, added)
        for ancestor in _enclosing_nodes(node, parents)
        if isinstance(ancestor, (ast.AsyncFunctionDef, ast.FunctionDef))
    )


def _is_unbounded_sync_wait(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"join", "wait"}
        and not any(
            isinstance(ancestor, ast.Await)
            for ancestor in _enclosing_nodes(node, parents)
        )
        and not _inside_deadline_argument(node, parents)
    )


def _unwrap_transparent(node: ast.AST) -> ast.AST:
    """Strip asyncio.shield/ensure_future/create_task, zero-arg lambdas, and functools.partial."""
    while True:
        if isinstance(node, ast.Lambda) and not (
            node.args.posonlyargs
            or node.args.args
            or node.args.vararg
            or node.args.kwonlyargs
            or node.args.kwarg
        ):
            node = node.body
            continue
        if (
            isinstance(node, ast.Call)
            and node.args
            and _name(node.func)
            in {
                "asyncio.shield",
                "shield",
                "asyncio.ensure_future",
                "ensure_future",
                "asyncio.create_task",
                "create_task",
                "functools.partial",
                "partial",
            }
        ):
            node = node.args[0]
            continue
        return node


def _is_blocking_to_thread(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or _name(node.func) not in {
        "asyncio.to_thread",
        "to_thread",
    }:
        return False
    if not node.args:
        return False
    target = _unwrap_transparent(node.args[0])
    if isinstance(target, ast.Call):
        target = target.func
    if not isinstance(target, ast.Attribute):
        return False
    if target.attr in {"recv", "recv_into", "read", "readline", "accept"}:
        return True
    return target.attr == "get" and "queue" in _name(target.value).lower()


def _bounded_asyncio_wait(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _name(node.func) == "asyncio.wait"
        and any(
            keyword.arg == "timeout"
            and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None
            )
            for keyword in node.keywords
        )
    )


def _deadline_findings(
    path: Path,
    node: ast.AST,
    lines: list[str],
    parents: dict[ast.AST, ast.AST],
) -> list[Finding]:
    if _is_unbounded_sync_wait(node, parents):
        return (
            []
            if _has_marker(lines, node.lineno, CONTRACT_KINDS)
            else [
                Finding(
                    path,
                    node.lineno,
                    node.col_offset + 1,
                    AWAIT_DEADLINE_CODE,
                    SYNC_DEADLINE_ERROR,
                )
            ]
        )
    if not isinstance(node, ast.Await):
        return []
    value = _unwrap_transparent(node.value)
    violates = (
        isinstance(value, (ast.Name, ast.Attribute))
        or _is_blocking_to_thread(value)
        or (
            isinstance(value, ast.Call)
            and (
                _name(value.func) == "asyncio.gather"
                or (isinstance(value.func, ast.Attribute) and value.func.attr == "wait")
            )
        )
    )
    if (
        not violates
        or _has_marker(lines, node.lineno, CONTRACT_KINDS)
        or _bounded_asyncio_wait(value)
        or _inside_timeout_context(node, parents)
        or _inside_deadline_argument(node, parents)
    ):
        return []
    return [
        Finding(
            path,
            node.lineno,
            node.col_offset + 1,
            AWAIT_DEADLINE_CODE,
            AWAIT_DEADLINE_ERROR,
        )
    ]


def _scan_added_nodes(
    path: Path,
    nodes: list[ast.AST],
    lines: list[str],
    literal_names: set[str],
    serialized_names: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    for node in nodes:
        findings.extend(_timing_findings(path, node, lines))
        findings.extend(_digest_findings(path, node, literal_names, serialized_names))
        findings.extend(_contract_findings(path, node, lines))
    return findings


def _scan_deadline_nodes(
    path: Path,
    nodes: list[ast.AST],
    lines: list[str],
    parents: dict[ast.AST, ast.AST],
) -> list[Finding]:
    return [
        finding
        for node in nodes
        for finding in _deadline_findings(path, node, lines, parents)
    ]


def _deadline_nodes(
    nodes: list[ast.AST],
    added_nodes: list[ast.AST],
    added: set[int],
    parents: dict[ast.AST, ast.AST],
) -> list[ast.AST]:
    return [
        node
        for node in nodes
        if node in added_nodes
        or _enclosing_function_touches_added_line(node, added, parents)
    ]


def _env_name(node: ast.AST, source: str) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value.startswith("ZETA_") else None
    return ast.get_source_segment(source, node) or ast.unparse(node)


def _environment_receivers(tree: ast.AST) -> set[str]:
    receivers = {"os.environ"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for imported in node.names:
                if imported.name == "environ":
                    receivers.add(imported.asname or imported.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _name(node.value) == "os.environ"
        ):
            receivers.add(node.targets[0].id)
    return receivers


def _env_receiver(node: ast.AST, receivers: set[str]) -> str | None:
    receiver = _name(node)
    return receiver if receiver in receivers else None


def _literal_key_names(node: ast.Dict, source: str) -> list[str]:
    return [
        name
        for key in node.keys
        if key is not None
        for name in [_env_name(key, source)]
        if name
    ]


def _assignment_bootstrap_names(
    node: ast.Assign | ast.AugAssign, source: str, receivers: set[str]
) -> list[str]:
    if isinstance(node, ast.AugAssign):
        return (
            _literal_key_names(node.value, source)
            if _env_receiver(node.target, receivers) is not None
            and isinstance(node.value, ast.Dict)
            else []
        )
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
        return []
    target = node.targets[0]
    if _env_receiver(target.value, receivers) is None:
        return []
    name = _env_name(target.slice, source)
    return [name] if name else []


def _direct_environment_call_names(node: ast.Call, source: str) -> list[str]:
    if _name(node.func) != "os.putenv" or not node.args:
        return []
    name = _env_name(node.args[0], source)
    return [name] if name else []


def _monkeypatch_call_names(node: ast.Call, source: str) -> list[str]:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "setenv":
        return []
    receiver = node.func.value
    if _name(receiver) != "monkeypatch" and not (
        isinstance(receiver, ast.Call) and _name(receiver.func) == "MonkeyPatch"
    ):
        return []
    if not node.args:
        return []
    name = _env_name(node.args[0], source)
    return [name] if name else []


def _call_bootstrap_names(
    node: ast.Call, source: str, receivers: set[str]
) -> list[str]:
    if direct := _direct_environment_call_names(node, source):
        return direct
    if monkeypatch := _monkeypatch_call_names(node, source):
        return monkeypatch
    if not isinstance(node.func, ast.Attribute) or not node.args:
        return []
    if _env_receiver(node.func.value, receivers) is None:
        return []
    if node.func.attr == "update" and isinstance(node.args[0], ast.Dict):
        return _literal_key_names(node.args[0], source)
    if node.func.attr == "setdefault":
        name = _env_name(node.args[0], source)
        return [name] if name else []
    return []


def _global_bootstrap_names(
    node: ast.AST, source: str, receivers: set[str]
) -> list[str]:
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        return _assignment_bootstrap_names(node, source, receivers)
    if isinstance(node, ast.Call):
        return _call_bootstrap_names(node, source, receivers)
    return []


def _is_session_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or _name(decorator.func) not in {
            "pytest.fixture",
            "fixture",
        }:
            continue
        return any(
            keyword.arg == "scope"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "session"
            for keyword in decorator.keywords
        )
    return False


def _global_bootstrap_findings(
    path: Path, tree: ast.AST, source: str, added: set[int]
) -> list[Finding]:
    module_scope = path.name in {"conftest.py", "sitecustomize.py"}
    findings: list[Finding] = []
    receivers = _environment_receivers(tree)
    lines = source.splitlines()

    def visit(node: ast.AST, global_scope: bool = module_scope) -> None:
        child_scope = global_scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child_scope = _is_session_fixture(node) or node.name in {
                "pytest_configure",
                "pytest_sessionstart",
                "pytest_load_initial_conftests",
            }
        if (
            global_scope
            and hasattr(node, "lineno")
            and _touches_added_line(node, added)
        ):
            for name in _global_bootstrap_names(node, source, receivers):
                if name in GLOBAL_BOOTSTRAP_ALLOWLIST or _has_marker(
                    lines, node.lineno, {"bootstrap"}
                ):
                    continue
                findings.append(
                    Finding(
                        path,
                        node.lineno,
                        node.col_offset + 1,
                        "TQ005",
                        GLOBAL_BOOTSTRAP_ERROR.format(name=name),
                    )
                )
        for child in ast.iter_child_nodes(node):
            visit(child, child_scope)

    visit(tree)
    return findings


def _path_receivers(tree: ast.AST) -> set[str]:
    """Infer path-like names; dynamic aliases remain a documented false negative."""
    names = {"Path", "PurePath", "PosixPath", "WindowsPath", "pathlib.Path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            names.update(item.asname or item.name for item in node.names)
        elif isinstance(node, ast.Import) and any(
            item.name == "pathlib" for item in node.names
        ):
            names.add("pathlib")
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation = _name(node.annotation)
            if annotation.endswith(("Path", "PurePath")):
                names.add(node.target.id)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _name(node.value.func) in names:
                names.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
    return names


def _is_path_receiver(node: ast.AST, path_names: set[str]) -> bool:
    if isinstance(node, ast.Call):
        return _name(node.func) in path_names
    name = _name(node)
    if not name:
        return False
    if name in path_names or name.startswith("pathlib."):
        return True
    return any(part.casefold() in ASYNC_PATH_NAME_HINTS for part in name.split("."))


def _async_blocking_call(node: ast.Call, path_names: set[str]) -> str | None:
    name = _name(node.func)
    if name in {"os." + function for function in ASYNC_OS_FUNCTIONS}:
        return name
    if name.startswith("os.path.") and name.rsplit(".", 1)[-1] in {
        "exists",
        "isdir",
        "isfile",
        "getsize",
    }:
        return name
    if name == "open":
        return name
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in ASYNC_PATH_METHODS
        and _is_path_receiver(node.func.value, path_names)
    ):
        return f"Path.{node.func.attr}"
    if name.startswith("shutil.") and name.rsplit(".", 1)[-1] in ASYNC_SHUTIL_FUNCTIONS:
        return name
    return None


def _async_blocking_findings(
    path: Path, tree: ast.AST, lines: list[str], added: set[int]
) -> list[Finding]:
    path_names = _path_receivers(tree)
    findings: list[Finding] = []
    off_loop = {
        "asyncio.to_thread",
        "to_thread",
        "asyncio.run_in_executor",
        "run_in_executor",
    }

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function: str | None = None
            self.async_depth = 0
            self.off_loop_depth = 0

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            previous = self.function
            self.function = node.name
            self.async_depth += 1
            for statement in node.body:
                self.visit(statement)
            self.async_depth -= 1
            self.function = previous

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self.async_depth == 0:
                self.generic_visit(node)
            # nested synchronous functions are deliberately excluded

        def visit_Lambda(self, node: ast.Lambda) -> None:
            if self.async_depth == 0:
                self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if self.async_depth and not self.off_loop_depth and node.lineno in added:
                call = _async_blocking_call(node, path_names)
                if call and not _has_marker(lines, node.lineno, {"async-blocking"}):
                    findings.append(
                        Finding(
                            path,
                            node.lineno,
                            node.col_offset + 1,
                            ASYNC_BLOCKING_CODE,
                            ASYNC_BLOCKING_ERROR.format(
                                call=call, function=self.function
                            ),
                        )
                    )
            if _name(node.func) in off_loop:
                self.off_loop_depth += 1
                for argument in node.args:
                    self.visit(argument)
                for keyword in node.keywords:
                    self.visit(keyword.value)
                self.off_loop_depth -= 1
            else:
                self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def lint_source(path: Path, source: str, added: set[int]) -> list[Finding]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            Finding(
                path,
                exc.lineno or 1,
                exc.offset or 1,
                "TQ000",
                "cannot parse changed test",
            )
        ]

    lines = source.splitlines()
    literal_names = _literal_names(tree)
    serialized_names = _serialized_names(tree)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    nodes = [node for node in ast.walk(tree) if hasattr(node, "lineno")]
    added_nodes = [node for node in nodes if _touches_added_line(node, added)]
    deadline_nodes = _deadline_nodes(nodes, added_nodes, added, parents)
    if path.parts[:2] == ("src", "zeta"):
        findings = []
        findings.extend(_async_blocking_findings(path, tree, lines, added))
    else:
        findings = _scan_added_nodes(
            path, added_nodes, lines, literal_names, serialized_names
        )
    findings.extend(_scan_deadline_nodes(path, deadline_nodes, lines, parents))
    findings.extend(_global_bootstrap_findings(path, tree, source, added))
    return sorted(set(findings), key=lambda item: (item.line, item.column, item.code))


def lint_diff(diff: str, root: Path | None = None) -> list[Finding]:
    root = Path.cwd() if root is None else root
    findings: list[Finding] = []
    for path, added in _added_lines(diff).items():
        if not added or not (
            _is_test_path(path)
            or path.name in {"conftest.py", "sitecustomize.py"}
            or path.parts[:2] == ("src", "zeta")
        ):
            continue
        source_path = root / path
        if source_path.is_file():
            findings.extend(
                lint_source(path, source_path.read_text(encoding="utf-8"), added)
            )
    return sorted(
        findings, key=lambda item: (str(item.path), item.line, item.column, item.code)
    )


def _function_at_line(tree: ast.AST, line: int) -> str:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not matches:
        return "<module>"
    return min(
        matches, key=lambda node: (node.end_lineno or node.lineno) - node.lineno
    ).name


def finding_ratchet_key(root: Path, finding: Finding) -> str:
    tree = ast.parse((root / finding.path).read_text(encoding="utf-8"))
    return f"{finding.path}::{_function_at_line(tree, finding.line)}"


def _scan_deadline_findings(
    root: Path,
) -> tuple[list[Finding], dict[Path, ast.AST]]:
    findings: list[Finding] = []
    trees: dict[Path, ast.AST] = {}
    for source_path in sorted((root / "tests").rglob("*.py")):
        path = source_path.relative_to(root)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        trees[path] = tree
        lines = source.splitlines()
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            findings.extend(_deadline_findings(path, node, lines, parents))
    return findings, trees


def _read_ratchet(root: Path) -> set[str]:
    ratchet_path = root / RATCHET_PATH
    if not ratchet_path.is_file():
        return set()
    return {
        line.strip()
        for line in ratchet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def lint_all(root: Path | None = None, *, use_ratchet: bool = True) -> list[Finding]:
    root = Path.cwd() if root is None else root
    findings, trees = _scan_deadline_findings(root)
    for source_path in sorted((root / "src" / "zeta").rglob("*.py")):
        path = source_path.relative_to(root)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        trees[path] = tree
        findings.extend(
            _async_blocking_findings(
                path,
                tree,
                source.splitlines(),
                set(range(1, len(source.splitlines()) + 1)),
            )
        )
    if not use_ratchet:
        return findings
    ratchet = _read_ratchet(root)
    return [
        finding
        for finding in findings
        if f"{finding.path}::{_function_at_line(trees[finding.path], finding.line)}"
        not in ratchet
    ]


def write_ratchet(root: Path | None = None) -> tuple[int, int]:
    root = Path.cwd() if root is None else root
    findings, _ = _scan_deadline_findings(root)
    keys = sorted({finding_ratchet_key(root, finding) for finding in findings})
    (root / RATCHET_PATH).write_text("".join(f"{key}\n" for key in keys))
    return len(findings), len(keys)


def select_base(event_name: str, base_ref: str = "") -> str:
    if event_name in {"pull_request", "workflow_dispatch"}:
        if event_name == "pull_request" and not base_ref:
            raise ValueError("pull_request requires a base ref")
        subprocess.run(
            ["git", "fetch", "--no-tags", "origin", base_ref or "main"], check=True
        )
        return subprocess.check_output(
            ["git", "merge-base", "HEAD", "FETCH_HEAD"], text=True
        ).strip()
    if event_name == "push":
        return subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip()
    raise ValueError(f"unsupported github event: {event_name}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["--base"] and len(arguments) in {2, 3}:
        print(select_base(arguments[1], arguments[2] if len(arguments) == 3 else ""))
        return 0
    if arguments == ["--write-ratchet"]:
        raw_count, key_count = write_ratchet()
        print(f"wrote {key_count} keys from {raw_count} findings to {RATCHET_PATH}")
        return 0
    if arguments == ["--all"]:
        findings = lint_all()
    elif len(arguments) == 1:
        diff = (
            sys.stdin.read()
            if arguments[0] == "-"
            else Path(arguments[0]).read_text(encoding="utf-8")
        )
        findings = lint_diff(diff)
    else:
        print(
            "usage: lint_test_quality.py <unified-diff-path|-> | --all | "
            "--write-ratchet | --base <pull_request|push|workflow_dispatch> "
            "[base-ref]",
            file=sys.stderr,
        )
        return 2
    for finding in findings:
        print(finding.render())
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
