"""Reject new low-signal test patterns found on added diff lines."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import Counter
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


def _file_diff_lines(diff: str, path: Path) -> list[str] | None:
    selected: list[str] | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if selected is not None:
                break
            continue
        if line.startswith("+++ "):
            selected = [] if _diff_header_path(line[4:]) == path else None
        elif selected is not None:
            selected.append(line)
    return selected


def _destination_cursor(match: re.Match[str]) -> int:
    start = int(match.group(1))
    count = int(match.group(2) or 1)
    return start if count == 0 else start - 1


def _base_source(diff: str, path: Path, source: str) -> str | None:
    file_diff = _file_diff_lines(diff, path)
    if file_diff is None:
        return None

    current = source.splitlines()
    previous: list[str] = []
    cursor = 0
    in_hunk = False
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in file_diff:
        if match := hunk.match(line):
            next_cursor = _destination_cursor(match)
            previous.extend(current[cursor:next_cursor])
            cursor = next_cursor
            in_hunk = True
        elif in_hunk and line.startswith(" "):
            previous.append(current[cursor])
            cursor += 1
        elif in_hunk and line.startswith("+"):
            cursor += 1
        elif in_hunk and line.startswith("-"):
            previous.append(line[1:])
    previous.extend(current[cursor:])
    ending = "\n" if previous and source.endswith("\n") else ""
    return "\n".join(previous) + ending


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


def _private_patch_identity(node: ast.Call) -> str:
    return ast.dump(node, include_attributes=False)


def _private_patch_counts(source: str | None) -> Counter[str]:
    if source is None:
        return Counter()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Counter()
    return Counter(
        _private_patch_identity(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _private_patch(node)
    )


def _new_private_patch_nodes(tree: ast.AST, base_source: str | None) -> set[int]:
    previous = _private_patch_counts(base_source)
    new_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _private_patch(node):
            continue
        identity = _private_patch_identity(node)
        if previous[identity]:
            previous[identity] -= 1
        else:
            new_nodes.add(id(node))
    return new_nodes


def _node_has_marker(lines: list[str], node: ast.AST, contracts: set[str]) -> bool:
    end = getattr(node, "end_lineno", node.lineno)
    return any(_has_marker(lines, line, contracts) for line in range(node.lineno, end + 1))


def _contract_findings(
    path: Path, node: ast.AST, lines: list[str], new_private_patches: set[int]
) -> list[Finding]:
    violates = (
        isinstance(node, ast.Call)
        and _private_patch(node)
        and id(node) in new_private_patches
    ) or _mock_contract_assertion(node)
    if not violates or _node_has_marker(
        lines, node, {"complexity", "durability", "seam"}
    ):
        return []
    return [Finding(path, node.lineno, node.col_offset + 1, "TQ003", CONTRACT_ERROR)]


def _enclosing_nodes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    ancestors: list[ast.AST] = []
    while node in parents:
        node = parents[node]
        ancestors.append(node)
    return ancestors


def _enclosing_function_touches_added_line(
    node: ast.AST, added: set[int], parents: dict[ast.AST, ast.AST]
) -> bool:
    return any(
        _touches_added_line(ancestor, added)
        for ancestor in _enclosing_nodes(node, parents)
        if isinstance(ancestor, (ast.AsyncFunctionDef, ast.FunctionDef))
    )


class _ImportResolver:
    def __init__(self, tree: ast.AST) -> None:
        self._scope: dict[ast.AST, ast.AST] = {}
        self._parent_scope: dict[ast.AST, ast.AST | None] = {}
        self._bindings: dict[ast.AST, dict[str, str]] = {}
        self._build(tree, tree, None)

    def _build(self, node: ast.AST, scope: ast.AST, parent: ast.AST | None) -> None:
        self._scope[node] = scope
        self._parent_scope.setdefault(scope, parent)
        if isinstance(node, ast.Import):
            bindings = self._bindings.setdefault(scope, {})
            for imported in node.names:
                local = imported.asname or imported.name.split('.')[0]
                bindings[local] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            bindings = self._bindings.setdefault(scope, {})
            for imported in node.names:
                if imported.name != '*':
                    bindings[imported.asname or imported.name] = (
                        f"{node.module}.{imported.name}"
                    )
        for child in ast.iter_child_nodes(node):
            child_scope = child if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ) else scope
            self._build(child, child_scope, scope if child_scope is not scope else parent)

    def resolve(self, node: ast.AST) -> str:
        value = _name(node)
        if not value:
            return value
        parts = value.split('.')
        scope = self._scope.get(node)
        while scope is not None:
            binding = self._bindings.get(scope, {}).get(parts[0])
            if binding is not None:
                return '.'.join([binding, *parts[1:]])
            scope = self._parent_scope.get(scope)
        return value


def _scoped_walk(node: ast.AST) -> list[ast.AST]:
    result: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def generic_visit(self, current: ast.AST) -> None:
            result.append(current)
            for child in ast.iter_child_nodes(current):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                self.visit(child)

    Visitor().visit(node)
    return result


def _finite_timeout_call(call: ast.Call, resolver: _ImportResolver) -> bool:
    return (
        resolver.resolve(call.func) in {"asyncio.timeout", "asyncio.timeout_at"}
        and bool(call.args)
        and not (isinstance(call.args[0], ast.Constant) and call.args[0].value is None)
    )


def _inside_timeout_context(
    node: ast.AST, parents: dict[ast.AST, ast.AST], resolver: _ImportResolver
) -> bool:
    return any(
        isinstance(ancestor, ast.AsyncWith)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _finite_timeout_call(item.context_expr, resolver)
            for item in ancestor.items
        )
        for ancestor in _enclosing_nodes(node, parents)
    )


def _inside_deadline_argument(
    node: ast.AST, parents: dict[ast.AST, ast.AST], resolver: _ImportResolver
) -> bool:
    child = node
    for ancestor in _enclosing_nodes(node, parents):
        if not isinstance(ancestor, ast.Call) or not any(
            child is argument for argument in ancestor.args
        ):
            if isinstance(ancestor, (ast.stmt, ast.comprehension)):
                return False
            child = ancestor
            continue
        function = resolver.resolve(ancestor.func)
        if function == "tests.support.owned.await_or_fail":
            return True
        if function in {"asyncio.wait_for"}:
            return (
                len(ancestor.args) > 1
                and not (
                    isinstance(ancestor.args[1], ast.Constant)
                    and ancestor.args[1].value is None
                )
                and child is ancestor.args[0]
            )
        return False
    return False


def _contains_async_sleep(node: ast.AST, resolver: _ImportResolver) -> bool:
    return any(
        isinstance(child, ast.Await)
        and isinstance(child.value, ast.Call)
        and resolver.resolve(child.value.func) in {"asyncio.sleep", "anyio.sleep"}
        for child in _scoped_walk(node)
    )


def _is_unbounded_iterator(node: ast.For, resolver: _ImportResolver) -> bool:
    if not isinstance(node.iter, ast.Call):
        return False
    name = resolver.resolve(node.iter.func)
    return name in {"itertools.count", "itertools.cycle"} or (
        name == "builtins.iter" and len(node.iter.args) == 2
    ) or (name == "iter" and len(node.iter.args) == 2)


def _is_unbounded_polling_loop(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    resolver: _ImportResolver,
    lines: list[str],
) -> bool:
    if not (
        isinstance(node, ast.While)
        or isinstance(node, ast.For) and _is_unbounded_iterator(node, resolver)
    ):
        return False
    if not _contains_async_sleep(node, resolver):
        return False
    if _has_marker(lines, node.lineno, set(CONTRACT_KINDS)):
        return False
    if _inside_timeout_context(node, parents, resolver) or _inside_deadline_argument(
        node, parents, resolver
    ):
        return False
    return not any(
        isinstance(child, ast.Await)
        and _inside_deadline_argument(child, parents, resolver)
        for child in _scoped_walk(node)
    )

def _is_unbounded_sync_wait(node: ast.AST, parents: dict[ast.AST, ast.AST], resolver: _ImportResolver) -> bool:
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
        and not _inside_deadline_argument(node, parents, resolver)
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
    resolver: _ImportResolver,
) -> list[Finding]:
    if _is_unbounded_polling_loop(node, parents, resolver, lines):
        return [
            Finding(
                path,
                node.lineno,
                node.col_offset + 1,
                AWAIT_DEADLINE_CODE,
                "polling loop without a deadline: wrap it in await_or_fail(...) / "
                "asyncio.timeout(...) or wait on an event",
            )
        ]
    if _is_unbounded_sync_wait(node, parents, resolver):
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
        or _inside_timeout_context(node, parents, resolver)
        or _inside_deadline_argument(node, parents, resolver)
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
    new_private_patches: set[int],
) -> list[Finding]:
    findings: list[Finding] = []
    for node in nodes:
        findings.extend(_timing_findings(path, node, lines))
        findings.extend(_digest_findings(path, node, literal_names, serialized_names))
        findings.extend(_contract_findings(path, node, lines, new_private_patches))
    return findings


def _scan_deadline_nodes(
    path: Path,
    nodes: list[ast.AST],
    lines: list[str],
    parents: dict[ast.AST, ast.AST],
    tree: ast.AST,
) -> list[Finding]:
    resolver = _ImportResolver(tree)
    return [
        finding
        for node in nodes
        for finding in _deadline_findings(
            path, node, lines, parents, resolver
        )
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


def lint_source(
    path: Path, source: str, added: set[int], base_source: str | None = None
) -> list[Finding]:
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
    new_private_patches = _new_private_patch_nodes(tree, base_source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    nodes = [node for node in ast.walk(tree) if hasattr(node, "lineno")]
    added_nodes = [node for node in nodes if _touches_added_line(node, added)]
    deadline_nodes = _deadline_nodes(nodes, added_nodes, added, parents)
    findings = _scan_added_nodes(
        path,
        added_nodes,
        lines,
        literal_names,
        serialized_names,
        new_private_patches,
    )
    findings.extend(_scan_deadline_nodes(path, deadline_nodes, lines, parents, tree))
    findings.extend(_global_bootstrap_findings(path, tree, source, added))
    return sorted(set(findings), key=lambda item: (item.line, item.column, item.code))


def lint_diff(diff: str, root: Path | None = None) -> list[Finding]:
    root = Path.cwd() if root is None else root
    findings: list[Finding] = []
    for path, added in _added_lines(diff).items():
        if not added or not (
            _is_test_path(path) or path.name in {"conftest.py", "sitecustomize.py"}
        ):
            continue
        source_path = root / path
        if source_path.is_file():
            findings.extend(
                lint_source(
                    path,
                    source := source_path.read_text(encoding="utf-8"),
                    added,
                    _base_source(diff, path, source),
                )
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
        resolver = _ImportResolver(tree)
        for node in ast.walk(tree):
            findings.extend(
                _deadline_findings(
                    path, node, lines, parents, resolver
                )
            )
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
