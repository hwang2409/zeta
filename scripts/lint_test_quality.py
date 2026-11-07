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
CONTRACT_KINDS = {"complexity", "durability", "latency", "seam"}
MIN_HANG_GUARD_SECONDS = 10
AWAIT_DEADLINE_ERROR = (
    "await without a deadline: use `await await_or_fail(x, what=...)` "
    "(tests/support/owned.py), or asyncio.wait_for(x, 10)"
)
SYNC_DEADLINE_ERROR = "blocking wait without a deadline: pass timeout=10"
AWAIT_DEADLINE_CODE = "TQ004"
RATCHET_PATH = Path("tests/test_quality_ratchet.txt")
CLEANUP_RATCHET_PATH = Path("tests/test_cleanup_owner_ratchet.txt")
CLEANUP_CODE = "TQ006"
CLEANUP_OWNERS = {
    "app": "`async with owned_app(TUIApp(...)):`",
    "store": (
        "`async with owned_resources() as owner: "
        "store = owner.store(ConversationStore(...))`"
    ),
    "async resource": (
        "`async with owned_resources() as owner: "
        "resource = owner.async_resource(..., what=...)`"
    ),
    "task": (
        "`async with owned_resources() as owner: "
        "task = owner.task(asyncio.create_task(...))`"
    ),
    "process": "`async with owned_child_process(...) as process:`",
}


@dataclass(frozen=True)
class HelperResource:
    kind: str
    open_call: ast.Call
    hazardous_after_open: bool


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
        match = re.search(
            r"#\s*test-contract:\s*([a-z-]+)(?:\s+([^#\s].*?))?\s*$",
            lines[candidate - 1],
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


def _cleanup_kind(node: ast.Call) -> str | None:
    function = _name(node.func)
    terminal = function.rsplit(".", 1)[-1]
    if terminal == "TUIApp":
        return "app"
    if terminal == "ConversationStore":
        return "store"
    if (
        terminal in {"AgentLoop", "ToolRegistry"}
        or function.endswith("ServeContract.start")
        or function
        in {
            "asyncio.start_server",
            "asyncio.start_unix_server",
            "start_server",
            "start_unix_server",
        }
    ):
        return "async resource"
    if function in {
        "asyncio.create_task",
        "create_task",
        "asyncio.ensure_future",
        "ensure_future",
    }:
        return "task"
    if terminal == "Process" or function in {
        "subprocess.Popen",
        "Popen",
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "create_subprocess_exec",
        "create_subprocess_shell",
    }:
        return "process"
    return None


def _owned_by_function(
    node: ast.AST,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    return (
        next(
            (
                ancestor
                for ancestor in _enclosing_nodes(node, parents)
                if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        is function
    )


def _local_resource_helpers(tree: ast.AST) -> dict[str, list[HelperResource]]:
    helpers: dict[str, list[HelperResource]] = {}
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        members = [
            node
            for node in ast.walk(function)
            if _owned_by_function(node, function, parents)
        ]
        returns = [
            node
            for node in members
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
        ]
        for call in members:
            if not isinstance(call, ast.Call) or (kind := _cleanup_kind(call)) is None:
                continue
            bound_name = _bound_name(call, parents)
            matching_returns = [
                node
                for node in returns
                if node.lineno > call.lineno
                and bound_name is not None
                and _contains_name(node, bound_name)
            ]
            for returned in matching_returns:
                hazardous = any(
                    isinstance(node, (ast.Await, ast.Raise))
                    and call.end_lineno < node.lineno < returned.lineno
                    for node in members
                )
                helpers.setdefault(function.name, []).append(
                    HelperResource(kind, call, hazardous)
                )
                break
    return helpers


def _helper_resources(
    node: ast.Call, helpers: dict[str, list[HelperResource]]
) -> list[HelperResource]:
    return helpers.get(_name(node.func).rsplit(".", 1)[-1], [])


def _owner_call_matches(node: ast.Call, kind: str) -> bool:
    function = _name(node.func)
    terminal = function.rsplit(".", 1)[-1]
    expected = {
        "app": {"owned_app"},
        "store": {"store"},
        "async resource": {"async_resource"},
        "task": {"task"},
        "process": {"owned_child_process"},
    }[kind]
    return terminal in expected


def _inside_owner_call(
    node: ast.Call, kind: str, parents: dict[ast.AST, ast.AST]
) -> bool:
    return any(
        isinstance(ancestor, ast.Call) and _owner_call_matches(ancestor, kind)
        for ancestor in _enclosing_nodes(node, parents)
        if not isinstance(ancestor, ast.stmt)
    )


def _bound_name(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str | None:
    for ancestor in _enclosing_nodes(node, parents):
        if isinstance(ancestor, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = (
                ancestor.targets
                if isinstance(ancestor, ast.Assign)
                else [ancestor.target]
            )
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            return names[0] if len(names) == 1 else None
        if isinstance(ancestor, ast.stmt):
            return None
    return None


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def _is_cleanup_call(node: ast.Call, kind: str, bound_name: str) -> bool:
    function = _name(node.func)
    if kind == "task":
        return function.endswith("_cancel_and_drain") and any(
            _contains_name(argument, bound_name) for argument in node.args
        )
    return function == f"{bound_name}.close"


def _statement_has_cleanup(statement: ast.stmt, kind: str, bound_name: str) -> bool:
    return any(
        isinstance(child, ast.Call) and _is_cleanup_call(child, kind, bound_name)
        for child in ast.walk(statement)
    )


def _cleanup_postdominates(
    statements: list[ast.stmt], kind: str, bound_name: str
) -> bool:
    if not statements:
        return False
    first = statements[0]
    if isinstance(first, ast.Try) and _cleanup_postdominates(
        first.finalbody, kind, bound_name
    ):
        return True
    return _statement_has_cleanup(first, kind, bound_name)


def _is_fixture(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _name(decorator.func if isinstance(decorator, ast.Call) else decorator)
        in {"fixture", "pytest.fixture"}
        for decorator in function.decorator_list
    )


def _statement_block(
    function: ast.FunctionDef | ast.AsyncFunctionDef, statement: ast.stmt
) -> tuple[list[ast.stmt], int] | None:
    for node in ast.walk(function):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list) or statement not in value:
                continue
            if all(isinstance(item, ast.stmt) for item in value):
                return value, value.index(statement)
    return None


def _reassigns_name(statement: ast.stmt, bound_name: str) -> bool:
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == bound_name
        for node in ast.walk(statement)
    )


def _cannot_raise_or_await(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        try:
            ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            return False
        return True
    return False


def _safe_pre_try_cleanup(
    node: ast.Call,
    kind: str,
    bound_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    open_statement = next(
        (
            ancestor
            for ancestor in _enclosing_nodes(node, parents)
            if isinstance(ancestor, ast.stmt)
        ),
        None,
    )
    if (
        open_statement is None
        or (block := _statement_block(function, open_statement)) is None
    ):
        return False
    statements, index = block
    for candidate_index in range(index + 1, len(statements)):
        candidate = statements[candidate_index]
        between = statements[index + 1 : candidate_index]
        if isinstance(candidate, ast.Try) and _cleanup_postdominates(
            candidate.finalbody, kind, bound_name
        ):
            return all(
                not _reassigns_name(statement, bound_name)
                and _cannot_raise_or_await(statement)
                for statement in between
            )
        if _reassigns_name(candidate, bound_name) or not _cannot_raise_or_await(
            candidate
        ):
            return False
    return False


def _manual_cleanup_is_safe(
    node: ast.Call,
    kind: str,
    bound_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    enclosing_tries = [
        ancestor
        for ancestor in _enclosing_nodes(node, parents)
        if isinstance(ancestor, ast.Try)
    ]
    if any(
        _cleanup_postdominates(candidate.finalbody, kind, bound_name)
        for candidate in enclosing_tries
    ) or _safe_pre_try_cleanup(node, kind, bound_name, function, parents):
        return True
    if not _is_fixture(function):
        return False
    return any(
        candidate.lineno > node.lineno
        and any(
            isinstance(child, (ast.Yield, ast.YieldFrom))
            for child in ast.walk(candidate)
        )
        and _cleanup_postdominates(candidate.finalbody, kind, bound_name)
        for candidate in ast.walk(function)
        if isinstance(candidate, ast.Try)
    )


def _cleanup_finding(
    path: Path,
    node: ast.Call,
    kind: str,
    parents: dict[ast.AST, ast.AST],
    *,
    helper_hazard: bool,
) -> Finding | None:
    if not helper_hazard and _inside_owner_call(node, kind, parents):
        return None
    function = next(
        (
            ancestor
            for ancestor in _enclosing_nodes(node, parents)
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    bound_name = _bound_name(node, parents)
    if (
        not helper_hazard
        and function is not None
        and bound_name is not None
        and _manual_cleanup_is_safe(node, kind, bound_name, function, parents)
    ):
        return None
    owner = CLEANUP_OWNERS[kind]
    return Finding(
        path,
        node.lineno,
        node.col_offset + 1,
        CLEANUP_CODE,
        f"closeable {kind} is not failure-safe; open it with {owner} "
        "so timeout, cancellation, and assertion paths always clean up",
    )


def _cleanup_findings(
    path: Path,
    nodes: list[ast.AST],
    parents: dict[ast.AST, ast.AST],
    helpers: dict[str, list[HelperResource]] | None = None,
) -> list[Finding]:
    if path == Path("tests/support/owned.py"):
        return []
    findings: list[Finding] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        explicit_kind = _cleanup_kind(node)
        resources = _helper_resources(node, helpers or {})
        kinds = ({explicit_kind} if explicit_kind is not None else set()) | {
            resource.kind for resource in resources
        }
        for kind in kinds:
            finding = _cleanup_finding(
                path,
                node,
                kind,
                parents,
                helper_hazard=any(
                    resource.kind == kind and resource.hazardous_after_open
                    for resource in resources
                ),
            )
            if finding is not None:
                findings.append(finding)
    return findings


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
    changed_function_nodes = [
        node
        for node in nodes
        if node in added_nodes
        or _enclosing_function_touches_added_line(node, added, parents)
    ]
    findings = _scan_added_nodes(
        path, added_nodes, lines, literal_names, serialized_names
    )
    findings.extend(_scan_deadline_nodes(path, changed_function_nodes, lines, parents))
    helpers = _local_resource_helpers(tree)
    called_helpers = {
        _name(node.func).rsplit(".", 1)[-1]
        for node in changed_function_nodes
        if isinstance(node, ast.Call)
    }
    helper_open_nodes = [
        resource.open_call
        for name in called_helpers
        for resource in helpers.get(name, [])
        if resource.hazardous_after_open
    ]
    cleanup_nodes = list(dict.fromkeys([*changed_function_nodes, *helper_open_nodes]))
    findings.extend(_cleanup_findings(path, cleanup_nodes, parents, helpers))
    return sorted(set(findings), key=lambda item: (item.line, item.column, item.code))


def lint_diff(diff: str, root: Path | None = None) -> list[Finding]:
    root = Path.cwd() if root is None else root
    findings: list[Finding] = []
    for path, added in _added_lines(diff).items():
        if not added or not _is_test_path(path):
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


def _scan_cleanup_findings(root: Path) -> tuple[list[Finding], dict[Path, ast.AST]]:
    findings: list[Finding] = []
    trees: dict[Path, ast.AST] = {}
    for source_path in sorted((root / "tests").rglob("*.py")):
        path = source_path.relative_to(root)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        trees[path] = tree
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        findings.extend(
            _cleanup_findings(
                path,
                list(ast.walk(tree)),
                parents,
                _local_resource_helpers(tree),
            )
        )
    return findings, trees


def cleanup_ratchet_key(finding: Finding, trees: dict[Path, ast.AST]) -> str:
    tree = trees[finding.path]
    kind = finding.message.split("closeable ", 1)[1].split(" is ", 1)[0]
    return f"{finding.path}::{_function_at_line(tree, finding.line)}::{kind}"


def _read_ratchet(root: Path, path: Path = RATCHET_PATH) -> set[str]:
    ratchet_path = root / path
    if not ratchet_path.is_file():
        return set()
    return {
        line.strip()
        for line in ratchet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def lint_all(root: Path | None = None, *, use_ratchet: bool = True) -> list[Finding]:
    root = Path.cwd() if root is None else root
    deadline_findings, deadline_trees = _scan_deadline_findings(root)
    cleanup_findings, cleanup_trees = _scan_cleanup_findings(root)
    if not use_ratchet:
        return sorted(
            [*deadline_findings, *cleanup_findings],
            key=lambda finding: (
                str(finding.path),
                finding.line,
                finding.column,
                finding.code,
            ),
        )
    deadline_ratchet = _read_ratchet(root)
    cleanup_ratchet = _read_ratchet(root, CLEANUP_RATCHET_PATH)
    current_cleanup = {
        cleanup_ratchet_key(finding, cleanup_trees) for finding in cleanup_findings
    }
    findings = [
        finding
        for finding in deadline_findings
        if f"{finding.path}::{_function_at_line(deadline_trees[finding.path], finding.line)}"
        not in deadline_ratchet
    ]
    findings.extend(
        finding
        for finding in cleanup_findings
        if cleanup_ratchet_key(finding, cleanup_trees) not in cleanup_ratchet
    )
    findings.extend(
        Finding(
            CLEANUP_RATCHET_PATH,
            line,
            1,
            CLEANUP_CODE,
            f"{key} no longer exists; remove it from {CLEANUP_RATCHET_PATH}",
        )
        for line, key in enumerate(sorted(cleanup_ratchet - current_cleanup), start=1)
    )
    return sorted(
        findings,
        key=lambda finding: (
            str(finding.path),
            finding.line,
            finding.column,
            finding.code,
        ),
    )


def write_ratchet(root: Path | None = None) -> tuple[int, int]:
    root = Path.cwd() if root is None else root
    findings, _ = _scan_deadline_findings(root)
    keys = sorted({finding_ratchet_key(root, finding) for finding in findings})
    (root / RATCHET_PATH).write_text("".join(f"{key}\n" for key in keys))
    return len(findings), len(keys)


def select_base(event_name: str, base_ref: str = "") -> str:
    if event_name == "pull_request":
        if not base_ref:
            raise ValueError("pull_request requires a base ref")
        subprocess.run(["git", "fetch", "--no-tags", "origin", base_ref], check=True)
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
            "--write-ratchet | --base <pull_request|push> [base-ref]",
            file=sys.stderr,
        )
        return 2
    for finding in findings:
        print(finding.render())
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
