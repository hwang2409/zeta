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


def _keyword_is_true(node: ast.Call, name: str) -> bool:
    return any(
        keyword.arg == name
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def _timing_findings(path: Path, node: ast.AST, lines: list[str]) -> list[Finding]:
    if isinstance(node, ast.Call):
        allow_short = _name(node.func) == "await_or_fail" and _keyword_is_true(
            node, "allow_short"
        )
        has_latency_marker = _has_marker(lines, node.lineno, {"latency"})
        if allow_short and not has_latency_marker:
            return [
                Finding(
                    path, node.lineno, node.col_offset + 1, "TQ001", SHORT_TIMEOUT_ERROR
                )
            ]
        timeout = _wait_timeout(node)
        short_override = allow_short and has_latency_marker
        if (
            timeout is not None
            and 0 < timeout < MIN_HANG_GUARD_SECONDS
            and not short_override
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


def _inside_test_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    return any(
        isinstance(ancestor, (ast.AsyncFunctionDef, ast.FunctionDef))
        and ancestor.name.startswith("test_")
        for ancestor in _enclosing_nodes(node, parents)
    )


def _is_unbounded_sync_wait(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    function = node.func
    if not isinstance(function, ast.Attribute):
        return False
    if function.attr == "join":
        return True
    return (
        function.attr == "wait"
        and isinstance(parents.get(node), ast.Expr)
        and _inside_test_function(node, parents)
    )


def _is_blocking_to_thread(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or _name(node.func) not in {
        "asyncio.to_thread",
        "to_thread",
    }:
        return False
    if not node.args or not isinstance(node.args[0], ast.Attribute):
        return False
    target = node.args[0]
    if target.attr in {"recv", "recv_into", "read", "readline", "accept"}:
        return True
    return target.attr == "get" and "queue" in _name(target.value).lower()


def _bounded_asyncio_wait(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _name(node.func) == "asyncio.wait"
        and any(keyword.arg == "timeout" for keyword in node.keywords)
    )


def _assigned_names(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return set()
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {
        part.id
        for target in targets
        for part in ast.walk(target)
        if isinstance(part, ast.Name)
    }


def _earlier_block_statements(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.stmt]:
    child = node
    for parent in _enclosing_nodes(node, parents):
        if isinstance(child, ast.stmt):
            for _, value in ast.iter_fields(parent):
                if isinstance(value, list) and child in value:
                    return value[: value.index(child)]
        child = parent
    return []


def _nearest_assignment(name: str, statements: list[ast.stmt]) -> ast.AST | None:
    return next(
        (
            statement
            for statement in reversed(statements)
            if name in _assigned_names(statement)
        ),
        None,
    )


def _is_bounded_wait_assignment(node: ast.AST | None) -> bool:
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    return isinstance(value, ast.Await) and _bounded_asyncio_wait(value.value)


def _support_gather_is_already_done(
    path: Path, node: ast.Await, parents: dict[ast.AST, ast.AST]
) -> bool:
    value = node.value
    if path.parts[:2] != ("tests", "support") or not (
        isinstance(value, ast.Call) and _name(value.func) == "asyncio.gather"
    ):
        return False
    gathered = {
        argument.value.id
        for argument in value.args
        if isinstance(argument, ast.Starred) and isinstance(argument.value, ast.Name)
    }
    earlier = _earlier_block_statements(node, parents)
    return bool(gathered) and all(
        _is_bounded_wait_assignment(_nearest_assignment(name, earlier))
        for name in gathered
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
    value = node.value
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
        or _support_gather_is_already_done(path, node, parents)
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
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not _touches_added_line(node, added):
            continue
        findings.extend(_timing_findings(path, node, lines))
        findings.extend(_digest_findings(path, node, literal_names, serialized_names))
        findings.extend(_contract_findings(path, node, lines))
        findings.extend(_deadline_findings(path, node, lines, parents))
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
            "--base <pull_request|push> [base-ref]",
            file=sys.stderr,
        )
        return 2
    for finding in findings:
        print(finding.render())
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
