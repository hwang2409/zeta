"""Reject new low-signal test patterns found on added diff lines."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TIME_ERROR = (
    "wait on a readiness event or inject a clock; timeouts are hang guards only "
    "(use '# test-contract: latency' on this line or the line above only when real "
    "product latency is the contract)"
)
DIGEST_ERROR = "assert semantic fields, not a hash"
CONTRACT_ERROR = (
    "assert through <public interface> or mark the contract with "
    "'# test-contract: complexity' or '# test-contract: durability' on this line "
    "or the line above"
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


def _added_lines(diff: str) -> dict[Path, set[int]]:
    changed: dict[Path, set[int]] = {}
    path: Path | None = None
    new_line = 0
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = Path(line[6:])
            changed.setdefault(path, set())
            continue
        match = hunk.match(line)
        if match:
            new_line = int(match.group(1))
            continue
        if path is None or line.startswith(("diff --git ", "--- ")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changed[path].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            new_line += 1
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


def _has_marker(lines: list[str], line: int, contracts: set[str]) -> bool:
    for candidate in (line, line - 1):
        if candidate < 1 or candidate > len(lines):
            continue
        match = re.search(r"#\s*test-contract:\s*([a-z-]+)\b", lines[candidate - 1])
        if match and match.group(1) in contracts:
            return True
    return False


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
    findings: list[Finding] = []
    literal_names = _literal_names(tree)
    serialized_names = _serialized_names(tree)
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not _touches_added_line(node, added):
            continue
        column = node.col_offset + 1
        if isinstance(node, ast.Call) and _name(node.func) in {
            "time.sleep",
            "asyncio.sleep",
        }:
            delay = _literal_number(node.args[0]) if node.args else None
            if (
                delay is not None
                and delay > 0
                and not _has_marker(lines, node.lineno, {"latency"})
            ):
                findings.append(Finding(path, node.lineno, column, "TQ001", TIME_ERROR))
        if (
            isinstance(node, ast.Assert)
            and any(isinstance(child, ast.Compare) for child in ast.walk(node.test))
            and _contains_timing_reference(node.test)
            and not _has_marker(lines, node.lineno, {"latency"})
        ):
            findings.append(Finding(path, node.lineno, column, "TQ001", TIME_ERROR))
        if isinstance(node, ast.Assert):
            comparisons = [
                child for child in ast.walk(node.test) if isinstance(child, ast.Compare)
            ]
            if any(
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
            ):
                findings.append(
                    Finding(path, node.lineno, column, "TQ002", DIGEST_ERROR)
                )
        if (
            (isinstance(node, ast.Call) and _private_patch(node))
            or _mock_contract_assertion(node)
        ) and not _has_marker(lines, node.lineno, {"complexity", "durability"}):
            findings.append(Finding(path, node.lineno, column, "TQ003", CONTRACT_ERROR))
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


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: lint_test_quality.py <unified-diff-path|->", file=sys.stderr)
        return 2
    diff = (
        sys.stdin.read()
        if arguments[0] == "-"
        else Path(arguments[0]).read_text(encoding="utf-8")
    )
    findings = lint_diff(diff)
    for finding in findings:
        print(finding.render())
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
