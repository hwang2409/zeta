"""Reject cancellation handlers that re-await one shielded operation once."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

RATCHET = {
    "src/zeta/core/hooks.py:HookManager._run:creation_task",
    "src/zeta/runtime/loop/project_inbox.py:ProjectInboxNotificationMixin._poll_project_inbox_once:worker",
    "src/zeta/runtime/loop/project_inbox.py:ProjectInboxNotificationMixin._record_sent_statuses:worker",
    "src/zeta/server/approval_settlement/approval_grants.py:finish_grant:operation",
    "src/zeta/tools/_shared/process.py:BackgroundTaskRegistry._drain_stdin:drain",
    "src/zeta/tools/_shared/tool_spawn.py:_cleanup_unwatched_transport:reap",
    "src/zeta/tools/_shared/tool_spawn.py:create_tool_subprocess_shell:cleanup",
}
HISTORICAL_PROOF = "ad1cfd43:accounts/session.py:_run_locked_apply:worker"


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _shielded_names(nodes: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or _name(child.func) != "shield":
                continue
            if child.args:
                name = _name(child.args[0])
                if name:
                    names.add(name)
    return names


def _qualified(stack: list[str], name: str) -> str:
    return ".".join([*stack, name])


def findings(path: Path, source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=str(path))
    found: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Try(self, node: ast.Try) -> None:
            before = _shielded_names(node.body)
            for handler in node.handlers:
                if _name(handler.type) not in {"CancelledError", "BaseException"}:
                    continue
                for name in sorted(before & _shielded_names(handler.body)):
                    key = f"{path.as_posix()}:{_qualified(self.stack, '<try>')}:{name}"
                    key = key.replace(".<try>:", ":")
                    found.append((handler.lineno, key))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def main() -> int:
    root = Path.cwd()
    observed: set[str] = set()
    errors: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root)
        for line, key in findings(relative, path.read_text(encoding="utf-8")):
            observed.add(key)
            if key not in RATCHET:
                errors.append(
                    f"{relative}:{line}: cancellation re-await of {key.rsplit(':', 1)[-1]}; "
                    "use run_owned_thread_call for thread work or one cancel-safe owner"
                )
    stale = RATCHET - observed
    errors.extend(f"stale cancellation-ownership ratchet: {key}" for key in sorted(stale))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
