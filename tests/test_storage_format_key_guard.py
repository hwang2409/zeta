"""Guard direct key-set checks that can bypass the stored-format registry.

This heuristic detects common set comparisons, subtraction, and ``issubset`` calls.
It does not detect loops over keys, length checks, strict constructor arguments,
Pydantic ``extra=forbid``, or other spellings of unknown-key validation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

_ALLOWED_REASONS = {
    "tool arguments",
    "settings",
    "rpc params",
    "model output",
    "in-memory value",
}

# Each row is (module, function, reason). Persisted payloads must not be allowlisted.
_KEY_SET_CHECK_ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    ("zeta.accounts.registry", "_registry_reference_index", "in-memory value"),
    ("zeta.config.settings", "_strip_unsafe_project_keys", "settings"),
    ("zeta.config.settings", "_merge_tool_policy", "settings"),
    ("zeta.config.settings", "_validate", "settings"),
    ("zeta.config.settings", "_validated_inbox", "settings"),
    ("zeta.config.settings", "_validated_approval", "settings"),
    ("zeta.core.session_files", "copy_session_tree", "in-memory value"),
    ("zeta.core.transcript_recovery.__init__", "reconcile_pending", "in-memory value"),
    ("zeta.memory.entry_history", "set_memory_profile", "in-memory value"),
    ("zeta.memory.entry_reconciler", "_exact", "model output"),
    ("zeta.memory.entry_reconciler", "_parse", "model output"),
    ("zeta.memory.entry_sync", "merge_entry_states", "in-memory value"),
    ("zeta.memory.migration", "migrate_format_one", "in-memory value"),
    ("zeta.memory.reconciler", "parse_proposal", "model output"),
    ("zeta.memory_migration_plan", "build_migration_plan", "in-memory value"),
    ("zeta.project_memory_history", "_validate_updates", "in-memory value"),
    ("zeta.project_memory_history", "_import_memory_locked", "in-memory value"),
    ("zeta.project_registry", "update_memory", "in-memory value"),
    ("zeta.protocol.tool_batches", "without_incomplete_tool_batches", "in-memory value"),
    ("zeta.server.connections", "capture_read", "rpc params"),
    ("zeta.server.connections", "read_child_run_history", "rpc params"),
    ("zeta.server.control", "revoke", "rpc params"),
    ("zeta.server.handlers.history", "_validate_history_params", "rpc params"),
    ("zeta.server.handlers.projects", "_only", "rpc params"),
    ("zeta.server.host_commands.__init__", "parse", "rpc params"),
    ("zeta.tools._validation", "_validate_schema", "tool arguments"),
    ("zeta.tools._validation", "_schema_equal", "tool arguments"),
    ("zeta.tools._validation", "_validate_schema_definition", "tool arguments"),
    ("zeta.tools.agent_control", "_status", "in-memory value"),
    ("zeta.tools.inbox.__init__", "_validate_action", "tool arguments"),
    ("zeta.tools.todo.__init__", "_todo", "tool arguments"),
)


@dataclass(frozen=True)
class _KeySetCheck:
    module: str
    function: str
    line: int


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts[1:])


def _is_set_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
    )


def _is_keys_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "keys"
    )


def _is_key_set_check(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare) and _is_set_call(node.left):
        return any(
            isinstance(operator, (ast.Eq, ast.NotEq, ast.LtE, ast.GtE))
            for operator in node.ops
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _is_set_call(node.left) or _is_keys_call(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "issubset" and (
            _is_set_call(node.func.value) or isinstance(node.func.value, ast.Set)
        )
    return False


def _function_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield one function's nodes without borrowing a nested function's checks."""

    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _uses_registry(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in _function_nodes(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reject_unknown_keys"
        ) or (
            isinstance(node, ast.Name)
            and node.id == "ACCEPTED_KEY_SETS"
            and isinstance(node.ctx, ast.Load)
        ):
            return True
    return False


def _source_checks() -> tuple[set[tuple[str, str]], list[_KeySetCheck]]:
    functions: set[tuple[str, str]] = set()
    checks: list[_KeySetCheck] = []
    for path in sorted(Path("src/zeta").rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        class Functions(ast.NodeVisitor):
            def __init__(self, module: str) -> None:
                self.module = module
                self.parents: list[str] = []

            def _visit_function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                name = ".".join((*self.parents, node.name))
                functions.add((self.module, name))
                if not _uses_registry(node):
                    checks.extend(
                        _KeySetCheck(self.module, name, candidate.lineno)
                        for candidate in _function_nodes(node)
                        if _is_key_set_check(candidate)
                    )
                self.parents.append(node.name)
                self.generic_visit(node)
                self.parents.pop()

            visit_FunctionDef = _visit_function
            visit_AsyncFunctionDef = _visit_function

        Functions(module).visit(tree)
    return functions, checks


def test_key_set_checks_use_stored_format_registry_or_explicit_allowlist() -> None:
    functions, checks = _source_checks()
    allowlist = {
        (module, function): reason
        for module, function, reason in _KEY_SET_CHECK_ALLOWLIST
    }
    assert len(allowlist) == len(_KEY_SET_CHECK_ALLOWLIST), "duplicate allowlist row"
    assert set(allowlist.values()) <= _ALLOWED_REASONS

    matching = {(check.module, check.function) for check in checks}
    stale = sorted(set(allowlist) - matching)
    assert not stale, f"stale key-set-check allowlist rows: {stale!r}"
    assert set(allowlist) <= functions

    unclassified = [
        check for check in checks if (check.module, check.function) not in allowlist
    ]
    if unclassified:
        pytest.fail(
            "\n".join(
                f"{check.module}:{check.function}:{check.line} checks a key set without "
                "reject_unknown_keys: route it through "
                "reject_unknown_keys(payload, '<shape>') or add an allowlist row "
                "(module, function, reason) with a non-persisted reason"
                for check in unclassified
            )
        )
