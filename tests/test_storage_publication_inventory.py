from __future__ import annotations

import ast
from pathlib import Path

# Direct atomic calls are permitted only inside the shared lease callback, for
# append/immutable payloads, or for the documented test-only rollback path.
_ALLOWED_DIRECT_PUBLICATION = {
    ("session_retention.py", "write_retention_pin", "atomic_publish_file"),
    ("project_registry.py", "_publish", "atomic_publish_file"),
    ("project_inbox.py", "_persist", "atomic_publish_file"),
    ("project_inbox.py", "_publish_record", "atomic_publish_file"),
    (
        "memory/entry_history.py",
        "_rollback_memory_migration_for_test",
        "atomic_publish_file",
    ),
    ("memory/version_store.py", "_publish_payloads", "atomic_publish_file"),
    ("memory/version_store.py", "_publish_legacy_json", "atomic_publish_file"),
    ("remote_sync/project_publish.py", "publish_version", "atomic_publish_file"),
    ("remote_sync/project_publish.py", "_publish_payloads", "atomic_publish_file"),
    ("remote_sync/project_publish.py", "_publish_side_files", "atomic_publish_file"),
    ("runtime/loop/persistence.py", "_persist", "write_json_file"),
}


def test_all_direct_json_publication_has_an_owned_adapter_or_exclusion() -> None:
    root = Path(__file__).parents[1] / "src" / "zeta"
    found: set[tuple[str, str, str]] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())

        class Visitor(ast.NodeVisitor):
            def __init__(self, source: Path) -> None:
                self.source = source
                self.functions: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    name = ""
                if name in {"atomic_publish_file", "write_json_file"}:
                    found.add(
                        (
                            self.source.relative_to(root).as_posix(),
                            self.functions[-1] if self.functions else "<module>",
                            name,
                        )
                    )
                self.generic_visit(node)

        Visitor(path).visit(tree)

    assert found == _ALLOWED_DIRECT_PUBLICATION
