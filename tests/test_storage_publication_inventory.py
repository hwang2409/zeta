from __future__ import annotations

import ast
from pathlib import Path

# Direct durability callbacks are permitted only behind a gated writer, for
# immutable payloads, or for an intrinsic append format.
_DURABILITY_CALLBACKS = {
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

# Raw replace/rename sites are classified so a newly named direct writer cannot
# evade the inventory. These are durability primitives, gated state moves, or
# authoritative paths owned by another accepted slice.
_RAW_PUBLICATION = {
    ("storage_format.py", "_publish_manifest", "os.replace"),
    ("project_registry.py", "_rename_without_replacement", "os.rename"),
    ("project_inbox.py", "claim", "os.rename"),
    ("project_inbox.py", "done", "os.rename"),
    ("project_inbox.py", "_recover_stale", "os.rename"),
    ("project_memory_history.py", "_publish_mirror_file", "os.replace"),
    ("tools/_spill.py", "write_group", "os.replace"),
    ("core/session.py", "create", "os.replace"),
    ("core/session_links.py", "_quarantine_entry", "os.rename"),
    ("core/session_files.py", "write_session_file", "os.replace"),
    ("memory/reconciliation_state.py", "_write_receipt_archive", "os.replace"),
    ("memory/reconciliation_state.py", "_publish_bytes", "os.replace"),
    ("memory/auto.py", "_append_diagnostic_record", "os.replace"),
    ("server/transport.py", "write_connection_file", "os.replace"),
    ("mcp/config.py", "write_mcp_config", "os.replace"),
    ("mcp/management.py", "_atomic_json", "os.replace"),
    ("mcp/oauth_store.py", "_publish_token", "os.replace"),
    ("remote_sync/memory.py", "_atomic_write_file", "os.replace"),
    ("remote_sync/memory.py", "_atomic_copy_file", "os.replace"),
    ("remote_sync/memory.py", "_machine_id", "os.replace"),
    ("remote_sync/__init__.py", "_rewrite_cwd", "os.replace"),
    ("remote_sync/__init__.py", "_atomic_copy_file", "os.replace"),
    ("remote_sync/__init__.py", "_write_json", "os.replace"),
    ("remote_sync/project_publish.py", "publish_local_project", "os.rename"),
    ("remote_sync/project_publish.py", "atomic_publish_file", "os.replace"),
    ("accounts/publication.py", "atomic_publish", "os.replace"),
    ("diagnostics/tui_stalls.py", "_append_bounded_record", "os.replace"),
    ("transcript_search/index.py", "_publish", "os.replace"),
}

_VARIANT_APPEND_SEAMS = {
    ("project_registry.py", "record_session", "publish_append"),
    ("project_registry.py", "record_sessions", "publish_append"),
}

_APPEND_PUBLICATION = {
    ("project_registry.py", "_append_reference_blob", "append"),
    ("memory/reconciliation_state.py", "_append_receipt_archive", "append"),
    ("diagnostics/tui_stalls.py", "_append_bounded_record", "append"),
    ("core/store/_log.py", "_write_bytes", "append"),
}

# p07 owns remote project publication. It is authoritative, not a derived or
# ephemeral bypass, and remains visible here until that lane gates it.
_REMOTE_PROJECT_PUBLICATION_P07 = {
    item
    for item in _DURABILITY_CALLBACKS | _RAW_PUBLICATION
    if item[0] == "remote_sync/project_publish.py"
}


def test_all_direct_publication_paths_are_classified() -> None:
    root = Path(__file__).parents[1] / "src" / "zeta"
    found_callbacks: set[tuple[str, str, str]] = set()
    found_raw: set[tuple[str, str, str]] = set()
    found_append: set[tuple[str, str, str]] = set()
    found_variant_append: set[tuple[str, str, str]] = set()
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
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
                relative = self.source.relative_to(root).as_posix()
                function = self.functions[-1] if self.functions else "<module>"
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    name = ""
                if name in {"atomic_publish_file", "write_json_file"}:
                    found_callbacks.add((relative, function, name))
                if name == "publish_append":
                    found_variant_append.add((relative, function, name))
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr in {"replace", "rename"}
                ):
                    found_raw.add((relative, function, f"os.{node.func.attr}"))
                if any(
                    isinstance(item, ast.Attribute) and item.attr == "O_APPEND"
                    for item in ast.walk(node)
                ):
                    found_append.add((relative, function, "append"))
                self.generic_visit(node)

        Visitor(path).visit(tree)

    assert found_callbacks == _DURABILITY_CALLBACKS
    assert found_raw == _RAW_PUBLICATION
    assert found_append == _APPEND_PUBLICATION
    assert found_variant_append == _VARIANT_APPEND_SEAMS
    assert _REMOTE_PROJECT_PUBLICATION_P07
