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
    ("mcp/oauth_store.py", "_publish_locked", "os.replace"),
    ("mcp/stderr_log.py", "_rotate", "os.replace"),
    ("tools/_atomic_replace.py", "atomic_replace", "os.replace"),
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

_RAW_DELETION = {
    ("accounts/logout.py", "_remove_directory_files", "unlink"),
    ("accounts/publication.py", "atomic_publish", "unlink"),
    ("accounts/store.py", "_delete_publication_pending", "unlink"),
    ("accounts/store.py", "_unlink_legacy_file", "unlink"),
    ("core/checkpoints/workspace.py", "_current_tree_sha", "unlink"),
    ("core/checkpoints/workspace.py", "_materialise_tree", "unlink"),
    ("core/checkpoints/workspace.py", "_prune_stale_entries", "unlink"),
    ("core/checkpoints/workspace.py", "_reserve_shadow_index_path", "unlink"),
    ("core/checkpoints/workspace.py", "_write_current_shadow_commit", "unlink"),
    ("core/checkpoints/workspace.py", "reset_corruption", "unlink"),
    ("core/project_context.py", "_load_home_identity", "unlink"),
    ("core/session.py", "_reconcile_project_link", "unlink"),
    ("core/session_files.py", "copy_session_tree", "unlink"),
    ("core/session_files.py", "unlink_session_file", "unlink"),
    ("core/session_files.py", "write_session_file", "unlink"),
    ("core/session_links.py", "_remove_entry", "unlink"),
    ("core/session_links.py", "remove_pending_child_link", "unlink"),
    ("diagnostics/tui_stalls.py", "_append_bounded_record", "unlink"),
    ("mcp/config.py", "write_mcp_config", "unlink"),
    ("mcp/management.py", "_atomic_json", "unlink"),
    ("mcp/oauth_store.py", "_publish_locked", "unlink"),
    ("mcp/stderr_log.py", "_rotate", "unlink"),
    ("memory/auto.py", "_append_diagnostic_record", "unlink"),
    ("memory/entry_access.py", "_refresh_entry_memory_mirror", "unlink"),
    ("memory/entry_history.py", "_rollback_memory_migration_for_test", "unlink"),
    ("memory/reconciliation_state.py", "_publish_bytes", "unlink"),
    ("memory/reconciliation_state.py", "_write_receipt_archive", "unlink"),
    ("project_inbox.py", "_prune_done", "unlink"),
    ("project_inbox.py", "claim_wake", "unlink"),
    ("project_memory_history.py", "_prune_versions", "unlink"),
    ("project_memory_history.py", "_publish_mirror_file", "unlink"),
    ("project_registry.py", "create_project", "unlink"),
    ("providers/stream_diagnostics.py", "write_stream_diagnostic", "unlink"),
    ("remote_sync/__init__.py", "_rewrite_cwd", "unlink"),
    ("remote_sync/memory.py", "_atomic_copy_file", "unlink"),
    ("remote_sync/memory.py", "_atomic_write_file", "unlink"),
    ("remote_sync/memory.py", "_copy_memory_file", "unlink"),
    ("remote_sync/memory.py", "_machine_id", "unlink"),
    ("remote_sync/project_publish.py", "_prune_versions", "unlink"),
    ("remote_sync/project_publish.py", "_remove_incoming", "unlink"),
    ("remote_sync/project_publish.py", "atomic_publish_file", "unlink"),
    ("runtime/loop/persistence.py", "_unlink", "unlink"),
    ("server/server.py", "_unlink_if_owned", "unlink"),
    ("server/transport.py", "remove_connection_file", "unlink"),
    ("server/transport.py", "write_connection_file", "unlink"),
    ("storage_format.py", "_publish_manifest", "unlink"),
    ("storage_format.py", "delete", "unlink"),
    ("tools/_atomic_replace.py", "atomic_create", "unlink"),
    ("tools/_atomic_replace.py", "atomic_replace", "unlink"),
    ("tools/_shared/background_output_archive.py", "_remove_generated_log", "unlink"),
    ("tools/_spill.py", "_evict_before", "unlink"),
    ("tools/_spill.py", "temporary_file", "unlink"),
    ("tools/_spill.py", "write_group", "unlink"),
    ("transcript_search/index.py", "_remove_sqlite_sidecars", "unlink"),
    ("transcript_search/index.py", "_temporary_database", "unlink"),
    ("tui/paste_runtime/__init__.py", "_delete_staged", "unlink"),
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
    found_delete: set[tuple[str, str, str]] = set()
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
                if name == "unlink":
                    found_delete.add((relative, function, name))
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
    assert found_delete == _RAW_DELETION
    assert found_variant_append == _VARIANT_APPEND_SEAMS
    assert _REMOTE_PROJECT_PUBLICATION_P07
