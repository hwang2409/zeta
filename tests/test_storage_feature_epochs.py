from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from tests.test_attention import _session
from zeta import storage_format
from zeta.memory.auto import AutoMemoryReconciler
from zeta.project_errors import ProjectRegistryError
from zeta.project_registry import ProjectRegistry
from zeta.skills import SkillCatalog
from zeta.storage_format import StorageFormatError, open_writer
from zeta.tools.registry import ToolRegistry


def _allowed_writer_default(node: ast.AST, parents: dict[ast.AST, ast.AST], path: Path) -> bool:
    if path != Path.cwd() / "src" / "zeta" / "cli" / "main.py":
        return False
    keyword = parents.get(node)
    call = parents.get(keyword) if isinstance(keyword, ast.keyword) else None
    return (
        isinstance(keyword, ast.keyword)
        and keyword.arg == "default"
        and isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and any(
            isinstance(argument, ast.Constant) and argument.value == "--target-epoch"
            for argument in call.args
        )
    )


def _writer_epoch_names(tree: ast.AST) -> set[str]:
    names = {"WRITER_EPOCH"}
    module_names = {"storage_format"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "zeta.storage_format":
            for imported in node.names:
                if imported.name == "WRITER_EPOCH":
                    names.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "zeta.storage_format":
                    module_names.add(imported.asname or "zeta.storage_format")
                elif imported.name == "storage_format":
                    module_names.add(imported.asname or imported.name)
    return names | {f"{module}.WRITER_EPOCH" for module in module_names}


def _writer_epoch_violations(path: Path, source: str) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    names = _writer_epoch_names(tree)
    violations = []
    for node in ast.walk(tree):
        is_name = isinstance(node, ast.Name) and node.id in names
        is_attribute = (
            isinstance(node, ast.Attribute)
            and node.attr == "WRITER_EPOCH"
        )
        if (is_name or is_attribute) and not _allowed_writer_default(node, parents, path):
            violations.append(f"{path}:{node.lineno}")
    return violations


def test_feature_gates_do_not_compare_against_writer_epoch() -> None:
    source_root = Path.cwd() / "src" / "zeta"
    violations = [
        violation
        for path in sorted(source_root.rglob("*.py"))
        if path.name != "storage_format.py"
        for violation in _writer_epoch_violations(path, path.read_text())
    ]
    assert not violations, (
        "compare against FEATURE_EPOCHS[<feature>] in storage_format.py: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from zeta.storage_format import WRITER_EPOCH as epoch\nif home_epoch == epoch: pass",
        "from zeta.storage_format import WRITER_EPOCH as epoch\ndef f(value=epoch): pass",
        "import zeta.storage_format as sf\nif home_epoch == sf.WRITER_EPOCH: pass",
    ],
)
def test_writer_epoch_guard_rejects_aliases(source: str, tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    assert _writer_epoch_violations(path, source)


@pytest.mark.parametrize("case", ["repair", "memory", "attention"])
def test_epoch5_home_writes_survive_a_new_writer_epoch(tmp_path, monkeypatch, case):
    opened = _session(tmp_path)
    opened.store.close()
    monkeypatch.setattr(storage_format, "WRITER_EPOCH", 6)

    if case == "repair":
        # the real conversation-store constructor repairs a torn tail.
        import zeta.core.store._log as log_module
        from zeta.core.store import ConversationStore

        monkeypatch.setattr(log_module, "WRITER_EPOCH", 6, raising=False)
        path = opened.store.path
        with path.open("ab") as handle:
            handle.write(b'{"seq": 99, "id": "torn"')
        repaired = ConversationStore(path.parent.parent, session_id=opened.store.session_id)
        try:
            assert list(path.parent.glob("conversation.quarantine.*"))
        finally:
            repaired.close()
        return

    if case == "memory":
        # recovery_committed writes through the real reconciler state path.
        import zeta.memory.auto as auto_module

        monkeypatch.setattr(auto_module, "WRITER_EPOCH", 6, raising=False)
        registry = ProjectRegistry(tmp_path / "projects")
        project = registry.create_project("memory", "scope", tmp_path)
        registry.initialize_memory(project.project_id)
        lease = open_writer(tmp_path)
        reconciler = AutoMemoryReconciler(
            registry=registry,
            project_id=project.project_id,
            session_id=opened.store.session_id,
            session_dir=opened.store.session_dir,
            invoke=lambda *_args: "unused",
            storage_lease=lease,
        )
        try:
            try:
                asyncio.run(
                    reconciler.recovery_committed(
                        quarantine_id="q1", retained_last_sequence=0, memory_review_required=True
                    )
                )
            except ProjectRegistryError:
                pass
            state_path = opened.store.session_dir / "memory-reconcile.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            assert state.get("recovery_quarantine_id") == "q1"
        finally:
            asyncio.run(reconciler.close())
        return

    # request_attention is the real non-decision request path and uses the observed epoch.
    from zeta import attention_records

    monkeypatch.setattr(attention_records, "ATTENTION_KIND_EPOCH", 6, raising=False)
    attention = _session(tmp_path)
    registry_tools = ToolRegistry(
        tmp_path, session_store=attention.store, skill_catalog=SkillCatalog.empty()
    )
    try:
        try:
            result = asyncio.run(
                registry_tools.definitions_by_name["request_attention"].handler(
                    {"title": "answer", "why": "context", "kind": "answer"}
                )
            )
        except (StorageFormatError, ValueError):
            result = {"structuredContent": {"kind": None}}
        assert result["structuredContent"]["kind"] == "answer"
    finally:
        asyncio.run(registry_tools.close())
        attention.store.close()
