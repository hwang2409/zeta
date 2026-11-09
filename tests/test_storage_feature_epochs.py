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


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _cli_main_path() -> Path:
    return Path.cwd() / "src" / "zeta" / "cli" / "main.py"


def _allowed_writer_import(node: ast.alias, path: Path) -> bool:
    # the single unaliased import that brings WRITER_EPOCH into cli/main.py.
    return path == _cli_main_path() and node.asname is None


def _allowed_writer_default(
    node: ast.Name, parents: dict[ast.AST, ast.AST], path: Path
) -> bool:
    if path != _cli_main_path():
        return False
    keyword = parents.get(node)
    if not isinstance(keyword, ast.keyword) or keyword.arg != "default":
        return False
    call = parents.get(keyword)
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    return (
        call.func.attr == "add_argument"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--target-epoch"
    )


def _writer_epoch_violation(
    node: ast.AST, parents: dict[ast.AST, ast.AST], path: Path
) -> bool:
    if isinstance(node, ast.alias):
        return node.name == "WRITER_EPOCH" and not _allowed_writer_import(node, path)
    if isinstance(node, ast.Name):
        return node.id == "WRITER_EPOCH" and not _allowed_writer_default(
            node, parents, path
        )
    return isinstance(node, ast.Attribute) and node.attr == "WRITER_EPOCH"


def _writer_epoch_violations(path: Path, source: str) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    parents = _parent_map(tree)
    return [
        f"{path}:{node.lineno}"
        for node in ast.walk(tree)
        if _writer_epoch_violation(node, parents, path)
    ]


def test_feature_gates_do_not_compare_against_writer_epoch() -> None:
    source_root = Path.cwd() / "src" / "zeta"
    violations = [
        violation
        for path in sorted(source_root.rglob("*.py"))
        if path.name != "storage_format.py"
        for violation in _writer_epoch_violations(path, path.read_text())
    ]
    assert not violations, (
        "compare against FEATURE_EPOCHS[...] / feature_enabled() in "
        "src/zeta/storage_format.py: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from zeta.storage_format import WRITER_EPOCH as epoch\nif home_epoch == epoch: pass",
        "from zeta.storage_format import WRITER_EPOCH as epoch\ndef f(value=epoch): pass",
        "from .storage_format import WRITER_EPOCH as e\nif home_epoch == e: pass",
        "from ..storage_format import WRITER_EPOCH as e\ndef f(value=e): pass",
        "def f(value=WRITER_EPOCH): pass",
        "import zeta.storage_format as sf\ndef f(value=sf.WRITER_EPOCH): pass",
        "import storage_format as sf\nif home_epoch == sf.WRITER_EPOCH: pass",
    ],
)
def test_writer_epoch_guard_rejects_aliases(source: str, tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    assert _writer_epoch_violations(path, source)


@pytest.mark.parametrize(
    "source",
    [
        # an aliased import hides the epoch behind another name.
        "from ..storage_format import WRITER_EPOCH as hidden\ndef f(value=hidden): pass",
        # a second use outside the allowed --target-epoch default.
        'p.add_argument("--target-epoch", type=int, default=WRITER_EPOCH)\nx = WRITER_EPOCH',
        # a module attribute use.
        "import zeta.storage_format as sf\nx = sf.WRITER_EPOCH",
    ],
)
def test_writer_epoch_guard_rejects_cli_aliases_and_extra_uses(source: str) -> None:
    assert _writer_epoch_violations(_cli_main_path(), source)


def test_writer_epoch_guard_accepts_the_one_cli_site() -> None:
    source = (
        "from ..storage_format import WRITER_EPOCH\n"
        'p.add_argument("--target-epoch", type=int, default=WRITER_EPOCH)\n'
    )
    assert not _writer_epoch_violations(_cli_main_path(), source)


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
