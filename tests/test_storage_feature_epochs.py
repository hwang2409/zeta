from __future__ import annotations

import ast
import asyncio
import io
import json
import tokenize
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


def test_feature_epochs_are_fixed_int_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "zeta" / "storage_format.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    feature_epochs = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "FEATURE_EPOCHS"
    )
    assert isinstance(feature_epochs.value, ast.Dict)
    for key, value in zip(feature_epochs.value.keys, feature_epochs.value.values):
        key_text = ast.unparse(key)
        source = ast.unparse(value)
        assert isinstance(value, ast.Constant) and type(value.value) is int, (
            f"FEATURE_EPOCHS[{key_text}] must be a fixed int literal, not {source}; "
            "a feature's epoch must not move when WRITER_EPOCH changes"
        )


def _writer_epoch_tokens(path: Path, source: str) -> list[str]:
    # token-level: only real WRITER_EPOCH name references count, not strings or
    # comments. an aliased import still carries the name on its import line.
    return [
        f"{path}:{token.start[0]}"
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME and token.string == "WRITER_EPOCH"
    ]


def test_writer_epoch_token_only_in_storage_format() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "zeta"
    scanned = sorted(source_root.rglob("*.py"))
    assert scanned, f"no python files found under {source_root}"
    violations = [
        hit
        for path in scanned
        if path != source_root / "storage_format.py"
        for hit in _writer_epoch_tokens(path, path.read_text())
    ]
    assert not violations, (
        "WRITER_EPOCH must stay inside src/zeta/storage_format.py. for a feature "
        "gate, compare against FEATURE_EPOCHS[...] / feature_enabled(); for the "
        "writer's own epoch, call storage_format.writer_epoch(). offenders: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize(
    "source",
    [
        # an aliased import cannot hide the name: the import line carries it.
        "from zeta.storage_format import WRITER_EPOCH as epoch\nx = epoch",
        # a direct comparison use.
        "if home_epoch == WRITER_EPOCH: pass",
    ],
)
def test_writer_epoch_token_checker_flags_real_references(
    source: str, tmp_path: Path
) -> None:
    assert _writer_epoch_tokens(tmp_path / "sample.py", source)


def test_writer_epoch_token_checker_ignores_strings_and_comments() -> None:
    source = '"""doc that names WRITER_EPOCH"""\n# WRITER_EPOCH in a comment\n'
    assert not _writer_epoch_tokens(Path("sample.py"), source)


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
