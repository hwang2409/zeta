from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from tests.test_attention import _session
from zeta import storage_format
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry


def test_feature_gates_do_not_compare_against_writer_epoch() -> None:
    source_root = Path.cwd() / "src" / "zeta"
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        if path.name == "storage_format.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        if path.relative_to(source_root).as_posix() == "cli/main.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "WRITER_EPOCH":
                violations.append(f"{path.relative_to(Path.cwd())}:{node.lineno}")

    assert not violations, (
        "compare against FEATURE_EPOCHS[<feature>] in storage_format.py: "
        + ", ".join(violations)
    )


def test_attention_write_uses_observed_epoch_when_writer_epoch_moves(tmp_path, monkeypatch):
    opened = _session(tmp_path)
    try:
        monkeypatch.setattr(storage_format, "WRITER_EPOCH", 6)
        registry = ToolRegistry(
            tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
        )
        result = asyncio.run(
            registry.definitions_by_name["request_attention"].handler(
                {"title": "answer", "why": "context", "kind": "answer"}
            )
        )
        assert result["structuredContent"]["kind"] == "answer"
        asyncio.run(registry.close())
    finally:
        opened.store.close()
