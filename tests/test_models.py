import json
from pathlib import Path

from zeta.tui.models import load_model_catalog


def test_codex_catalog_reads_models_cache(tmp_path: Path, monkeypatch) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "fetched_at": "2026-08-25T18:19:32.918844Z",
                "models": [{"slug": f"gpt-5.{index}"} for index in range(8)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    catalog = load_model_catalog("codex")

    assert catalog == frozenset(f"gpt-5.{index}" for index in range(8))
