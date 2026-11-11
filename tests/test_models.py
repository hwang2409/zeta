import json
from pathlib import Path

import pytest

from zeta.tui.models import _model_names_from_payload, load_model_catalog


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"data": [{"id": "claude-new"}, {"id": 3}, "raw"]}, frozenset({"claude-new", "raw"})),
        ({"models": [{"name": "llama:latest"}, {"name": None}]}, frozenset({"llama:latest"})),
        ({"data": [{"id": {}}]}, frozenset()),
        ([], None),
        ({"data": "bad"}, None),
    ],
)
def test_model_catalog_payload_parsing_is_tolerant(payload, expected) -> None:
    assert _model_names_from_payload(payload) == expected


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
