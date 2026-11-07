from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPECTED = {
    "context.json": "8eeebf7a00e96a1ade54bcc1225ca26718244cc788520a1c09f4a7e12bd5ff93",
    "wire.json": "366f874c7fd1626c5087a8c04fa4908fca9008c1d92da941d1601a55cd796cda",
    "appends.json": "9b3b346a2b2cfe5a307a7c9259b684921a6fbb64fdd807e3f8031e5a944e4b09",
    "readback.json": "8eeebf7a00e96a1ade54bcc1225ca26718244cc788520a1c09f4a7e12bd5ff93",
}


def test_fixture_replay_uses_production_path_and_golden_digests(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "run", "--fixture", "--artifact-dir", str(tmp_path / "artifacts")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    record = json.loads(result.stdout.strip())
    assert Path(record["zeta_file"]).resolve().is_relative_to(ROOT.resolve())
    assert {key: record["records"][0]["digests"][key] for key in EXPECTED} == EXPECTED


def test_fixture_compare_same_ref(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "compare", "--base", str(ROOT), "--head", str(ROOT), "--fixture", "--out", str(tmp_path / "manifest.json")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["sessions"][0]["status"] == "PASS"
