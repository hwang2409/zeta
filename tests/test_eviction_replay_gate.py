from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
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
    expected = {
        "context.json": "3715866d9380bbe995e8f913b4f09f06d886274fea368d891fb9fde6fe2c7c93",
        "wire.json": "5e87bf4d29a4ff4e83e587d42aa47058e73bd3dfa2ae3cf96a1d950133d24302",
        "appends.json": "c1eb7feded93c55dff6c82afe2f69bdf53f5074dd5fd69f5dcc2e9957c32a36b",
        "readback.json": "3715866d9380bbe995e8f913b4f09f06d886274fea368d891fb9fde6fe2c7c93",
    }
    assert {key: record["records"][0]["digests"][key] for key in expected} == expected


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


def _mutated_ref(tmp_path: Path, replacements: list[tuple[str, str]]) -> Path:
    ref = tmp_path / "mutated"
    import shutil
    for name in ("src", "scripts", "tests/fixtures"):
        source = ROOT / name
        target = ref / name
        shutil.copytree(source, target)
    script = ref / "scripts/eviction_replay_gate.py"
    text = script.read_text()
    for old, new in replacements:
        assert text.count(old) == 1
        text = text.replace(old, new)
    script.write_text(text)
    return ref


def _compare(base: Path, head: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "compare", "--base", str(base), "--head", str(head), "--fixture", "--out", str(tmp_path / "manifest.json")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_fixture_compare_rejects_receipt_mutation_without_values(tmp_path: Path) -> None:
    head = _mutated_ref(tmp_path, [("fixture result ", "mutated result ")])
    result = _compare(ROOT, head, tmp_path)
    assert result.returncode != 0
    assert "FAIL fixture: invocation 1: appends.json at /" in result.stderr
    assert "mutated result" not in result.stderr
    assert "fixture result" not in result.stderr


def test_fixture_compare_rejects_retained_floor_bound_mutation(tmp_path: Path) -> None:
    head = _mutated_ref(tmp_path, [])
    fact_floor = head / "src/zeta/context_eviction/_fact_floor.py"
    text = fact_floor.read_text()
    text = text.replace("FACT_FLOOR_CHAR_CAP = 4_096", "FACT_FLOOR_CHAR_CAP = 100")
    text = text.replace("    assert len(text) <= FACT_FLOOR_CHAR_CAP", "    # mutated: production no longer enforces its floor cap")
    fact_floor.write_text(text)
    result = _compare(ROOT, head, tmp_path)
    assert result.returncode != 0
    assert "retained-floor bound check failed" in result.stderr
    assert "digest" not in result.stderr.lower()


def test_fixture_compare_rejects_retained_floor_limit_change(tmp_path: Path) -> None:
    head = _mutated_ref(tmp_path, [])
    fact_floor = head / "src/zeta/context_eviction/_fact_floor.py"
    fact_floor.write_text(fact_floor.read_text().replace("_NEWEST_LIMIT = 16", "_NEWEST_LIMIT = 17"))
    result = _compare(ROOT, head, tmp_path)
    assert result.returncode != 0
    assert "limit change" in result.stderr
