from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_controller_snapshot_is_stable_and_production_readable(tmp_path: Path) -> None:
    from scripts.eviction_replay_gate import _fixture_store, _snapshot_session

    source_root = tmp_path / "source"
    _fixture_store(source_root)
    source = source_root / "fixture"
    snapshot = tmp_path / "snapshot" / "fixture"
    info = _snapshot_session(source, snapshot)
    original = (snapshot / "conversation.jsonl").read_bytes()
    with (source / "conversation.jsonl").open("ab") as handle:
        handle.write(b'{"late":true}\n')
    assert (snapshot / "conversation.jsonl").read_bytes() == original
    assert info["row_count"] == 145
    assert info["sha256"] == hashlib.sha256(original).hexdigest()
    assert info["sha256"] != hashlib.sha256((source / "conversation.jsonl").read_bytes()).hexdigest()


def test_compare_manifest_uses_one_snapshot_for_both_sides(tmp_path: Path) -> None:
    from scripts.eviction_replay_gate import _fixture_store

    home = tmp_path / "home"
    source = home / ".zeta" / "sessions" / "fixture"
    source.parent.mkdir(parents=True)
    _fixture_store(source.parent)
    original = (source / "conversation.jsonl").read_bytes()
    (source / "meta.json").write_text(json.dumps({"compaction_budget": 4000, "retained_tail": 3, "provider": "anthropic"}))
    (home / ".zeta" / "storage-format.json").write_text(json.dumps({"writer_epoch": 2}))
    head = _mutated_ref(tmp_path, [])
    script = head / "scripts/eviction_replay_gate.py"
    marker = "    root = Path(args.session).expanduser() if args.session else None\n"
    mutation = marker + '    if os.environ.get("EVICTION_MUTATE_SOURCE"):\n        source = Path(os.environ["EVICTION_MUTATE_SOURCE"])\n        row = json.loads(source.read_text().splitlines()[-1])\n        row.update(id="00000000000000000000000000000093", parent_id="00000000000000000000000000000092", seq=146)\n        with source.open("ab") as handle:\n            handle.write((json.dumps(row) + "\\n").encode())\n'
    script.write_text(script.read_text().replace(marker, mutation, 1))
    result = _compare(
        ROOT, head, tmp_path, fixture=False,
        env={"HOME": str(home), "EVICTION_MUTATE_SOURCE": str(source / "conversation.jsonl")},
        extra=["--session", "fixture"],
    )
    assert result.returncode == 0, result.stderr
    session = json.loads((tmp_path / "manifest.json").read_text())["sessions"][0]
    input_info = session["input"]
    # the source mutation occurs after the controller snapshot; both sides use its digest
    assert input_info["row_count"] == 145
    assert input_info["sha256"] == hashlib.sha256(original).hexdigest()
    assert input_info["sha256"] != hashlib.sha256((source / "conversation.jsonl").read_bytes()).hexdigest()


def test_fixture_replay_uses_production_path_and_golden_digests(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "run", "--fixture", "--keep-private-artifacts", str(tmp_path / "artifacts")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    record = json.loads(result.stdout.strip())
    assert "zeta_file" not in record
    expected = {
        "context.json": "3715866d9380bbe995e8f913b4f09f06d886274fea368d891fb9fde6fe2c7c93",
        "wire.json": "5e87bf4d29a4ff4e83e587d42aa47058e73bd3dfa2ae3cf96a1d950133d24302",
        "appends.json": "c1eb7feded93c55dff6c82afe2f69bdf53f5074dd5fd69f5dcc2e9957c32a36b",
        "readback.json": "3715866d9380bbe995e8f913b4f09f06d886274fea368d891fb9fde6fe2c7c93",
    }
    assert {key: record["records"][0]["digests"][key] for key in expected} == expected


def test_fixture_replay_removes_private_artifacts_by_default(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "run", "--fixture"],
        cwd=ROOT, check=True, capture_output=True, text=True,
        env={**__import__("os").environ, "TMPDIR": str(tmp_path)},
    )
    assert "artifact_dir" not in json.loads(result.stdout.strip())
    assert not list(tmp_path.glob("eviction-artifacts-*"))
    assert not list(tmp_path.glob("eviction-gate-*"))


def test_fixture_replay_keeps_private_artifacts_when_opted_in(tmp_path: Path) -> None:
    artifacts = tmp_path / "kept"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "run", "--fixture", "--keep-private-artifacts", str(artifacts)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert list(artifacts.rglob("context.json"))


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


def _compare(base: Path, head: Path, tmp_path: Path, *, env: dict[str, str] | None = None, extra: list[str] | None = None, fixture: bool = True) -> subprocess.CompletedProcess[str]:
    fixture_arg = ["--fixture"] if fixture else []
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "compare", "--base", str(base), "--head", str(head), *fixture_arg, "--out", str(tmp_path / "manifest.json"), *(extra or [])],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})},
    )


def test_compare_records_exact_identity_for_each_worktree(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    subprocess.run(["git", "worktree", "add", "--detach", str(base), "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(head), "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(head), "config", "user.email", "test@example.invalid"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(head), "config", "user.name", "test"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(head), "commit", "--allow-empty", "-m", "identity test"], check=True, capture_output=True, text=True)
        result = _compare(base, head, tmp_path)
        assert result.returncode == 0, result.stderr
        manifest = json.loads((tmp_path / "manifest.json").read_text())["sessions"][0]
        assert manifest["base"]["git_head"] == subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
        assert manifest["head"]["git_head"] == subprocess.check_output(["git", "-C", str(head), "rev-parse", "HEAD"], text=True).strip()
        assert manifest["base"]["git_head"] != manifest["head"]["git_head"]
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(head)], cwd=ROOT, check=False, capture_output=True)
        subprocess.run(["git", "worktree", "remove", "--force", str(base)], cwd=ROOT, check=False, capture_output=True)


def test_compare_rejects_import_outside_expected_worktree(tmp_path: Path) -> None:
    head = _mutated_ref(tmp_path, [])
    shutil.rmtree(head / "src" / "zeta")
    result = _compare(ROOT, head, tmp_path)
    assert result.returncode != 0
    assert "identity outside expected worktree" in result.stderr


def test_compare_reports_stored_floor_violation_in_manifest(tmp_path: Path) -> None:
    from scripts.eviction_replay_gate import _assembler, _fixture_store
    from zeta.core.store import ConversationStore

    home = tmp_path / "home"
    source = home / ".zeta" / "sessions" / "fixture"
    source.parent.mkdir(parents=True)
    _fixture_store(source.parent)
    import asyncio

    store = ConversationStore(source.parent, session_id="fixture")
    try:
        assembler = _assembler(store, budget=4000, tail=3)
        asyncio.run(assembler.assemble_context(backend=None, force=True, bypass_eviction_hysteresis=True))
        rows = [json.loads(line) for line in store.path.read_text().splitlines()]
        marker = rows[-1]
        assert marker["seq"] == 146
        store.append_compaction_marker("superseding marker", 1, 146, replaces=[marker["id"]])
        rows = [json.loads(line) for line in store.path.read_text().splitlines()]
        marker = next(row for row in rows if row.get("seq") == 146)
        floor = {
            "version": 1,
            "oldest": [
                {"fact": {"type": "child_head", "handle": f"child-{number}-" + "x" * 70, "sha": "a" * 40}, "first_seq": number, "last_seq": number, "value_rank": [number, 0]}
                for number in range(1, 10)
            ],
            "newest": [
                {"fact": {"type": "child_head", "handle": f"child-{number}-" + "x" * 70, "sha": "b" * 40}, "first_seq": number, "last_seq": number, "value_rank": [number, 0]}
                for number in range(25, 9, -1)
            ],
        }
        marker["data"]["fact_floor"] = floor
        store.path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    finally:
        store.close()
    (source / "meta.json").write_text(json.dumps({"compaction_budget": 4000, "retained_tail": 3, "provider": "anthropic"}))
    head = _mutated_ref(tmp_path, [])
    fact_floor = head / "src/zeta/context_eviction/_fact_floor.py"
    fact_floor.write_text(fact_floor.read_text().replace("_OLDEST_LIMIT = 8", "_OLDEST_LIMIT = 4"))
    script = head / "scripts/eviction_replay_gate.py"
    script.write_text(script.read_text().replace('"oldest": _fact_floor._OLDEST_LIMIT,', '"oldest": 8,', 1).replace('"tokens": _fact_floor.FACT_FLOOR_TOKEN_CAP,', '"tokens": 831,', 1))

    result = _compare(
        ROOT,
        head,
        tmp_path,
        env={"HOME": str(home)},
        extra=["--session", "fixture", "--timeout", "10"],
        fixture=False,
    )
    assert result.returncode == 0, result.stderr
    session = json.loads((tmp_path / "manifest.json").read_text())["sessions"][0]
    assert session["status"] == "PASS"
    for side in ("base", "head"):
        violations = [
            item
            for record in session[side]["records"]
            for item in record["trace"]["stored_floor_violations"]
        ]
        assert violations
        assert set(violations) == {"seq=146 invalid"}


def test_fixture_compare_fails_when_code_appends_floor_violation() -> None:
    from scripts.eviction_replay_gate import _floor_metrics

    floor = {"version": 1, "oldest": [], "newest": [{"fact": {"type": "pr", "pr": 1}, "first_seq": 1, "last_seq": 1, "value_rank": [1, 0]}]}
    with pytest.raises(RuntimeError, match="retained-floor bound check failed"):
        _floor_metrics([{"seq": 1, "data": {"fact_floor": floor}}], {"oldest": 8, "newest": 0, "chars": 4096, "tokens": 4096}, "appended")


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
    text = text.replace('    lines = ["[retained typed facts]"]\n', '    lines = ["[retained typed facts]"] + ["x" * 5000]\n', 1)
    text = text.replace("    assert len(text) <= FACT_FLOOR_CHAR_CAP", "    # mutated: production emits an oversized floor", 1)
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


def test_fixture_replay_fails_when_stability_cap_is_not_reached(tmp_path: Path) -> None:
    head = _mutated_ref(tmp_path, [("for invocation in range(1, 9):", "for invocation in range(1, 2):") ])
    result = subprocess.run(
        [sys.executable, str(head / "scripts/eviction_replay_gate.py"), "run", "--fixture", "--keep-private-artifacts", str(tmp_path / "artifacts")],
        cwd=ROOT, check=False, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(head / "src")},
    )
    assert result.returncode != 0
    assert "did not stabilize within 8 invocations" in result.stderr


def _failure_copy(tmp_path: Path, mode: str) -> Path:
    head = _mutated_ref(tmp_path, [])
    script = head / "scripts/eviction_replay_gate.py"
    marker = 'assembler = _assembler(store, budget=budget, tail=tail, system_prompt=meta.get("system_prompt", ""))'
    if mode == "raise":
        replacement = 'raise RuntimeError("fixture side failure")'
    elif mode == "crash":
        replacement = "os._exit(3)"
    else:
        replacement = "time.sleep(60)"
    text = script.read_text()
    assert text.count(marker) == 1
    script.write_text(text.replace(marker, replacement))
    return head


def _bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@pytest.mark.parametrize("mode", ["success", "raise", "crash", "timeout"])
def test_failure_mode_cleanup_through_cli(tmp_path: Path, mode: str) -> None:
    head = ROOT if mode == "success" else _failure_copy(tmp_path, mode)
    controlled = tmp_path / mode
    tmpdir = controlled / "tmp"
    scratch = controlled / "scratch"
    tmpdir.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "compare", "--base", str(ROOT), "--head", str(head), "--fixture", "--timeout", str(0.2 if mode == "timeout" else 60), "--scratch-root", str(scratch), "--out", str(controlled / "manifest.json")],
        cwd=ROOT, env={**os.environ, "TMPDIR": str(tmpdir)}, capture_output=True, text=True, check=False,
    )
    if mode == "success":
        assert result.returncode == 0
    else:
        assert result.returncode != 0
        assert "FAIL fixture" in result.stderr
    assert _bytes(tmpdir) == 0
    assert _bytes(scratch) == 0
    for line in result.stderr.splitlines():
        if "pgid=" in line:
            pgid = int(line.rsplit("pgid=", 1)[1])
            with pytest.raises(ProcessLookupError):
                os.killpg(pgid, 0)


def test_interrupt_cleanup_through_cli(tmp_path: Path) -> None:
    head = _failure_copy(tmp_path, "interrupt")
    controlled = tmp_path / "interrupt"
    tmpdir, scratch = controlled / "tmp", controlled / "scratch"
    tmpdir.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "compare", "--base", str(ROOT), "--head", str(head), "--fixture", "--timeout", "60", "--scratch-root", str(scratch), "--out", str(controlled / "manifest.json")],
        cwd=ROOT, env={**os.environ, "TMPDIR": str(tmpdir)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (scratch / "fixture" / "head" / "sessions").exists():
                break
            time.sleep(0.01)  # test-contract: latency wait for child scratch readiness
        os.kill(process.pid, signal.SIGINT)
        _stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode != 0
    assert "FAIL fixture" in stderr
    assert _bytes(tmpdir) == 0
    assert _bytes(scratch) == 0
    for line in stderr.splitlines():
        if "pgid=" in line:
            with pytest.raises(ProcessLookupError):
                os.killpg(int(line.rsplit("pgid=", 1)[1]), 0)


def test_child_run_does_not_create_tmp_paths(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    scratch = tmp_path / "scratch"
    empty.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts/eviction_replay_gate.py"), "run", "--fixture", "--scratch-root", str(scratch)], cwd=ROOT, env={**os.environ, "TMPDIR": str(empty)}, check=True, capture_output=True, text=True)
    assert _bytes(empty) == 0


def test_summary_guard_is_installed_on_every_assembler() -> None:
    import asyncio
    import importlib.util
    spec = importlib.util.spec_from_file_location("eviction_replay_gate", ROOT / "scripts/eviction_replay_gate.py")
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    class Policy:
        async def summarize_chunked(self) -> str:
            return "unexpected"

    class Assembler:
        compaction_policy = Policy()

    gate._install_summary_guard(Assembler())
    try:
        asyncio.run(Assembler.compaction_policy.summarize_chunked())
    except RuntimeError as error:
        assert str(error) == "eviction replay reached model-summary fallback"
    else:
        raise AssertionError("summary fallback was not blocked")
