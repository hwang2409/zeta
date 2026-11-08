#!/usr/bin/env python3
"""Run the stored-format probe against the selected first-parent predecessor."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zeta.storage_format import WRITER_EPOCH, StorageFormatError, compare_registry_dumps

PROBE_PATHS = ("src/zeta/storage_format.py", "tests/compatibility/storage_probe.py")
ROUNDTRIP = Path(__file__).resolve().parents[1] / "tests/compatibility/storage_roundtrip.py"
CLASSIFIER_PREFIXES = ("src/zeta/", "tests/compatibility/", ".github/workflows/", "scripts/")
CLASSIFIER_FILES = {"pyproject.toml", "uv.lock"}
_SUBPROCESS_TIMEOUT = 180
_GIT_TIMEOUT = 60


def _run_phase(
    command: tuple[str, ...],
    *,
    phase: str,
    timeout: int = _SUBPROCESS_TIMEOUT,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    check = kwargs.pop("check", False)
    try:
        return subprocess.run(command, timeout=timeout, check=check, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{phase} timed out after {timeout} seconds") from exc


def _git(*args: str) -> str:
    result = _run_phase(
        ("git", *args),
        phase=f"git {' '.join(args[:2])}",
        timeout=_GIT_TIMEOUT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout.strip()


def qualifying(commit: str) -> bool:
    """Classify first-parent changes, including merge commits."""
    parents = _git("rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) == 1:
        changed = _git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit).splitlines()
    else:
        changed = _git("diff", "--name-only", f"{commit}^1", commit).splitlines()
    return any(path in CLASSIFIER_FILES or path.startswith(CLASSIFIER_PREFIXES) for path in changed)


def has_probe(commit: str) -> bool:
    return all(
        _run_phase(
            ("git", "cat-file", "-e", f"{commit}:{path}"),
            phase=f"probe lookup for {commit}",
            timeout=_GIT_TIMEOUT,
            capture_output=True,
            check=False,
        ).returncode == 0
        for path in PROBE_PATHS
    )


def select_predecessor(*, event: str, head: str, base: str) -> tuple[str, list[str]]:
    anchor = base if event == "pull_request" else _git("rev-parse", f"{head}^")
    commits = _git("rev-list", "--first-parent", anchor).splitlines()
    predecessor = next((commit for commit in commits if qualifying(commit)), None)
    if predecessor is None:
        raise RuntimeError(f"no qualifying predecessor at or before {anchor}")
    return predecessor, commits[commits.index(predecessor):]


def _digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != ".storage-format.lock"
        and ".storage-format-shape-locks" not in path.parts
    }


def _assert_mutation_delta(before: dict[str, str], after: dict[str, str], allowed: Iterable[str]) -> None:
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    unexpected = changed - set(allowed)
    if unexpected:
        raise RuntimeError(f"old-writer mutation changed undeclared paths: {sorted(unexpected)!r}")


def _assert_roundtrip_delta(
    before: dict[str, str], after: dict[str, str], skipped: set[str]
) -> None:
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    required = {
        "session transcript and metadata": any(path.startswith("sessions/") for path in changed),
        "project memory and inbox": any("/memory/" in path for path in changed)
        and any("/inbox/" in path for path in changed),
        "accounts metadata": "accounts.json" in changed
        and any(path.startswith("accounts/") for path in changed),
        "mcp token store": any(path.startswith("mcp-tokens/") and path.endswith(".json") for path in changed),
    }
    required = {family: present for family, present in required.items() if family not in skipped}
    missing = [family for family, present in required.items() if not present]
    if missing:
        raise RuntimeError(f"predecessor did not write stored families: {', '.join(missing)}")
    if "session transcript and metadata" not in skipped:
        recovery = [Path(path).name for path in changed if path.startswith("sessions/")]
        if not any(name.startswith("conversation.recovery.") for name in recovery):
            raise RuntimeError("predecessor did not publish a transcript recovery record")
        if not any(name.startswith("conversation.quarantine.") and name.endswith(".jsonl") for name in recovery):
            raise RuntimeError("predecessor did not publish transcript quarantine payload")


def _roundtrip_result(result: subprocess.CompletedProcess[str], phase: str) -> dict[str, object]:
    if result.returncode:
        raise RuntimeError(f"{phase} failed: {result.stderr or result.stdout}")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{phase} did not report observed stored data") from exc
    if not isinstance(value, dict) or value.get("stage") != phase:
        raise RuntimeError(f"{phase} did not report observed stored data")
    suffix = "predecessor" if phase == "predecessor" else "candidate"
    expected = {
        "session": f"{suffix}-session",
        "projects": f"{suffix}-memory\n",
        "accounts": f"fake-{suffix}-access",
        "mcp": f"fake-{suffix}-mcp",
    }
    accessors = {
        "session": lambda item: item["texts"][-1],
        "projects": lambda item: item["memory"],
        "accounts": lambda item: item["access_token"],
        "mcp": lambda item: item["access_token"],
    }
    try:
        skipped = set(value["skipped"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{phase} did not report skipped stored families") from exc
    family_names = {
        "session": "session transcript and metadata",
        "projects": "project memory and inbox",
        "accounts": "accounts metadata",
        "mcp": "mcp token store",
    }
    try:
        observed = {
            family: accessors[family](value[family])
            for family in expected
            if family_names[family] not in skipped
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{phase} did not report observed stored data") from exc
    wanted = {
        family: expected_value
        for family, expected_value in expected.items()
        if family_names[family] not in skipped
    }
    if observed != wanted:
        raise RuntimeError(f"{phase} observed unexpected stored data: {observed!r}")
    return value


def _roundtrip_command(
    home: Path,
    stage: str,
    *,
    advance_from: int | None = None,
    skipped: Iterable[str] = (),
) -> tuple[str, ...]:
    command = ("uv", "run", "--frozen", "python", str(ROUNDTRIP), "--home", str(home), "--stage", stage)
    if advance_from is not None:
        command += ("--advance-from", str(advance_from))
    for family in skipped:
        command += ("--skip", family)
    return command


def _run_roundtrips(source: Path, predecessor_epoch: int) -> None:
    with (
        tempfile.TemporaryDirectory(prefix="storage-format-roundtrip-") as temporary,
        tempfile.TemporaryDirectory(prefix="storage-format-uv-cache-") as cache,
    ):
        home = Path(temporary)
        environment = {**os.environ, "UV_CACHE_DIR": cache}
        (home / "storage-format.json").write_text(json.dumps({"writer_epoch": predecessor_epoch}) + "\n")
        (home / ".storage-format.lock").touch()
        high = home / "higher"
        shutil.copytree(home, high)
        (high / "storage-format.json").write_text(json.dumps({"writer_epoch": predecessor_epoch + 1}) + "\n")
        before_high = _digest_tree(high)
        rejected = _run_phase(
            _roundtrip_command(high, "predecessor"),
            phase="higher-epoch predecessor refusal",
            cwd=source,
            env={**environment, "PYTHONPATH": str(source / "src")},
            text=True,
            capture_output=True,
            check=False,
        )
        if rejected.returncode == 0:
            raise RuntimeError("predecessor accepted a higher-epoch home")
        refusal = rejected.stderr or rejected.stdout
        if "is newer than writer capability" not in refusal:
            raise RuntimeError(f"predecessor higher-epoch refusal was not clean: {refusal}")
        if _digest_tree(high) != before_high:
            raise RuntimeError("predecessor mutated a higher-epoch home")

        before = _digest_tree(home)
        predecessor = _run_phase(
            _roundtrip_command(home, "predecessor"),
            phase="predecessor",
            cwd=source,
            env={**environment, "PYTHONPATH": str(source / "src")},
            text=True,
            capture_output=True,
            check=False,
        )
        predecessor_value = _roundtrip_result(predecessor, "predecessor")
        skipped = set(predecessor_value["skipped"])
        _assert_roundtrip_delta(before, _digest_tree(home), skipped)

        advance_from = predecessor_epoch if WRITER_EPOCH > predecessor_epoch else None
        candidate = _run_phase(
            _roundtrip_command(
                home, "candidate", advance_from=advance_from, skipped=skipped
            ),
            phase="candidate",
            cwd=ROUNDTRIP.parents[1],
            text=True,
            capture_output=True,
            env={**environment, "PYTHONPATH": str(ROUNDTRIP.parents[1] / "src")},
            check=False,
        )
        _roundtrip_result(candidate, "candidate")

        before_reader = _digest_tree(home)
        reader = _run_phase(
            _roundtrip_command(home, "reader", skipped=skipped),
            phase="predecessor reader",
            cwd=source,
            env={**environment, "PYTHONPATH": str(source / "src")},
            text=True,
            capture_output=True,
            check=False,
        )
        if predecessor_epoch == WRITER_EPOCH:
            _roundtrip_result(reader, "reader")
        else:
            if reader.returncode == 0:
                raise RuntimeError("predecessor read the candidate-advanced higher-epoch home")
            refusal = reader.stderr or reader.stdout
            if "is newer than writer capability" not in refusal:
                raise RuntimeError(f"predecessor candidate-output refusal was not clean: {refusal}")
        if _digest_tree(home) != before_reader:
            raise RuntimeError("predecessor reader mutated the candidate-written home")
    print("::notice::selected predecessor two-way stored-family round trips passed")


def _writer_epoch(source: Path) -> int:
    tree = ast.parse((source / "src/zeta/storage_format.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "WRITER_EPOCH"
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) is int
        ):
            return node.value.value
    raise RuntimeError("predecessor WRITER_EPOCH is not a literal integer")


def _probe_source(source: Path, output: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(source / "src")}
    result = _run_phase(
        ("uv", "run", "--frozen", "python", "tests/compatibility/storage_probe.py"),
        phase="predecessor registry probe",
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"predecessor probe failed: {result.stderr or result.stdout}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"predecessor probe emitted malformed json: {exc}") from exc
    output.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _copy_predecessor_lock(predecessor: str, destination: Path) -> None:
    with destination.open("wb") as handle:
        _run_phase(
            ("git", "show", f"{predecessor}:uv.lock"),
            phase="predecessor lock extraction",
            timeout=_GIT_TIMEOUT,
            check=True,
            stdout=handle,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    event = os.environ.get("GITHUB_EVENT_NAME", "push")
    head = os.environ.get("GITHUB_SHA", "HEAD")
    base = os.environ.get("GITHUB_BASE_SHA") or _git("rev-parse", "origin/main")
    predecessor, earlier = select_predecessor(event=event, head=head, base=base)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "storage-format-predecessor-sha").write_text(predecessor + "\n")
    _git("archive", "--format=tar", predecessor, "-o", str(output / "predecessor.tar"))
    with tempfile.TemporaryDirectory(prefix="storage-format-lock-") as lock_dir:
        lock_path = Path(lock_dir) / "uv.lock"
        _copy_predecessor_lock(predecessor, lock_path)
        shutil.copy2(lock_path, output / "uv.lock")
    candidate = output / "storage-format-candidate.json"
    result = _run_phase(
        ("uv", "run", "--frozen", "python", "tests/compatibility/storage_probe.py"),
        phase="candidate registry probe",
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"candidate probe failed: {result.stderr or result.stdout}")
    try:
        candidate.write_text(json.dumps(json.loads(result.stdout), sort_keys=True, separators=(",", ":")) + "\n")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"candidate probe emitted malformed json: {exc}") from exc
    bootstrap = not has_probe(predecessor)
    if bootstrap and any(has_probe(commit) for commit in earlier[1:] if qualifying(commit)):
        raise RuntimeError(f"selected predecessor {predecessor} is missing the compatibility probe")
    with tempfile.TemporaryDirectory(prefix="storage-format-predecessor-") as temporary:
        source = Path(temporary)
        _run_phase(
            ("tar", "xf", str(output / "predecessor.tar"), "-C", str(source)),
            phase="predecessor archive extraction",
            timeout=_GIT_TIMEOUT,
            check=True,
        )
        if bootstrap:
            predecessor_epoch = _writer_epoch(source)
            print(f"::notice::bootstrap predecessor {predecessor}: registry comparison starts with this predecessor")
        else:
            _probe_source(source, output / "storage-format-predecessor.json")
            predecessor_registry = json.loads((output / "storage-format-predecessor.json").read_text())
            compare_registry_dumps(predecessor_registry, json.loads(candidate.read_text()))
            predecessor_epoch = predecessor_registry["writer_epoch"]
        _run_roundtrips(source, predecessor_epoch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StorageFormatError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
