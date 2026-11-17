"""ExtensionStore trust, publication, recovery, and concurrency behavior."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
from pathlib import Path

import pytest

from zeta.extensions.store import ExtensionStore, ExtensionTrustError


def _extension(path: Path, name: str = "demo", marker: str = "old") -> Path:
    path.mkdir(parents=True)
    (path / "zeta-extension.toml").write_text(
        f'[extension]\nname = "{name}"\nversion = "1"\napi_version = 1\nentry = "mod:activate"\n',
        encoding="utf-8",
    )
    (path / "mod.py").write_text(f"# {marker}\ndef activate(zx):\n    pass\n", encoding="utf-8")
    return path


def _git_extension(path: Path, marker: str = "old") -> tuple[Path, str]:
    _extension(path, marker=marker)
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", marker],
        check=True,
    )
    commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return path, commit


def _trust(home: Path) -> dict[str, str]:
    path = home / "extensions-trust.json"
    return json.loads(path.read_text()) if path.exists() else {}


class _Stop(RuntimeError):
    pass


class _CheckpointStore(ExtensionStore):
    stop: str | None = None

    def _checkpoint(self, name: str) -> None:
        if name == self.stop:
            raise _Stop(name)


@pytest.mark.parametrize(
    ("case", "action", "confirmation", "checkpoint", "error", "installed", "trusted"),
    [
        ("install-success", "install", "y", None, None, True, True),
        ("install-cancel", "install", "n", None, "install cancelled", False, False),
        ("install-stop-after-target", "install", "y", "install_target_published", "install_target_published", True, False),
        ("trust-success", "trust", "y", None, None, True, True),
        ("trust-cancel", "trust", "n", None, "trust cancelled", True, False),
        ("untrust-success", "untrust", "y", None, None, True, False),
        ("remove-success", "remove", "y", None, None, False, False),
        ("remove-idempotent", "remove-absent", "y", None, None, False, False),
    ],
)
def test_public_action_outcomes(
    tmp_path: Path,
    case: str,
    action: str,
    confirmation: str,
    checkpoint: str | None,
    error: str | None,
    installed: bool,
    trusted: bool,
) -> None:
    home = tmp_path / case
    source = _extension(tmp_path / f"source-{case}")
    store = _CheckpointStore(home=home, project_dir=None)
    if action not in {"install", "remove-absent"}:
        store.install(str(source), scope="user", accept=True)
        store.untrust("demo", scope="user")
    store.stop = checkpoint
    call = {
        "install": lambda: store.install(str(source), scope="user", confirm=lambda _: confirmation),
        "trust": lambda: store.trust("demo", scope="user", confirm=lambda _: confirmation),
        "untrust": lambda: store.untrust("demo", scope="user"),
        "remove": lambda: store.remove("demo", scope="user"),
        "remove-absent": lambda: store.remove("demo", scope="user"),
    }[action]
    if error:
        expected = _Stop if checkpoint else ExtensionTrustError
        with pytest.raises(expected, match=error):
            call()
    else:
        call()
    target = home / "extensions" / "demo"
    assert target.exists() is installed
    values = _trust(home)
    assert ("user:demo" in values) is trusted
    result = ExtensionStore(home=home, project_dir=None).discover("user")
    assert bool(result.extensions) is installed
    if result.extensions:
        assert result.extensions[0].trusted is trusted


def test_install_uses_reviewed_stage_without_reopening_source(tmp_path: Path) -> None:
    source = _extension(tmp_path / "source", marker="reviewed")
    home = tmp_path / "home"

    def confirm(_: str) -> str:
        (source / "mod.py").write_text("raise AssertionError('source reopened')\n", encoding="utf-8")
        return "y"

    result = ExtensionStore(home=home, project_dir=None).install(str(source), scope="user", confirm=confirm)
    target = home / "extensions" / "demo"
    assert "reviewed" in (target / "mod.py").read_text()
    assert _trust(home)["user:demo"] == result.digest


def test_install_pending_retry_requires_identical_fresh_review(tmp_path: Path) -> None:
    source = _extension(tmp_path / "source")
    home = tmp_path / "home"
    broken = _CheckpointStore(home=home, project_dir=None)
    broken.stop = "install_target_published"
    with pytest.raises(_Stop):
        broken.install(str(source), scope="user", accept=True)
    store = ExtensionStore(home=home, project_dir=tmp_path / "project")
    pending = ExtensionStore(home=home, project_dir=None).discover("user")
    assert pending.extensions and not pending.extensions[0].trusted
    result = store.install(str(source), scope="user", accept=True)
    assert _trust(home)["user:demo"] == result.digest
    (source / "mod.py").write_text("# conflict\n", encoding="utf-8")
    with pytest.raises(ExtensionTrustError, match="extension already installed with different content: demo"):
        store.install(str(source), scope="user", accept=True)
    assert _trust(home)["user:demo"] == result.digest


@pytest.mark.parametrize(
    "checkpoint",
    [
        "update_owner_published",
        "update_backup_published",
        "update_target_published",
        "update_trust_published",
        "update_backup_deleted",
        "update_owner_deleted",
    ],
)
def test_update_recovery_converges_at_every_checkpoint(tmp_path: Path, checkpoint: str) -> None:
    repository, _ = _git_extension(tmp_path / "repo")
    source = f"git+{repository}@HEAD"
    home = tmp_path / checkpoint
    store = _CheckpointStore(home=home, project_dir=None)
    old = store.install(source, scope="user", accept=True)
    (repository / "mod.py").write_text("# new\ndef activate(zx):\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", "new"], check=True)
    store.stop = checkpoint
    with pytest.raises(_Stop):
        store.update("demo", scope="user", accept=True)
    recovered = ExtensionStore(home=home, project_dir=None).discover("user")
    assert len(recovered.extensions) == 1
    item = recovered.extensions[0]
    assert item.trusted
    if checkpoint in {"update_owner_published", "update_backup_published", "update_target_published"}:
        assert item.digest == old.digest
    else:
        assert item.digest != old.digest
    assert not list((home / "extensions" / ".zeta-extension-store").glob("update-*"))


def test_malformed_update_suppresses_only_affected_scope(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    store = ExtensionStore(home=home, project_dir=project)
    store.install(str(_extension(tmp_path / "user-source", name="user-good")), scope="user", accept=True)
    store.install(str(_extension(tmp_path / "project-source", name="project-good")), scope="project", accept=True)
    bad = home / "extensions" / ".zeta-extension-store" / "update-bad"
    bad.mkdir()
    (bad / "owner.json").write_text("{}")
    before = sorted(str(path.relative_to(home)) for path in home.rglob("*"))
    user = store.discover("user")
    project_result = store.discover("project")
    assert user.extensions == ()
    assert user.notices == ("extension loading disabled for user scope: update recovery requires attention",)
    assert [item.name for item in project_result.extensions] == ["project-good"]
    assert sorted(str(path.relative_to(home)) for path in home.rglob("*")) == before


def _writer(home: str, source: str, barrier: multiprocessing.Barrier) -> None:
    store = ExtensionStore(home=Path(home), project_dir=None)
    barrier.wait(10)
    store.install(source, scope="user", accept=True)


class _ExitStore(ExtensionStore):
    stop = ""

    def _checkpoint(self, name: str) -> None:
        if name == self.stop:
            os._exit(91)


def _crash_update(home: str, checkpoint: str) -> None:
    store = _ExitStore(home=Path(home), project_dir=None)
    store.stop = checkpoint
    store.update("demo", scope="user", accept=True)


def test_two_process_writers_preserve_both_trust_keys(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = _extension(tmp_path / "first", name="first")
    second = _extension(tmp_path / "second", name="second")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    processes = [context.Process(target=_writer, args=(str(home), str(source), barrier)) for source in (first, second)]
    for process in processes:
        process.start()
    barrier.wait(10)
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
        process.close()
    assert set(_trust(home)) == {"user:first", "user:second"}


@pytest.mark.parametrize(
    "checkpoint",
    [
        "update_owner_published",
        "update_backup_published",
        "update_target_published",
        "update_trust_published",
        "update_backup_deleted",
        "update_owner_deleted",
    ],
)
def test_real_process_stop_recovers_update(tmp_path: Path, checkpoint: str) -> None:
    repository, _ = _git_extension(tmp_path / "repo")
    source = f"git+{repository}@HEAD"
    home = tmp_path / "home"
    old = ExtensionStore(home=home, project_dir=None).install(source, scope="user", accept=True)
    (repository / "mod.py").write_text("# new\ndef activate(zx):\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", "new"], check=True)
    process = multiprocessing.get_context("spawn").Process(target=_crash_update, args=(str(home), checkpoint))
    process.start()
    process.join(20)
    assert process.exitcode == 91
    process.close()
    result = ExtensionStore(home=home, project_dir=None).discover("user")
    assert len(result.extensions) == 1 and result.extensions[0].trusted
    expected_old = checkpoint in {"update_owner_published", "update_backup_published", "update_target_published"}
    assert (result.extensions[0].digest == old.digest) is expected_old
    assert not list((home / "extensions" / ".zeta-extension-store").glob("update-*"))


def test_git_metadata_is_reviewed_and_in_digest(tmp_path: Path) -> None:
    repository, commit = _git_extension(tmp_path / "repo")
    home = tmp_path / "home"
    seen: list[str] = []
    result = ExtensionStore(home=home, project_dir=None).install(
        f"git+{repository}@HEAD",
        scope="user",
        confirm=lambda prompt: seen.append(prompt) or "y",
    )
    metadata = json.loads((home / "extensions" / "demo" / ".zeta-extension-git.json").read_text())
    assert metadata == {"commit": commit, "source": f"git+{repository}@HEAD"}
    assert seen == ["install this extension? [y/N] "]
    assert _trust(home)["user:demo"] == result.digest


def test_malformed_trust_fails_closed_for_project_but_user_loads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _extension(home / "extensions" / "user", name="user")
    _extension(project / ".zeta" / "extensions" / "project", name="project")
    home.mkdir(exist_ok=True)
    (home / "extensions-trust.json").write_text('{"x":"y","x":"z"}')
    store = ExtensionStore(home=home, project_dir=project)
    user = store.discover("user")
    project_result = store.discover("project")
    assert [item.name for item in user.extensions] == ["user"]
    assert not user.extensions[0].trusted
    assert project_result.extensions and not project_result.extensions[0].trusted
    assert "invalid extension trust file" in user.notices[0]


def test_event_loop_call_is_rejected_before_lock(tmp_path: Path) -> None:
    import asyncio

    async def call() -> None:
        with pytest.raises(ExtensionTrustError, match="event-loop"):
            ExtensionStore(home=tmp_path / "home", project_dir=None).discover("user")

    asyncio.run(call())
