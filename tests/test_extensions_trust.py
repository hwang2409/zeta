"""Trust and installation gates for extensions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import zeta.extensions.trust as trust_module
from zeta.extensions import apply_extensions
from zeta.extensions.trust import (
    ExtensionTrustError,
    fingerprint,
    install,
    is_trusted,
    set_trusted,
    trust,
    update,
)
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def _extension(path: Path, name: str = "demo") -> Path:
    path.mkdir(parents=True)
    (path / "zeta-extension.toml").write_text(
        f'[extension]\nname = "{name}"\nversion = "1"\napi_version = 1\nentry = "mod:activate"\n',
        encoding="utf-8",
    )
    (path / "mod.py").write_text("def activate(zx):\n    pass\n", encoding="utf-8")
    return path


def test_fingerprint_match_and_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    extension = _extension(tmp_path / "extension")
    set_trusted(home, "project", "demo", extension, True)
    assert is_trusted(home, "project", "demo", extension)
    (extension / "mod.py").write_text("def activate(zx):\n    pass\n# changed\n", encoding="utf-8")
    assert not is_trusted(home, "project", "demo", extension)
    assert fingerprint(extension) != ""

@pytest.mark.asyncio
async def test_project_scope_trusted_fingerprint_allows_load(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    extension = _extension(project / ".zeta" / "extensions" / "demo")
    set_trusted(home, "project", "demo", extension, True)
    (tmp_path / "cwd").mkdir()
    registry = ToolRegistry(tmp_path / "cwd", skill_catalog=SkillCatalog.empty())
    try:
        host = apply_extensions(registry, home=home, project_dir=project, builtin_names=registry.registered_names)
        assert host.loaded and not host.pending_project
        assert host.loaded[0].name == "demo"
        assert is_trusted(home, "project", "demo", extension)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_project_scope_fingerprint_mismatch_blocks_load(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    extension = _extension(project / ".zeta" / "extensions" / "demo")
    set_trusted(home, "project", "demo", extension, True)
    (extension / "mod.py").write_text("def activate(zx):\n    raise AssertionError\n", encoding="utf-8")
    (tmp_path / "cwd").mkdir()
    registry = ToolRegistry(tmp_path / "cwd", skill_catalog=SkillCatalog.empty())
    try:
        host = apply_extensions(registry, home=home, project_dir=project, builtin_names=registry.registered_names)
        assert host.pending_project and not host.loaded
        assert "untrusted project extensions" in host.warnings[0]
    finally:
        await registry.close()


def test_management_name_cannot_escape_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    root = home / "extensions"
    _extension(root / "demo")
    outside = tmp_path / "outside"
    _extension(outside)
    with pytest.raises(ExtensionTrustError):
        trust(home=home, project_dir=None, scope="user", name="../outside", accept=True)
    assert outside.exists()
    with pytest.raises(ExtensionTrustError):
        trust(home=home, project_dir=None, scope="user", name=str(outside), accept=True)
    escape = root / "link"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ExtensionTrustError):
        trust(home=home, project_dir=None, scope="user", name="link", accept=True)


def test_fingerprint_rejects_symlink_and_size_bound(tmp_path: Path) -> None:
    extension = _extension(tmp_path / "extension")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (extension / "link").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ExtensionTrustError, match="symlink"):
        fingerprint(extension)
    (extension / "link").unlink()
    (extension / "large").write_bytes(b"x" * (64 * 1024 * 1024 + 1))
    with pytest.raises(ExtensionTrustError, match="byte limit"):
        fingerprint(extension)


def test_trust_requires_review(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _extension(home / "extensions" / "demo")
    with pytest.raises(ExtensionTrustError, match="cancelled"):
        trust(home=home, project_dir=None, scope="user", name="demo", confirm=lambda _: "n")
    assert not trust_module.trust_path(home).exists()


def test_git_install_requires_review_and_pins_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _extension(source)
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", "initial"], check=True)
    bare = tmp_path / "repo.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(source), str(bare)], check=True)
    home = tmp_path / "home"
    with pytest.raises(ExtensionTrustError, match="cancelled"):
        install(f"git+{bare}@HEAD", home=home, project_dir=None, scope="user", confirm=lambda _: "n")
    name, commit = install(f"git+{bare}@HEAD", home=home, project_dir=None, scope="user", accept=True)
    assert name == "demo" and len(commit) == 40
    assert is_trusted(home, "user", name, home / "extensions" / name)
    assert (home / "extensions" / name / ".zeta-extension-git.json").exists()
    with pytest.raises(ExtensionTrustError, match="cancelled"):
        update(name, home=home, project_dir=None, scope="user", confirm=lambda _: "n")
