from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def repo_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet", "-b", "main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "test")
    (source / "pyproject.toml").write_text("[project]\nname='zeta'\n")
    (source / "uv.lock").write_text("base\n")
    (source / "src/zeta").mkdir(parents=True)
    (source / "src/zeta/__init__.py").write_text("")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "base")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(source), str(origin)], check=True)
    remote = tmp_path / "remote checkout with 'quote'"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(remote)], check=True)
    (source / "README").write_text("new\n")
    (source / "uv.lock").write_text("new lock\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "target")
    target = git(source, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "--git-dir", str(origin), "fetch", "--quiet", str(source), "main"],
        check=True,
    )
    subprocess.run(
        ["git", "--git-dir", str(origin), "update-ref", "refs/heads/main", target],
        check=True,
    )
    old = git(remote, "rev-parse", "HEAD")
    return origin, remote, old, target


def fake_host(tmp_path: Path, remote: Path, *, fail_uv: bool = False) -> Path:
    home = tmp_path / "remote home"
    uv = home / ".local/bin/uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    tool_dir = home / ".local/share/uv/tools/zeta/bin"
    tool_dir.mkdir(parents=True, exist_ok=True)
    if not tool_dir.joinpath("python").exists():
        tool_dir.joinpath("python").symlink_to(os.sys.executable)
    uv.write_text(
        "#!/bin/sh\n"
        + ("exit 77\n" if fail_uv else "")
        + "if [ \"$1 $2\" = 'tool dir' ] && [ $# -eq 2 ]; then\n"
        "  printf '%s\\n' \"$HOME/.local/share/uv/tools\"\n"
        "elif [ \"$1 $2\" = 'tool install' ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$HOME/uv-install.log\"\n"
        "else exit 1; fi\n"
    )
    uv.chmod(0o755)
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = -- ] || exit 91\n"
        "[ \"$2\" = fake-host ] || exit 92\n"
        f"HOME={shlex_quote(str(home))} PYTHONPATH={shlex_quote(str(remote / 'src'))} sh -c \"$3\"\n"
    )
    ssh.chmod(0o755)
    return home


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def run_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote: Path, target: str, *, fail_uv: bool = False) -> tuple[int, str, str]:
    from zeta.cli.main import main

    try:
        from zeta.remote import update
    except ImportError:
        update = None
    if update is not None:
        # test-contract: seam target revision is supplied by the local checkout in production.
        monkeypatch.setattr(update, "_target", lambda: (target, "target subject"))
    home = fake_host(tmp_path, remote, fail_uv=fail_uv)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["HOME"] = str(home)
    old_path = os.environ.copy()
    os.environ.update(env)
    try:
        with _capture() as captured:
            try:
                code = main(["remote", "update", "fake-host"])
            except SystemExit as exc:
                code = int(exc.code)
    finally:
        os.environ.clear()
        os.environ.update(old_path)
    return code, captured.out.getvalue(), captured.err.getvalue()


class _capture:
    def __enter__(self):
        import contextlib
        import io

        self.out, self.err = io.StringIO(), io.StringIO()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(self.out))
        self.stack.enter_context(contextlib.redirect_stderr(self.err))
        return self

    def __exit__(self, *args):
        self.stack.close()
        return False


def test_clean_checkout_moves_to_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    code, out, err = run_cli(monkeypatch, tmp_path, remote, target)
    assert code == 0, err
    assert "updated remote zeta" in out and target in out and old in out
    assert git(remote, "rev-parse", "HEAD") == target
    assert "reinstalled the editable zeta tool" in out
    assert "git -C '" in out
    assert (tmp_path / "remote home/uv-install.log").read_text().startswith(
        "tool install --force --editable"
    )


def test_dirty_checkout_is_unchanged_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    (remote / "pyproject.toml").write_text("dirty\n")
    code, out, err = run_cli(monkeypatch, tmp_path, remote, target)
    assert code == 2 and "not clean" in err and "updated remote zeta" not in out
    assert git(remote, "rev-parse", "HEAD") == old


def test_backwards_update_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    assert run_cli(monkeypatch, tmp_path, remote, target)[0] == 0
    code, out, err = run_cli(monkeypatch, tmp_path, remote, old)
    assert code == 2 and "backwards" in err and "updated remote zeta" not in out
    assert git(remote, "rev-parse", "HEAD") == target


def test_up_to_date_exits_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, _, target = repo_pair(tmp_path)
    git(remote, "fetch", "origin")
    git(remote, "checkout", "--detach", target)
    code, out, err = run_cli(monkeypatch, tmp_path, remote, target)
    assert code == 0 and "already up to date" in out and not err


def test_target_unreachable_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    other = tmp_path / "other.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(remote), str(other)], check=True)
    git(remote, "remote", "set-url", "origin", str(other))
    code, out, err = run_cli(monkeypatch, tmp_path, remote, target)
    assert code == 2 and "not reachable" in err and "updated remote zeta" not in out
    assert git(remote, "rev-parse", "HEAD") == old


def test_failed_remote_step_has_no_success_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    code, out, err = run_cli(monkeypatch, tmp_path, remote, target, fail_uv=True)
    assert code == 2 and "updated remote zeta" not in out and "remote zeta tool" in err
    assert git(remote, "rev-parse", "HEAD") == old


def test_host_starting_with_dash_is_rejected() -> None:
    try:
        from zeta.remote import update
    except ImportError:
        from zeta.cli.main import main

        with pytest.raises(SystemExit):
            main(["remote", "update", "--", "-oProxyCommand=x"])
    else:
        with pytest.raises(ValueError, match="invalid HOST"):
            update.update_remote("-oProxyCommand=x")
