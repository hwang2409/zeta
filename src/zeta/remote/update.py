"""Update a remote editable zeta checkout over ssh."""
from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path


def _remote_script(target: str, host: str) -> str:
    qtarget = shlex.quote(target)
    qhost = shlex.quote(host)
    return f"""set -eu
uv=\"$HOME/.local/bin/uv\"
[ -x \"$uv\" ] || uv=uv
if ! tool_dir=\"$(\"$uv\" tool dir 2>/dev/null)\"; then
  echo 'remote zeta tool is not installed' >&2; exit 2
fi
python=\"$tool_dir/zeta/bin/python\"
if [ ! -x \"$python\" ]; then echo 'remote zeta tool python was not found' >&2; exit 2; fi
path=\"$(\"$python\" -c 'import os,zeta; print(os.path.dirname(zeta.__file__))')\"
if ! path=\"$(git -C \"$path\" rev-parse --show-toplevel)\"; then
  echo \"remote zeta path is not a git checkout: $path\" >&2; exit 2
fi
dirty=\"$(git -C \"$path\" status --porcelain --untracked-files=no)\"
if [ -n \"$dirty\" ]; then
  printf 'remote checkout is not clean:\\n%s\\n' \"$dirty\" >&2
  exit 3
fi
old=\"$(git -C \"$path\" rev-parse HEAD)\"
if [ \"$old\" = {qtarget} ]; then echo \"already up to date ($old)\"; exit 0; fi
git -C \"$path\" fetch origin
if ! git -C \"$path\" cat-file -e {qtarget}^{{commit}} 2>/dev/null; then
  echo \"target {qtarget} is not reachable after fetch\" >&2; exit 4
fi
if ! git -C \"$path\" merge-base --is-ancestor \"$old\" {qtarget}; then
  echo \"refusing backwards update: remote HEAD $old is not an ancestor of {qtarget}\" >&2
  exit 5
fi
reinstall=no
if ! git -C \"$path\" diff --quiet \"$old\" {qtarget} -- pyproject.toml uv.lock; then reinstall=yes; fi
git -C \"$path\" checkout --detach {qtarget}
if [ \"$reinstall\" = yes ]; then
  \"$uv\" tool install --force --editable \"$path\"
  reinstall_note='reinstalled the editable zeta tool'
else
  reinstall_note='no reinstall needed'
fi
undo=\"$(\"$python\" -c 'import shlex,sys; print(f\"git -C {{shlex.quote(sys.argv[1])}} checkout --detach {{sys.argv[2]}}\")' \"$path\" \"$old\")\"
undo_q=\"$(\"$python\" -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' \"$undo\")\"
printf 'updated remote zeta\\npath: %s\\nold sha: %s\\nnew sha: %s\\n%s\\nundo: ssh -- %s %s\\nrestart: running zeta sessions there must /restart\\n' \"$path\" \"$old\" {qtarget} \"$reinstall_note\" {qhost} \"$undo_q\"
"""


def _package_dir():
    import zeta

    return Path(zeta.__file__).resolve().parent


def _target() -> tuple[str, str]:
    package = _package_dir()
    try:
        root = subprocess.check_output(
            ["git", "-C", package, "rev-parse", "--show-toplevel"], text=True
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "origin/main"], text=True
        ).strip()
        subject = subprocess.check_output(
            ["git", "-C", root, "show", "-s", "--format=%s", sha], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(detail or "could not resolve local origin/main") from None
    return sha, subject


def update_remote(
    host: str, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> str:
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("invalid HOST")
    target, subject = _target()
    command = _remote_script(target, host)
    result = runner(
        ["ssh", "--", host, f"sh -c {shlex.quote(command)}"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = f"target sha: {target} ({subject})\n{result.stdout}"
    if result.returncode:
        detail = result.stderr.strip()
        if detail:
            output += detail + "\n"
        raise RuntimeError(output.rstrip())
    return output.rstrip()
