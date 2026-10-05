"""``zeta computer``: sandbox VM lifecycle and spectating."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

from ..core.session import env_home
from .docker import DockerError
from .lima import IsolationError, SandboxError, SandboxVM
from .local import (
    CONTAINER_LABEL,
    IMAGE_SIZE_NOTE,
    SESSION_LABEL,
    build_image,
    image_present,
    image_tag,
    verified_docker,
)
from .session import RECORDING_DIRECTORY
from .settings import ComputerSettingsError, load_computer_settings
from .spectate import Spectator, run_live

GIB = 1024**3
# ``watch`` output is read by people and by scripts through a pipe.
say = functools.partial(print, flush=True)


def add_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("computer", help="manage the computer-use sandbox")
    actions = parser.add_subparsers(dest="computer_action", required=True)
    actions.add_parser("setup", help="create or start the sandbox VM, verify it, build the image")
    actions.add_parser("status", help="show the VM, isolation checks, image, and desktops")
    actions.add_parser("stop", help="remove all desktops and stop the sandbox VM")
    destroy = actions.add_parser("destroy", help="delete the sandbox VM and its images")
    destroy.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    watch = actions.add_parser("watch", help="open the spectator page for a session")
    watch.add_argument("session", nargs="?", help="session ID or prefix (default: newest)")
    watch.add_argument("--live", action="store_true", help="also open a live VNC view")


def _setup(vm: SandboxVM) -> int:
    settings = load_computer_settings(env_home())
    info = vm.info()
    if info is None:
        print(
            f"creating VM {vm.name}: --mount-none, {settings.cpus} CPUs, "
            f"{settings.memory_gib} GiB memory, {settings.disk_gib} GiB disk"
        )
        vm.create(cpus=settings.cpus, memory_gib=settings.memory_gib, disk_gib=settings.disk_gib)
        info = vm.info()
    elif (info.cpus, info.memory_bytes) != (settings.cpus, settings.memory_gib * GIB):
        print(
            f"note: VM {vm.name} has {info.cpus} CPUs and {info.memory_bytes / GIB:g} GiB "
            "memory; settings apply only when setup creates the VM"
        )
    if info is None or info.status != "Running":
        print(f"starting VM {vm.name}")
        vm.start()
    info = vm.require_running()
    with verified_docker(vm) as docker:
        tag = image_tag()
        if image_present(docker, tag):
            print(f"image {tag} is ready")
        else:
            print(f"building image {tag} ({IMAGE_SIZE_NOTE}); this takes a few minutes")
            build_image(docker)
        size = docker.output("image", "inspect", "--format", "{{.Size}}", tag, timeout=30)
        print(f"image {tag}: {int(size.decode().strip()) / GIB:.2f} GiB")
    print("ready: run `zeta --computer`")
    return 0


def _status(vm: SandboxVM) -> int:
    info = vm.info()
    if info is None:
        print(f"VM {vm.name}: absent (run `zeta computer setup`)")
        return 1
    print(
        f"VM {vm.name}: {info.status}, {info.cpus} CPUs, {info.memory_bytes / GIB:g} GiB memory, "
        f"{info.disk_bytes / GIB:g} GiB disk"
    )
    if info.status != "Running":
        return 0
    with verified_docker(vm) as docker:
        tag = image_tag()
        print(f"image {tag}: {'ready' if image_present(docker, tag) else 'missing'}")
        listing = docker.output(
            "ps", "--filter", f"label={CONTAINER_LABEL}=true",
            "--format", f'{{{{.ID}}}} session={{{{.Label "{SESSION_LABEL}"}}}} {{{{.Status}}}}',
            timeout=30,
        ).decode().strip()
        print("desktops: " + (listing.replace("\n", "; ") if listing else "none"))
    return 0


def _stop(vm: SandboxVM) -> int:
    info = vm.info()
    if info is None or info.status != "Running":
        print(f"VM {vm.name}: {'absent' if info is None else info.status}")
        return 0
    with verified_docker(vm) as docker:
        removed = docker.remove_labeled(f"{CONTAINER_LABEL}=true")
    print(f"removed {len(removed)} desktop(s)")
    vm.stop()
    print(f"VM {vm.name}: stopped")
    return 0


def _destroy(vm: SandboxVM, *, yes: bool) -> int:
    if vm.info() is None:
        print(f"VM {vm.name}: absent")
        return 0
    if not yes:
        answer = input(f"delete VM {vm.name} and its images? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("unchanged")
            return 1
    vm.delete()
    print(f"VM {vm.name}: deleted")
    return 0


def find_recording(sessions: Path, session: str | None) -> Path:
    """Return the recording of one session (by ID prefix) or the newest one."""

    candidates = [
        path.parent
        for path in sessions.glob(f"*/{RECORDING_DIRECTORY}/metadata.json")
        if session is None or path.parent.parent.name.startswith(session)
    ]
    if not candidates:
        raise SandboxError("no computer recording found" + (f" for {session}" if session else ""))
    if session is not None and len(candidates) > 1:
        raise SandboxError(f"session prefix {session} is ambiguous")
    return max(candidates, key=lambda path: (path / "metadata.json").stat().st_mtime)


def _watch(vm: SandboxVM, args: argparse.Namespace) -> int:
    recording = find_recording(env_home() / "sessions", args.session)
    session_id = recording.parent.name
    spectator = Spectator(recording)
    try:
        say(f"session {session_id}")
        say(f"spectator: {spectator.url}")
        if not args.live:
            say("press Ctrl-C to stop")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                return 0
        with verified_docker(vm) as docker:
            running = docker.running(f"{SESSION_LABEL}={session_id}")
            if len(running) != 1:
                raise SandboxError(
                    f"session {session_id} has {len(running)} running desktops; "
                    "the desktop starts on the first computer action"
                )
            run_live(docker, running[0], announce=say)
        return 0
    finally:
        spectator.stop()


def run(args: argparse.Namespace) -> int:
    vm = SandboxVM()
    try:
        action = args.computer_action
        if action == "setup":
            return _setup(vm)
        if action == "status":
            return _status(vm)
        if action == "stop":
            return _stop(vm)
        if action == "destroy":
            return _destroy(vm, yes=args.yes)
        return _watch(vm, args)
    except IsolationError as exc:
        print(f"zeta computer: isolation check FAILED, refusing to use the VM: {exc}", file=sys.stderr)
        return 1
    except (SandboxError, DockerError, ComputerSettingsError, RuntimeError, OSError) as exc:
        print(f"zeta computer: {exc}", file=sys.stderr)
        return 1


__all__ = ["add_parser", "find_recording", "run"]
