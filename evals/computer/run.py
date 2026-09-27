"""Run a Zeta turn with one tool inside an isolated Docker guest."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from zeta.core.store import ConversationStore
from zeta.mcp.config import MCPConfig, MCPServerConfig
from zeta.mcp.mount import mount_mcp_servers
from zeta.providers.factory import build_backend
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry

TASKS = Path(__file__).resolve().parents[1] / "tasks.jsonl"
BROWSER_TASKS = Path(__file__).with_name("browser_tasks.jsonl")
IMAGE = "zeta-computer-eval:local"
BROWSER_IMAGE = "zeta-computer-browser-eval:local"
BROWSER_SECCOMP = Path(__file__).resolve().with_name("seccomp_profile.json")
MAX_FILE_BYTES = 2_000_000


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or str(path) != name or ".." in path.parts:
        raise ValueError(f"unsafe computer file name: {name!r}")


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in files.items():
            _safe_name(name)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError(f"computer file too large: {name}")
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            member.uid = member.gid = 65532
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _unarchive(payload: bytes, names: tuple[str, ...]) -> dict[str, bytes]:
    expected = set(names)
    for name in names:
        _safe_name(name)
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive:
            if (
                member.name not in expected
                or not member.isfile()
                or member.size > MAX_FILE_BYTES
            ):
                raise ValueError(f"unexpected computer artifact: {member.name!r}")
            if member.name in result:
                raise ValueError(f"duplicate computer artifact: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"missing computer artifact: {member.name}")
            result[member.name] = source.read(MAX_FILE_BYTES + 1)
    if set(result) != expected:
        raise ValueError(
            f"missing computer artifacts: {sorted(expected - set(result))}"
        )
    return result


def _docker(
    binary: str,
    context: str,
    *args: str,
    payload: bytes | None = None,
    timeout: int = 120,
) -> bytes:
    result = subprocess.run(
        [binary, "--context", context, *args],
        input=payload,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"docker {args[0]} failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')[-1000:]}"
        )
    return result.stdout


def _container_args(name: str, image: str = IMAGE) -> tuple[str, ...]:
    browser = image == BROWSER_IMAGE
    # ponytail: bash can disable Chromium's own sandbox; keep this networkless
    # until a restricted browser action path enforces sandboxed launches.
    return (
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "1g" if browser else "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "256" if browser else "64",
        "--user",
        "65532:65532",
        "--tmpfs",
        "/workspace:rw,nosuid,nodev,size=128m,mode=1777",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={'128m' if browser else '32m'},mode=1777",
        *(
            (
                "--init",
                "--shm-size",
                "256m",
                "--security-opt",
                f"seccomp={BROWSER_SECCOMP}",
            )
            if browser
            else ()
        ),
        image,
    )


@contextmanager
def _container(binary: str, context: str, image: str = IMAGE):
    name = f"zeta-computer-eval-{uuid.uuid4().hex[:12]}"
    _docker(binary, context, *_container_args(name, image))
    try:
        details = json.loads(_docker(binary, context, "inspect", name))[0]
        host = details["HostConfig"]
        if (
            details["Mounts"] != []
            or host["NetworkMode"] != "none"
            or host["ReadonlyRootfs"] is not True
            or details["Config"]["User"] != "65532:65532"
            or host["CapDrop"] != ["ALL"]
        ):
            raise RuntimeError("computer container failed isolation check")
        if image == BROWSER_IMAGE:
            profile = json.loads(BROWSER_SECCOMP.read_text())
            options = host.get("SecurityOpt") or []
            if not any(
                json.loads(option.removeprefix("seccomp=")) == profile
                for option in options
                if option.startswith("seccomp={")
            ):
                raise RuntimeError("computer browser seccomp profile was not applied")
        yield name
    finally:
        try:
            _docker(binary, context, "stop", "--time", "1", name, timeout=20)
        except RuntimeError as exc:
            print(f"computer cleanup warning: {exc}", file=sys.stderr)


def _seed(binary: str, context: str, container: str, files: dict[str, bytes]) -> None:
    _docker(
        binary,
        context,
        "exec",
        "-i",
        container,
        "tar",
        "-C",
        "/workspace",
        "-xf",
        "-",
        payload=_archive(files),
    )


def _export(
    binary: str, context: str, container: str, names: tuple[str, ...]
) -> dict[str, bytes]:
    payload = _docker(
        binary,
        context,
        "exec",
        container,
        "tar",
        "-C",
        "/workspace",
        "-cf",
        "-",
        *names,
    )
    return _unarchive(payload, names)


async def _agent(
    binary: str,
    context: str,
    container: str,
    task: dict[str, Any],
    provider: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    browser_only = task.get("mode") == "browser"
    tool_name = "computer__browser" if browser_only else "computer__bash"
    with tempfile.TemporaryDirectory(prefix="zeta-computer-host-") as host_dir:
        previous_runtime_dir = os.environ.get("WIKI_AGENT_RUNTIME_DIR")
        os.environ["WIKI_AGENT_RUNTIME_DIR"] = host_dir
        skills = SkillCatalog.empty()
        store = ConversationStore(host_dir)
        registry = ToolRegistry(host_dir, register_builtin=False, skill_catalog=skills)
        server = MCPServerConfig(
            name="computer",
            transport="stdio",
            command=binary,
            args=(
                "--context",
                context,
                "exec",
                "-i",
                container,
                "python3",
                "/opt/zeta/guest.py",
                *(("--browser",) if browser_only else ()),
            ),
        )
        config = MCPConfig(
            path=Path(host_dir) / "mcp.json", servers={"computer": server}
        )
        loop: AgentLoop | None = None
        mount = None
        try:
            mount = await mount_mcp_servers(registry, config, home=host_dir)
            names = [schema["name"] for schema in registry.schemas]
            if names != [tool_name]:
                raise RuntimeError(f"expected only guest {tool_name}, got {names}")
            backend, _ = build_backend(provider, model)
            loop = AgentLoop(
                backend,
                store,
                registry=registry,
                skill_catalog=skills,
                skip_mcp_mount=True,
                max_turns=task.get("max_turns", 12),
                system_prompt=(
                    "You have only the computer__browser tool. It controls sandboxed "
                    "Chromium over a local file:///workspace/ page. There is no shell, "
                    "public network, host files, or credentials. Verify the final page "
                    "state from its accessible snapshot before finishing."
                    if browser_only
                    else "You have only the computer__bash tool. It runs in an isolated "
                    "Linux /workspace; no host files or credentials are available. "
                    "Verify changes in that workspace before finishing."
                ),
            )
            loop.attach_mcp_mount(mount)
            calls: dict[str, str] = {}
            errors: list[str] = []
            turns = 0
            completed = False
            last_result: str | None = None
            last_result_error = False

            async def drive() -> None:
                nonlocal turns, completed, last_result, last_result_error
                async for event in loop.run_turn(task["prompt"]):
                    if event.tool_call is not None:
                        calls[event.tool_call.id] = event.tool_call.name
                    if (
                        event.type.value == "tool_execution_end"
                        and event.tool_result is not None
                    ):
                        last_result = event.tool_result.content
                        last_result_error = event.tool_result.is_error
                    if event.error is not None:
                        errors.append(event.error.message)
                    if event.type.value == "turn_end":
                        turns += 1
                        if (
                            event.data.get("tool_calls") == 0
                            and event.message is not None
                        ):
                            completed = True

            await asyncio.wait_for(drive(), timeout)
            if not calls or set(calls.values()) != {tool_name}:
                raise RuntimeError(
                    f"agent used unexpected tools: {list(calls.values())}"
                )
            return {
                "tool_calls": len(calls),
                "turns": turns,
                "completed": completed,
                "errors": errors,
                "last_result": last_result,
                "last_result_error": last_result_error,
            }
        finally:
            try:
                if loop is not None:
                    await loop.close()
                else:
                    if mount is not None:
                        await mount.close()
                    await registry.close()
            finally:
                store.close()
                if previous_runtime_dir is None:
                    os.environ.pop("WIKI_AGENT_RUNTIME_DIR", None)
                else:
                    os.environ["WIKI_AGENT_RUNTIME_DIR"] = previous_runtime_dir


def _task(task_id: str) -> dict[str, Any]:
    for source in (TASKS, BROWSER_TASKS):
        for line in source.read_text().splitlines():
            task = json.loads(line)
            if task["id"] == task_id:
                return task
    raise ValueError(f"unknown computer task: {task_id}")


def _verify(
    binary: str,
    context: str,
    container: str,
    task: dict[str, Any],
    agent: dict[str, Any],
) -> dict[str, bytes]:
    task_id = task["id"]
    setup = {name: content.encode() for name, content in task.get("setup", {}).items()}
    if task.get("mode") == "browser":
        result = agent["last_result"]
        if agent["last_result_error"] or not isinstance(result, str):
            raise ValueError("browser ended without a successful observation")
        for check in task["checks"]:
            if "contains" in check and check["contains"] not in result:
                raise ValueError(f"browser observation missing {check['contains']!r}")
            if "not_contains" in check and check["not_contains"] in result:
                raise ValueError(
                    f"browser observation contains {check['not_contains']!r}"
                )
        artifacts = _export(binary, context, container, tuple(setup))
        if artifacts != setup:
            raise ValueError("agent changed browser-only fixture")
        return artifacts
    if task_id == "count-and-write":
        artifacts = _export(binary, context, container, ("input.txt", "count.txt"))
        expected = next(
            check["equals"]
            for check in task["checks"]
            if check.get("path") == "count.txt"
        )
        if (
            artifacts["input.txt"] != setup["input.txt"]
            or artifacts["count.txt"] != expected.encode()
        ):
            raise ValueError("count task artifacts differed")
        return artifacts
    if task_id in {"csv-parser-repair", "browser-todo-repair"}:
        source, test = (
            ("index.html", "test_browser_todo.py")
            if task_id == "browser-todo-repair"
            else ("csv_summary.py", "test_csv_summary.py")
        )
        artifacts = _export(binary, context, container, (source, test))
        if artifacts[test] != setup[test]:
            raise ValueError(f"agent changed the {task_id} regression test")
        image = BROWSER_IMAGE if task_id == "browser-todo-repair" else IMAGE
        with _container(binary, context, image) as verifier:
            _seed(
                binary,
                context,
                verifier,
                {source: artifacts[source], test: setup[test]},
            )
            _docker(
                binary,
                context,
                "exec",
                "-w",
                "/workspace",
                verifier,
                "python3",
                "-m",
                "unittest",
                "-q",
                test,
            )
        return artifacts
    raise ValueError(f"unsupported computer task: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument(
        "--task",
        choices=(
            "count-and-write",
            "csv-parser-repair",
            "browser-todo-repair",
            "browser-issue-triage",
        ),
        required=True,
    )
    parser.add_argument("--provider", choices=("codex", "claude"), default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-artifacts", type=Path)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    binary = shutil.which("docker")
    if binary is None:
        parser.error("docker is required")
    task = _task(args.task)
    image = (
        BROWSER_IMAGE
        if args.task in {"browser-todo-repair", "browser-issue-triage"}
        else IMAGE
    )
    started = time.monotonic()
    try:
        context_dir = Path(__file__).resolve().parent
        build_args = (
            ("-f", str(context_dir / "Dockerfile.browser"))
            if image == BROWSER_IMAGE
            else ()
        )
        _docker(
            binary,
            args.docker_context,
            "build",
            *build_args,
            "-t",
            image,
            str(context_dir),
            timeout=300,
        )
        with _container(binary, args.docker_context, image) as container:
            _seed(
                binary,
                args.docker_context,
                container,
                {
                    name: content.encode()
                    for name, content in task.get("setup", {}).items()
                },
            )
            agent = asyncio.run(
                _agent(
                    binary,
                    args.docker_context,
                    container,
                    task,
                    args.provider,
                    args.model,
                    args.timeout,
                )
            )
            artifacts = _verify(binary, args.docker_context, container, task, agent)
            if agent["errors"] or not agent["completed"]:
                raise RuntimeError(f"agent did not complete cleanly: {agent['errors']}")
        artifact_path = None
        if args.keep_artifacts is not None:
            args.keep_artifacts.mkdir(parents=True, exist_ok=True)
            root = Path(
                tempfile.mkdtemp(prefix="zeta-computer-", dir=args.keep_artifacts)
            )
            for name, content in artifacts.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            artifact_path = str(root)
        agent.pop("last_result")
        agent.pop("last_result_error")
        print(
            json.dumps(
                {
                    "task": args.task,
                    "provider": args.provider,
                    "model": args.model,
                    "image": image,
                    "passed": True,
                    **agent,
                    "seconds": round(time.monotonic() - started, 2),
                    "artifacts": artifact_path,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps(
                {
                    "task": args.task,
                    "provider": args.provider,
                    "model": args.model,
                    "image": image,
                    "passed": False,
                    "seconds": round(time.monotonic() - started, 2),
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
