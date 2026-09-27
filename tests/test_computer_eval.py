"""Keep the disposable-computer file boundary and launch flags honest."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from evals.computer.browser_guest import BrowserGuest, _workspace_url
from evals.computer.run import (
    BROWSER_IMAGE,
    BROWSER_SECCOMP,
    TASKS,
    _archive,
    _container_args,
    _task,
    _unarchive,
    _verify,
)


def test_computer_archive_only_round_trips_expected_regular_files() -> None:
    assert _unarchive(_archive({"input.txt": b"hello\n"}), ("input.txt",)) == {
        "input.txt": b"hello\n"
    }
    with pytest.raises(ValueError, match="unsafe"):
        _archive({"../host.txt": b"no"})
    with pytest.raises(ValueError, match="unexpected"):
        _unarchive(_archive({"other.txt": b"no"}), ("input.txt",))

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        link = tarfile.TarInfo("input.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../host.txt"
        archive.addfile(link)
    with pytest.raises(ValueError, match="unexpected"):
        _unarchive(payload.getvalue(), ("input.txt",))


def test_computer_has_no_network_or_host_mounts() -> None:
    for args in (_container_args("test-computer"), _container_args("test-computer", BROWSER_IMAGE)):
        assert args[:5] == ("run", "-d", "--rm", "--name", "test-computer")
        for pair in (
            ("--network", "none"),
            ("--cap-drop", "ALL"),
            ("--user", "65532:65532"),
        ):
            index = args.index(pair[0])
            assert args[index : index + 2] == pair
        assert "--read-only" in args
        assert not {"-v", "--volume", "--mount"}.intersection(args)

    browser = _container_args("test-computer", BROWSER_IMAGE)
    assert "--init" in browser
    assert browser[browser.index("--pids-limit") + 1] == "256"
    assert browser[browser.index("--shm-size") + 1] == "256m"
    assert browser[browser.index(f"seccomp={BROWSER_SECCOMP}") - 1] == "--security-opt"
    profile = json.loads(BROWSER_SECCOMP.read_text())
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert profile["syscalls"][0]["names"] == ["clone", "setns", "unshare"]
    assert next(rule for rule in profile["syscalls"] if rule["names"] == ["chroot"])["includes"] == {}


def test_browser_fixture_stays_out_of_default_workflow_evals() -> None:
    assert "browser-todo-repair" not in {
        json.loads(line)["id"] for line in TASKS.read_text().splitlines()
    }
    assert "chromium_sandbox=True" in _task("browser-todo-repair")["setup"]["test_browser_todo.py"]


def test_browser_guest_exposes_no_shell_and_rejects_public_url() -> None:
    script = Path(__file__).resolve().parents[1] / "evals/computer/guest.py"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "browser", "arguments": {"action": "open", "url": "https://example.com/"},
        }},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "bash", "arguments": {"command": "id"},
        }},
    ]
    result = subprocess.run(
        [sys.executable, str(script), "--browser"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True, text=True, timeout=10, check=True,
    )
    listed, blocked, no_shell = (json.loads(line) for line in result.stdout.splitlines())
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["browser"]
    assert blocked["result"]["isError"] is True
    assert "only file:///workspace/" in blocked["result"]["content"][0]["text"]
    assert "unknown tool" in no_shell["error"]["message"]
    with pytest.raises(ValueError, match="inside /workspace"):
        _workspace_url("file:///workspace/../../etc/passwd")
    with pytest.raises(ValueError, match="open a page"):
        BrowserGuest().call({"action": "fill", "role": "textbox", "value": ""})


def test_browser_eval_checks_observed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task("browser-issue-triage")
    setup = {name: content.encode() for name, content in task["setup"].items()}
    monkeypatch.setattr("evals.computer.run._export", lambda *_args: setup)
    observation = 'checkbox "Review docs" [checked]\n2 open'
    agent = {"last_result": observation, "last_result_error": False}
    assert _verify("docker", "context", "container", task, agent) == setup
    with pytest.raises(ValueError, match="missing"):
        _verify("docker", "context", "container", task, {**agent, "last_result": "2 open"})
    with pytest.raises(ValueError, match="without a successful observation"):
        _verify("docker", "context", "container", task, {**agent, "last_result_error": True})
