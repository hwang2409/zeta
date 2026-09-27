"""Keep the disposable-computer file boundary and launch flags honest."""

import io
import json
import tarfile

import pytest

from evals.computer.run import (
    BROWSER_IMAGE,
    BROWSER_SECCOMP,
    TASKS,
    _archive,
    _container_args,
    _task,
    _unarchive,
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
