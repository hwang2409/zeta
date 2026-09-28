"""Keep the disposable-computer file boundary and launch flags honest."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evals.computer.browser_guest import BrowserGuest, _workspace_url
from evals.computer.egress_proxy import Handler, _destination, _public_address, fetch
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
    for args in (
        _container_args("test-computer"),
        _container_args("test-computer", BROWSER_IMAGE),
    ):
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
    assert (
        next(rule for rule in profile["syscalls"] if rule["names"] == ["chroot"])[
            "includes"
        ]
        == {}
    )
    public = _container_args("test-computer", BROWSER_IMAGE, "test-egress-volume")
    assert public[public.index("--network") + 1] == "none"
    assert public[public.index("--mount") + 1] == (
        "type=volume,src=test-egress-volume,dst=/proxy,readonly"
    )
    assert "-v" not in public and "--volume" not in public
    with pytest.raises(ValueError, match="restricted browser"):
        _container_args("test-computer", socket_volume="test-egress-volume")


def test_browser_fixture_stays_out_of_default_workflow_evals() -> None:
    assert "browser-todo-repair" not in {
        json.loads(line)["id"] for line in TASKS.read_text().splitlines()
    }
    assert (
        "chromium_sandbox=True"
        in _task("browser-todo-repair")["setup"]["test_browser_todo.py"]
    )
    assert (
        "Copper Glow" in _task("browser-deep-catalog")["setup"]["catalog_fixture.html"]
    )


def test_browser_guest_exposes_no_shell_and_rejects_public_url() -> None:
    script = Path(__file__).resolve().parents[1] / "evals/computer/guest.py"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "browser",
                "arguments": {"action": "open", "url": "https://example.com/"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "bash",
                "arguments": {"command": "id"},
            },
        },
    ]
    result = subprocess.run(
        [sys.executable, str(script), "--browser"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    listed, blocked, no_shell = (
        json.loads(line) for line in result.stdout.splitlines()
    )
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["browser"]
    assert blocked["result"]["isError"] is True
    assert "only file:///workspace/" in blocked["result"]["content"][0]["text"]
    assert "unknown tool" in no_shell["error"]["message"]
    with pytest.raises(ValueError, match="inside /workspace"):
        _workspace_url("file:///workspace/../../etc/passwd")
    with pytest.raises(ValueError, match="open a page"):
        BrowserGuest().call({"action": "fill", "role": "textbox", "value": ""})


def test_browser_guest_find_returns_bounded_context() -> None:
    class Match:
        def count(self) -> int:
            return 1

        def nth(self, _index: int) -> "Match":
            return self

        def locator(self, _selector: str) -> "Match":
            return self

        def evaluate(self, _script: str) -> int:
            return 100

        def aria_snapshot(self, **_options: object) -> str:
            return '- article "Listing 387":\n  - text: Solar patio lantern\n'

    class Page:
        url = "file:///workspace/catalog_fixture.html"

        def title(self) -> str:
            return "Catalog"

        def get_by_text(self, text: str) -> Match:
            assert text == "solar"
            return Match()

    guest = BrowserGuest()
    guest.page = Page()
    result = guest.call({"action": "find", "text": "solar"})
    assert 'article "Listing 387"' in result["content"][0]["text"]


def test_browser_broker_rejects_unapproved_and_private_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = "developer.mozilla.org"
    assert _destination(f"https://{host}/en-US/docs/?x=1", host) == (
        host,
        "/en-US/docs/?x=1",
    )
    for url in (
        "http://developer.mozilla.org/",
        "https://example.com/",
        "https://developer.mozilla.org.evil.test/",
        "https://developer.mozilla.org@127.0.0.1/",
        "https://developer.mozilla.org:444/",
        "https://developer.mozilla.org/\r\nHost: 127.0.0.1/",
    ):
        with pytest.raises(ValueError):
            _destination(url, host)
    monkeypatch.setattr(
        "evals.computer.egress_proxy.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="public IPs"):
        _public_address(host)


def test_browser_broker_rejects_redirect_before_browser_can_follow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, tls_context, response = MagicMock(), MagicMock(), MagicMock(status=302)
    monkeypatch.setattr(
        "evals.computer.egress_proxy._public_address", lambda _host: "93.184.216.34"
    )
    monkeypatch.setattr(
        "evals.computer.egress_proxy.socket.create_connection", connection
    )
    monkeypatch.setattr(
        "evals.computer.egress_proxy.ssl.create_default_context", lambda: tls_context
    )
    monkeypatch.setattr(
        "evals.computer.egress_proxy.http.client.HTTPResponse", lambda _socket: response
    )

    with pytest.raises(ValueError, match="HTTP 302"):
        fetch("https://developer.mozilla.org/", "developer.mozilla.org")
    assert connection.call_args.args[0] == ("93.184.216.34", 443)
    assert (
        tls_context.wrap_socket.call_args.kwargs["server_hostname"]
        == "developer.mozilla.org"
    )
    response.read.assert_not_called()


def test_browser_broker_rejects_non_get_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = MagicMock()
    monkeypatch.setattr("evals.computer.egress_proxy.fetch", upstream)
    handler = object.__new__(Handler)
    handler.rfile = io.BytesIO(
        b'{"method":"POST","url":"https://developer.mozilla.org/"}\n'
    )
    handler.wfile = io.BytesIO()
    handler.handle()
    assert "only GET" in json.loads(handler.wfile.getvalue())["error"]
    upstream.assert_not_called()


def test_public_browser_guest_keeps_one_tool_and_rejects_other_host() -> None:
    script = Path(__file__).resolve().parents[1] / "evals/computer/guest.py"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "browser",
                "arguments": {"action": "open", "url": "https://example.com/"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "bash",
                "arguments": {"command": "id"},
            },
        },
    ]
    result = subprocess.run(
        [sys.executable, str(script), "--browser-public", "developer.mozilla.org"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    listed, blocked, no_shell = (
        json.loads(line) for line in result.stdout.splitlines()
    )
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["browser"]
    assert blocked["result"]["isError"] is True
    assert "approved HTTPS host" in blocked["result"]["content"][0]["text"]
    assert "unknown tool" in no_shell["error"]["message"]


def test_browser_eval_checks_observed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task("browser-issue-triage")
    setup = {name: content.encode() for name, content in task["setup"].items()}
    monkeypatch.setattr("evals.computer.run._export", lambda *_args: setup)
    observation = 'checkbox "Review docs" [checked]\n2 open'
    agent = {"last_result": observation, "last_result_error": False}
    assert _verify("docker", "context", "container", task, agent) == setup
    with pytest.raises(ValueError, match="missing"):
        _verify(
            "docker", "context", "container", task, {**agent, "last_result": "2 open"}
        )
    with pytest.raises(ValueError, match="without a successful observation"):
        _verify(
            "docker", "context", "container", task, {**agent, "last_result_error": True}
        )
