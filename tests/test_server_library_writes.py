"""Skill and agent profile writes over real serve sockets."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from rich.text import Text

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.serve_contract import ServeContract
from tests.support.server_client import send
from tests.support.sockets import short_socket_path
from tests.test_server import _connect, _read, _request
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.server import ZetaServer
from zeta.skills import resource_store
from zeta.tui.render import render_event

_FEATURES = ["serve_observers", "library_read_v1", "library_write_v1"]
_WRITES = ("skill_write", "skill_delete", "agent_profile_write", "agent_profile_delete")
_STALE = {"code": "stale_library_revision", "reason": "stale_library_revision"}


def _document(name: str, body: str = "body", description: str = "notes") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _reason(reply: dict) -> str | None:
    assert reply["error"]["code"] == -32602, reply
    return reply["error"]["data"].get("reason")


@pytest.mark.asyncio
async def test_observer_cannot_write_and_controller_writes_with_cas(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside.mkdir()
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        observer = await contract.connect("observer")
        await observer.request("hello", "hello", {"protocol_version": "1.2", "role": "observer", "features": _FEATURES})
        for method in _WRITES:
            reply = await observer.request(method, method, {"name": "alpha", "scope": "home"})
            assert reply["error"]["code"] == -32008
            assert reply["error"]["data"] == {
                "code": "observer_read_only", "reason": "observer_read_only", "method": method,
            }
        assert not (home / "skills").exists()

        client = await contract.connect()
        hello = await client.request("hello", "hello", {"protocol_version": "1.2", "features": _FEATURES})
        assert "library_write_v1" in hello["result"]["capabilities"]["features"]

        # create, then the second create of the same name is stale
        token = "sk_library_write_secret_12345"
        first = _document("alpha", f"uses {token}")
        created = await client.request("create", "skill_write", {
            "name": "alpha", "scope": "home", "mutation": {"document": first},
        })
        assert created["result"] == {
            "revision": _sha(first), "data": {"name": "alpha", "scope": "home", "redacted": True},
        }
        assert token not in json.dumps(created)
        assert (home / "skills" / "alpha.md").read_text() == first
        again = await client.request("again", "skill_write", {
            "name": "alpha", "scope": "home", "mutation": {"document": _document("alpha", "other")},
        })
        assert again["error"]["code"] == -32009 and again["error"]["data"] == _STALE

        # update with the read revision; the old revision is then stale
        read = (await client.request("read", "skill_read", {"name": "alpha", "scope": "home"}))["result"]
        assert read["revision"] == _sha(first)
        second = _document("alpha", "second body")
        updated = await client.request("update", "skill_write", {
            "name": "alpha", "scope": "home", "revision": read["revision"], "mutation": {"document": second},
        })
        assert updated["result"]["revision"] == _sha(second)
        assert (home / "skills" / "alpha.md").read_text() == second
        stale = await client.request("stale", "skill_write", {
            "name": "alpha", "scope": "home", "revision": read["revision"], "mutation": {"document": first},
        })
        assert stale["error"]["data"] == _STALE
        catalog = (await client.request("catalog", "skill_catalog", {"scope": "home"}))["result"]["data"]
        assert [item["name"] for item in catalog["items"]] == ["alpha"]

        # a held root lock is a typed busy refusal
        with (home / "skills" / ".library.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = await client.request("busy", "skill_delete", {
                "name": "alpha", "scope": "home", "revision": _sha(second),
            })
        assert busy["error"]["code"] == -32004 and busy["error"]["data"]["code"] == "operation_busy"

        # delete with CAS
        stale_delete = await client.request("stale-delete", "skill_delete", {
            "name": "alpha", "scope": "home", "revision": _sha(first),
        })
        assert stale_delete["error"]["data"] == _STALE
        deleted = await client.request("delete", "skill_delete", {
            "name": "alpha", "scope": "home", "revision": _sha(second),
        })
        assert deleted["result"] == {"data": {"name": "alpha", "scope": "home"}}
        assert not (home / "skills" / "alpha.md").exists()
        gone = await client.request("gone", "skill_delete", {"name": "alpha", "scope": "home", "revision": _sha(second)})
        assert _reason(gone) == "resource_not_found"

        # packaged items are read-only
        for params in (
            {"name": "general", "scope": "home", "revision": "0" * 64},
            {"name": "general", "scope": "effective", "revision": "0" * 64},
        ):
            assert _reason(await client.request("packaged", "agent_profile_delete", params)) == "packaged_read_only"

        # invalid documents are refused before anything is written
        invalid = await client.request("invalid", "skill_write", {
            "name": "broken", "scope": "home", "mutation": {"document": "---\nname: [\n---\nbody\n"},
        })
        assert _reason(invalid) == "invalid_document"
        assert invalid["error"]["data"]["detail"] == "skill broken.md has invalid YAML frontmatter"
        profile = await client.request("profile", "agent_profile_write", {
            "name": "bad", "scope": "home", "mutation": {"document": "---\nname: bad\ndescription: d\nmax_turns: -1\n---\nb\n"},
        })
        assert _reason(profile) == "invalid_document"
        mismatch = await client.request("mismatch", "skill_write", {
            "name": "beta", "scope": "home", "mutation": {"document": _document("gamma")},
        })
        assert _reason(mismatch) == "name_mismatch"
        large = await client.request("large", "skill_write", {
            "name": "big", "scope": "home", "mutation": {"document": _document("big", "x" * (128 * 1024))},
        })
        assert _reason(large) == "document_too_large"
        assert sorted(path.name for path in (home / "skills").iterdir()) == [".library.lock"]
        assert not (home / "agents").exists()

        # containment: names never leave the root, and links out of it are refused
        for name in ("../x", "/abs", "a/b", "..", "a\\b"):
            reply = await client.request("escape", "skill_write", {
                "name": name, "scope": "home", "mutation": {"document": _document("x")},
            })
            assert reply["error"]["code"] == -32602
        too_long = await client.request("too-long", "skill_delete", {"name": "n" * 300, "scope": "home", "revision": "0" * 64})
        assert _reason(too_long) == "name_too_long"
        (outside / "target.md").write_text("outside")
        (home / "skills" / "evil.md").symlink_to(outside / "target.md")
        linked_file = await client.request("linked-file", "skill_write", {
            "name": "evil", "scope": "home", "mutation": {"document": _document("evil")},
        })
        assert _reason(linked_file) == "unsafe_path"
        (outside / "sneaky").mkdir()
        (outside / "sneaky" / "SKILL.md").write_text(_document("sneaky", "outside body"))
        (home / "skills" / "sneaky").symlink_to(outside / "sneaky", target_is_directory=True)
        sneaky = _document("sneaky", "inside body")
        linked_dir = await client.request("linked-dir", "skill_write", {
            "name": "sneaky", "scope": "home", "mutation": {"document": sneaky},
        })
        assert linked_dir["result"]["revision"] == _sha(sneaky)
        assert (home / "skills" / "sneaky.md").read_text() == sneaky
        assert sorted(path.name for path in outside.rglob("*")) == ["SKILL.md", "sneaky", "target.md"]
        assert (outside / "target.md").read_text() == "outside"
        assert (outside / "sneaky" / "SKILL.md").read_text() == _document("sneaky", "outside body")

        # project scope needs an attached session and writes under its .zeta
        profile_doc = _document("helper", "help")
        missing = await client.request("no-project", "agent_profile_write", {
            "name": "helper", "scope": "project", "mutation": {"document": profile_doc},
        })
        assert missing["error"]["data"]["code"] == "state_unavailable"
        assert "result" in await client.request("start", "new_session", {"provider": "codex"})
        project = await client.request("project", "agent_profile_write", {
            "name": "helper", "scope": "project", "mutation": {"document": profile_doc},
        })
        assert project["result"]["data"] == {"name": "helper", "scope": "project", "redacted": False}
        assert (tmp_path / ".zeta" / "agents" / "helper.md").read_text() == profile_doc
        profiles = (await observer.request("profiles", "agent_profile_catalog", {"scope": "project"}))["result"]
        assert [item["name"] for item in profiles["data"]["items"]] == ["helper"]
    finally:
        await contract.close()


def _server(tmp_path: Path) -> ZetaServer:
    return ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )


async def _hello(reader, writer) -> None:
    await _request(reader, writer, "hello", "hello", {
        "protocol_version": "1.2", "features": ["library_read_v1", "library_write_v1", "live_config_changed"],
    })


@pytest.mark.asyncio
async def test_create_race_loses_to_the_file_published_first(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "skills" / "alpha.md"
    winner = _document("alpha", "winner")
    original = resource_store._check

    def check_then_lose(current, revision, name, *, profiles):
        original(current, revision, name, profiles=profiles)
        target.write_text(winner)  # another creator publishes after our missing-file check

    # test-contract: seam forces the create race between the missing-file check and the publish
    monkeypatch.setattr(resource_store, "_check", check_then_lose)
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        reply = (await _request(reader, writer, "create", "skill_write", {
            "name": "alpha", "scope": "home", "mutation": {"document": _document("alpha", "loser")},
        }))[-1]
        assert reply["error"]["code"] == -32009 and reply["error"]["data"] == _STALE
        assert target.read_text() == winner
        assert sorted(path.name for path in target.parent.iterdir()) == [".library.lock", "alpha.md"]
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_interrupted_publish_leaves_old_or_new_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "agents" / "alpha.md"
    target.parent.mkdir()
    old = _document("alpha", "old")
    target.write_text(old)
    real_replace = os.replace
    mode = {"fail": True}
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def publish(source, destination, **kwargs):
        if destination != "alpha.md":
            return real_replace(source, destination, **kwargs)
        if mode["fail"]:
            raise OSError("interrupted at publish")
        entered.set()
        release.wait(10)
        try:
            return real_replace(source, destination, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(os, "replace", publish)
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        failed = (await _request(reader, writer, "fail", "agent_profile_write", {
            "name": "alpha", "scope": "home", "revision": _sha(old), "mutation": {"document": _document("alpha", "x")},
        }))[-1]
        assert failed["error"] == {"code": -32000, "message": "unexpected server or storage failure"}
        assert target.read_text() == old
        assert sorted(path.name for path in target.parent.iterdir()) == [".library.lock", "alpha.md"]

        # the client goes away while the write waits at the publish step
        mode["fail"] = False
        new = _document("alpha", "new")
        frame = {"jsonrpc": "2.0", "id": "cancel", "method": "agent_profile_write", "params": {
            "name": "alpha", "scope": "home", "revision": _sha(old), "mutation": {"document": new},
        }}
        writer.write((json.dumps(frame) + "\n").encode())
        await await_or_fail(writer.drain(), what="send write")
        assert await await_or_fail(asyncio.to_thread(entered.wait, 10), what="write reached publish")
        writer.close()
        release.set()
        assert await await_or_fail(asyncio.to_thread(finished.wait, 10), what="publish finished")
        assert target.read_text() in (old, new)
        assert sorted(path.name for path in target.parent.iterdir()) == [".library.lock", "alpha.md"]
    finally:
        release.set()
        await close_serve(server)


@pytest.mark.asyncio
async def test_written_skill_refreshes_live_catalog_once(tmp_path: Path) -> None:
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        await _request(reader, writer, "start", "new_session", {"provider": "codex"})
        loop = server.runtime.loop
        assert loop is not None
        created = (await _request(reader, writer, "create", "skill_write", {
            "name": "fresh", "scope": "home", "mutation": {"document": _document("fresh")},
        }))[-1]
        assert "result" in created

        await await_or_fail(loop.refresh_live_config(), what="turn-boundary refresh")
        await await_or_fail(loop.refresh_live_config(), what="second refresh")
        assert "fresh" in {skill.name for skill in loop.tool_registry.skill_catalog.skills}
        # the serve sink delivers every published event before the probe's reply
        await send(reader, writer, "probe", "skill_catalog")
        changes = []
        while (frame := await _read(reader)).get("id") != "probe":
            if frame.get("params", {}).get("event") == "config_changed":
                changes.append(frame["params"]["data"])
        assert len(changes) == 1, "one write must produce one config_changed"
        assert changes[0]["message"] == "skills changed: added fresh"
        rendered = render_event(StreamEvent(StreamEventType.CONFIG_CHANGED, data=changes[0]))
        assert isinstance(rendered, Text) and rendered.plain == "skills changed: added fresh"
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_project_scope_uses_the_root_sessions_load_from_a_worktree(tmp_path: Path) -> None:
    repo, home = tmp_path / "repo", tmp_path / "home"
    worktree = repo / ".worktrees" / "wt"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                       cwd=repo, check=True, capture_output=True)

    git("init")
    git("commit", "--allow-empty", "-m", "initial")
    git("worktree", "add", "-b", "feature", str(worktree))
    cwd = worktree / "sub"
    cwd.mkdir()
    server = ZetaServer(
        provider="codex", home=home, cwd=cwd, socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        await _request(reader, writer, "start", "new_session", {"provider": "codex"})
        document = _document("alpha")
        written = (await _request(reader, writer, "write", "skill_write", {
            "name": "alpha", "scope": "project", "mutation": {"document": document},
        }))[-1]
        assert "result" in written, written
        landed = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("alpha.md"))
        assert landed == ["repo/.zeta/skills/alpha.md"], "write must land in the root sessions load from"
        assert (repo / ".zeta" / "skills" / "alpha.md").read_text() == document
        catalog = (await _request(reader, writer, "catalog", "skill_catalog", {"scope": "project"}))[-1]
        assert [item["name"] for item in catalog["result"]["data"]["items"]] == ["alpha"]
        loop = server.runtime.loop
        assert loop is not None
        await await_or_fail(loop.refresh_live_config(), what="live refresh")
        assert "alpha" in {skill.name for skill in loop.tool_registry.skill_catalog.skills}
    finally:
        await close_serve(server, writer)
