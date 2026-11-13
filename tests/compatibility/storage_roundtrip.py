"""Exercise real stored-family readers and writers in a supplied temporary home."""
from __future__ import annotations

import argparse
import asyncio
import json
import warnings
from collections.abc import Callable
from pathlib import Path

from zeta.protocol.types import Message
from zeta.storage_format import WRITER_EPOCH, advance

_SESSION_NAME = "format-roundtrip-predecessor"
_SESSION_ID_A = "a" * 32
_SESSION_ID_B = "b" * 32
_MESSAGE_ID_A = "1" * 32
_MESSAGE_ID_B = "2" * 32
_SKIPPED: dict[str, str] = {}


def _notice(family: str, reason: str) -> None:
    _SKIPPED[family] = reason
    print(f"::notice::skipped {family}: {reason}")


def _missing_api(family: str, owner: object, names: tuple[str, ...]) -> bool:
    missing = [name for name in names if not callable(getattr(owner, name, None))]
    if not missing:
        return False
    _notice(family, f"missing public API: {', '.join(missing)}")
    return True


def _message_texts(store: object) -> list[str]:
    return [
        part.text
        for row in store.path.read_text().splitlines()  # type: ignore[attr-defined]
        for data in [json.loads(row)]
        if data.get("type") == "message"
        for part in Message.from_dict(data["data"]["message"]).content
        if hasattr(part, "text")
    ]


def _session(home: Path, stage: str) -> dict[str, object] | None:
    try:
        from zeta.core.session import SessionManager
        from zeta.protocol.types import (
            Message,
            MessageOrigin,
            MessageRole,
            TextContent,
            with_message_origin,
        )
    except (ImportError, AttributeError) as exc:
        _notice("session transcript and metadata", str(exc))
        return None

    family = "session transcript and metadata"
    if _missing_api(
        family,
        SessionManager,
        ("create", "list_sessions_read_only", "open", "record_name"),
    ):
        return None
    manager = SessionManager(home, user_home=home)
    if stage == "predecessor":
        opened = manager.create(
            provider="ollama",
            model="compatibility",
            cwd=home,
            name=_SESSION_NAME,
            auto_project=False,
        )
        opened.store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("predecessor-session")]),
                MessageOrigin.USER,
            )
        )
        path = opened.store.path
        opened.store.close()
        with path.open("ab") as handle:
            handle.write(b'{"seq": 3, "id": "torn"')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            repaired = manager.open(opened.metadata.session_id)
        try:
            warning = repaired.store.entry(repaired.store.resident_refs()[-1])
            if warning.type != "warning" or warning.data.get("kind") != "transcript_recovery":
                raise RuntimeError("predecessor did not record transcript recovery")
        finally:
            repaired.store.close()
        return {"name": _SESSION_NAME, "texts": ["predecessor-session"]}

    metadata = next(
        (item for item in manager.list_sessions_read_only() if item.name in {_SESSION_NAME, "format-roundtrip-candidate"}),
        None,
    )
    if metadata is None:
        raise RuntimeError(f"{stage} did not read predecessor session metadata")
    opened = manager.open(metadata.session_id)
    try:
        texts = _message_texts(opened.store)
        if "predecessor-session" not in texts:
            raise RuntimeError(f"{stage} did not read predecessor transcript")
        if stage == "candidate":
            opened.store.append_message(
                with_message_origin(
                    Message(MessageRole.ASSISTANT, [TextContent("candidate-session")]),
                    MessageOrigin.UNKNOWN,
                )
            )
            manager.record_name(opened.metadata, name="format-roundtrip-candidate")
            texts.append("candidate-session")
        elif "candidate-session" not in texts or metadata.name != "format-roundtrip-candidate":
            raise RuntimeError("predecessor did not read candidate session output")
    finally:
        opened.store.close()
    return {"name": opened.metadata.name, "texts": texts}


def _projects(home: Path, stage: str) -> dict[str, object] | None:
    try:
        from zeta.project_inbox import ProjectInbox
        from zeta.project_registry import ProjectRegistry
    except (ImportError, AttributeError) as exc:
        _notice("project memory and inbox", str(exc))
        return None

    family = "project memory and inbox"
    if _missing_api(
        family,
        ProjectRegistry,
        ("create_project", "list_projects", "load_memory", "update_memory"),
    ) or _missing_api(family, ProjectInbox, ("send", "read")):
        return None
    registry = ProjectRegistry(home / "projects")
    if stage == "predecessor":
        sender = registry.create_project("format-sender", "compatibility sender")
        target = registry.create_project("format-target", "compatibility target")
        registry.update_memory(target.project_id, {"brief.md": "predecessor-memory\n"})
        ProjectInbox(registry, sessions_root=home / "sessions").send(
            from_project=sender.project_id,
            from_session=_SESSION_ID_A,
            to_project=target.project_id,
            kind="question",
            title="predecessor-inbox",
            body="predecessor-body",
            message_id=_MESSAGE_ID_A,
        )
    else:
        projects = {project.name: project for project in registry.list_projects()}
        sender = projects.get("format-sender")
        target = projects.get("format-target")
        if sender is None or target is None:
            raise RuntimeError(f"{stage} did not read predecessor projects")
        memory = dict(registry.load_memory(target.project_id))
        if memory.get("brief.md") != "predecessor-memory\n" and stage == "candidate":
            raise RuntimeError("candidate did not read predecessor project memory")
        inbox = ProjectInbox(registry, sessions_root=home / "sessions")
        messages = inbox.read(target.project_id)["new"]
        ids = {message["id"] for message in messages}
        if _MESSAGE_ID_A not in ids:
            raise RuntimeError(f"{stage} did not read predecessor project inbox")
        if stage == "candidate":
            registry.update_memory(target.project_id, {"brief.md": "candidate-memory\n"})
            inbox.send(
                from_project=sender.project_id,
                from_session=_SESSION_ID_B,
                to_project=target.project_id,
                kind="question",
                title="candidate-inbox",
                body="candidate-body",
                message_id=_MESSAGE_ID_B,
            )
            memory = dict(registry.load_memory(target.project_id))
            ids.add(_MESSAGE_ID_B)
        elif memory.get("brief.md") != "candidate-memory\n" or _MESSAGE_ID_B not in ids:
            raise RuntimeError("predecessor did not read candidate project output")
        return {"memory": memory["brief.md"], "inbox_ids": sorted(ids)}
    return {"memory": "predecessor-memory\n", "inbox_ids": [_MESSAGE_ID_A]}


def _accounts(home: Path, stage: str) -> dict[str, object] | None:
    try:
        from zeta.accounts import AccountId
        from zeta.oauth import OAuthTokens
        from zeta.providers.factory import credential_store
    except (ImportError, AttributeError) as exc:
        _notice("accounts metadata", str(exc))
        return None

    account = AccountId("codex", "compatibility")
    credential = credential_store("codex", home=home, account=account)
    if credential is None:
        _notice("accounts metadata", "codex account credential store is unavailable")
        return None
    if _missing_api("accounts metadata", credential, ("read", "save")):
        return None
    expected = {
        "predecessor": None,
        "candidate": "fake-predecessor-access",
        "reader": "fake-candidate-access",
    }[stage]
    current = credential.read()
    if expected is not None and (current is None or current.access_token != expected):
        raise RuntimeError(f"{stage} did not read expected account metadata")
    if stage != "reader":
        value = "fake-predecessor-access" if stage == "predecessor" else "fake-candidate-access"
        credential.save(OAuthTokens(value, f"{value}-refresh", 4_000_000_000.0))
        current = credential.read()
    if current is None or current.access_token != "fake-candidate-access" and stage == "reader":
        raise RuntimeError("predecessor did not read candidate account metadata")
    return {"account": str(account), "access_token": current.access_token}


def _mcp(home: Path, stage: str) -> dict[str, object] | None:
    try:
        from zeta.mcp.oauth_store import MCPCredentialStore, MCPOAuthToken
    except (ImportError, AttributeError) as exc:
        _notice("mcp token store", str(exc))
        return None

    if _missing_api("mcp token store", MCPCredentialStore, ("load", "save_login")):
        return None

    async def operate() -> dict[str, object]:
        store = MCPCredentialStore(home=home)
        snapshot = await store.load("compatibility")
        expected = {
            "predecessor": None,
            "candidate": "fake-predecessor-mcp",
            "reader": "fake-candidate-mcp",
        }[stage]
        if expected is not None and (
            snapshot.token is None or snapshot.token.access_token != expected
        ):
            raise RuntimeError(f"{stage} did not read expected mcp token")
        if stage != "reader":
            value = "fake-predecessor-mcp" if stage == "predecessor" else "fake-candidate-mcp"
            token = MCPOAuthToken(
                access_token=value,
                refresh_token=f"{value}-refresh",
                expires_at=None,
                token_type="Bearer",
                scope="compatibility",
                authorization_server="https://auth.test",
                token_endpoint="https://auth.test/token",
                authorization_endpoint="https://auth.test/authorize",
                client_id="fake-client",
                client_secret="fake-secret",
                redirect_uri="http://127.0.0.1/callback",
                resource="https://mcp.test/resource",
            )
            snapshot = await store.save_login("compatibility", token)
        if snapshot.token is None or snapshot.token.access_token != "fake-candidate-mcp" and stage == "reader":
            raise RuntimeError("predecessor did not read candidate mcp token")
        return {"access_token": snapshot.token.access_token, "generation": snapshot.generation}

    return asyncio.run(operate())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--stage", choices=("predecessor", "candidate", "reader"), required=True)
    parser.add_argument("--advance-from", type=int)
    parser.add_argument("--skip", action="append", default=[])
    args = parser.parse_args()
    if args.advance_from is not None:
        advance(args.home, expected_epoch=args.advance_from)
    skip = set(args.skip)

    def run(family: str, operation: Callable[[Path, str], object]) -> object:
        if family in skip:
            _notice(family, "API unavailable in selected predecessor")
            return None
        return operation(args.home, args.stage)

    observed = {
        "stage": args.stage,
        "writer_epoch": WRITER_EPOCH,
        "session": run("session transcript and metadata", _session),
        "projects": run("project memory and inbox", _projects),
        "accounts": run("accounts metadata", _accounts),
        "mcp": run("mcp token store", _mcp),
        "skipped": _SKIPPED,
    }
    print(json.dumps(observed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
