"""Controller writes to project memory and inbox through the user-path owners."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from ....memory.user_authorization import MemoryMutationAuthorization
from ....project_errors import (
    MemoryDigestMismatchError,
    ProjectNotFoundError,
    ProjectRegistryBusyError,
    ProjectRegistryError,
)
from ....project_inbox import (
    KINDS,
    InboxBusyError,
    InboxConflictError,
    ProjectInbox,
    is_message_id,
)
from ....project_memory_history import PROJECT_MEMORY_FILES
from ...resource_protocol import (
    Access,
    ResourceMethod,
    require_params,
    require_revision,
)

FEATURE = "project_writes_v1"
_DELIVERY_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_ENTRY_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MAX_TEXT_BYTES = 1024 * 1024


def _text(value: object, *, empty: bool = False) -> str:
    from ...protocol import invalid_resource_params

    if type(value) is not str or (not empty and not value.strip()) or "\0" in value:
        raise invalid_resource_params()
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise invalid_resource_params() from None
    if size > _MAX_TEXT_BYTES:
        raise invalid_resource_params()
    return value


def _entry_id(value: object) -> str:
    from ...protocol import invalid_resource_params

    if type(value) is not str or _ENTRY_ID.fullmatch(value) is None:
        raise invalid_resource_params()
    return value


def _registry(client):
    from ...protocol import ErrorDescriptor, resource_error

    registry = client.server.runtime.manager.project_registry
    if registry is None:
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)
    return registry


def _project_id(params: dict[str, Any]) -> str:
    from ...protocol import invalid_resource_params

    value = params["project_id"]
    if type(value) is not str or not value or len(value) > 256:
        raise invalid_resource_params()
    return value


def _require_project(registry, project_id: str) -> None:
    """Refuse an unknown project exactly as the project reads do."""
    from ...protocol import ErrorDescriptor, ProtocolError
    from ..projects import ProjectNotFound

    try:
        registry.show_project(project_id)
    except ProjectNotFoundError:
        message = str(ProjectNotFound(project_id))
        raise ProtocolError(ErrorDescriptor.PROJECT_NOT_FOUND, message, {"project_id": project_id}) from None


def _owner_call(registry, project_id: str, write: Callable[[], Any]) -> Any:
    from ...protocol import (
        ErrorDescriptor,
        invalid_resource_params,
        resource_error,
        stale_resource_revision,
    )

    _require_project(registry, project_id)
    try:
        return write()
    except MemoryDigestMismatchError:
        raise stale_resource_revision("project_memory") from None
    except (ProjectRegistryBusyError, InboxBusyError):
        raise resource_error(ErrorDescriptor.RESOURCE_BUSY) from None
    except InboxConflictError:
        raise invalid_resource_params(reason="delivery_id_reused") from None
    except ProjectRegistryError:
        # Owner rejections (unknown entry, empty history, tool subprocess) carry details we must not relay.
        raise invalid_resource_params(reason="operation_rejected") from None


def _memory_write(registry, project_id: str, revision: str, mutation: object) -> Callable[[], dict]:
    """Validate one format-specific user operation and bind it to its owner method."""
    from ...protocol import invalid_resource_params, stale_resource_revision

    require_params(mutation, "format", "op", optional=("file", "kind", "content", "target", "entry_ids"))
    memory_format, op = mutation["format"], mutation["op"]
    fields = set(mutation) - {"format", "op"}
    authorize = MemoryMutationAuthorization.direct_slash().authorize
    digest = registry._memory_digest_value

    def contents_result(contents: list[tuple[str, str]]) -> dict:
        return {"revision": digest(dict(contents)), "data": {"format": 1}}

    def entry_result(result) -> dict:
        return {"revision": result.digest, "data": {"format": 2, "version_id": result.version}}

    if memory_format == 1 and op == "set" and fields == {"file", "content"}:
        name, content = mutation["file"], _text(mutation["content"], empty=True)
        if name not in PROJECT_MEMORY_FILES:
            raise invalid_resource_params()
        call = lambda: contents_result(registry.compare_and_swap_memory(
            project_id, expected_digest=revision, updates={name: content},
        ).contents)
        action, target = "set", name
    elif memory_format == 1 and op == "accept" and fields == {"file"}:
        name = mutation["file"]
        if name not in PROJECT_MEMORY_FILES:
            raise invalid_resource_params()
        call = lambda: contents_result(registry.accept_memory(project_id, name, expected_digest=revision))
        action, target = "accept", name
    elif memory_format == 1 and op == "undo" and fields <= {"target"}:
        version = _entry_id(mutation["target"]) if "target" in fields else None
        call = lambda: contents_result(registry.undo_memory(
            project_id, expected_current=version, expected_digest=revision,
        ))
        action, target = "undo", version or "latest"
    elif memory_format == 2 and op == "set" and fields == {"kind", "content"}:
        kind, content = _entry_id(mutation["kind"]), _text(mutation["content"], empty=True)
        call = lambda: entry_result(registry._replace_entry_kind(
            project_id, kind, content, expected_digest=revision,
        ))
        action, target = "set", kind
    elif memory_format == 2 and op == "accept" and fields == {"target"}:
        entry = _entry_id(mutation["target"])
        call = lambda: entry_result(registry._accept_memory_entry(project_id, entry, expected_digest=revision))
        action, target = "accept", entry
    elif memory_format == 2 and op == "resolve" and fields == {"entry_ids"}:
        ids = mutation["entry_ids"]
        if type(ids) is not list or not 1 <= len(ids) <= 64:
            raise invalid_resource_params()
        entries = tuple(map(_entry_id, ids))
        call = lambda: entry_result(registry._resolve_memory_entries(
            project_id, entries, expected_digest=revision,
        ))
        action, target = "resolve", ", ".join(entries)
    elif memory_format == 2 and op == "undo" and fields <= {"target"}:
        entry = _entry_id(mutation["target"]) if "target" in fields else None
        call = lambda: entry_result(registry._undo_entry_transaction(
            project_id, entry, expected_digest=revision,
        ))
        action, target = "undo", entry or "latest"
    else:
        raise invalid_resource_params()

    def write() -> dict:
        if registry.memory_transactions.read(project_id).format != memory_format:
            # A migration changed the format, so the client's view and revision are stale.
            raise stale_resource_revision("project_memory")
        authorize(action=action, target=target)
        return call()

    return write


async def project_memory_update(client, request_id, params):
    require_params(params, "project_id", "revision", "mutation")
    require_revision(params["revision"])
    registry = _registry(client)
    project_id = _project_id(params)
    write = _memory_write(registry, project_id, params["revision"], params["mutation"])
    return await client.connections.run_blocking(lambda: _owner_call(registry, project_id, write))


def _sender(client) -> tuple[str, str, Any]:
    """The attached session and its project are the sender, exactly as `/inbox send` uses them."""
    from ...protocol import ErrorDescriptor, resource_error

    runtime = client.server.runtime
    if runtime.opened is None or runtime.loop is None:
        raise resource_error(ErrorDescriptor.NO_ACTIVE_SESSION)
    project_id = runtime.opened.metadata.project_id
    if project_id is None:
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)
    return project_id, runtime.session_id, runtime.loop


async def project_inbox_send(client, request_id, params):
    from ...protocol import invalid_resource_params

    require_params(params, "project_id", "mutation")
    to_project = _project_id(params)
    mutation = params["mutation"]
    require_params(mutation, "kind", "title", "body", "delivery_id", optional=("to_session",))
    kind, delivery_id = mutation["kind"], mutation["delivery_id"]
    to_session = mutation.get("to_session")
    if (
        kind not in KINDS
        or type(delivery_id) is not str
        or _DELIVERY_ID.fullmatch(delivery_id) is None
        or (to_session is not None and not is_message_id(to_session))
    ):
        raise invalid_resource_params()
    title, body = _text(mutation["title"]), _text(mutation["body"], empty=True)
    registry = _registry(client)
    from_project, from_session, loop = _sender(client)
    # One delivery id per sender session and recipient: a retry maps to the same message id,
    # so the owner's immutable-record check gives idempotency and rejects a changed payload.
    message_id = hashlib.sha256(
        f"serve-inbox\0{from_session}\0{to_project}\0{delivery_id}".encode()
    ).hexdigest()[:32]
    inbox = ProjectInbox(registry, sessions_root=registry.root.parent / "sessions")

    def send() -> dict:
        sent, _outcome = inbox.send_user_with_outcome(
            from_project=from_project,
            from_session=from_session,
            to_project=to_project,
            kind=kind,
            title=title,
            body=body,
            message_id=message_id,
            to_session=to_session,
        )
        loop.notify_project_inbox_publication()  # wake the attached poller; others poll as usual
        return {"data": {"message_id": sent}}

    return await client.connections.run_blocking(lambda: _owner_call(registry, to_project, send))


METHODS = (
    ResourceMethod("project_memory_update", Access.CONTROLLER_WRITE, project_memory_update),
    ResourceMethod("project_inbox_send", Access.CONTROLLER_WRITE, project_inbox_send),
)
