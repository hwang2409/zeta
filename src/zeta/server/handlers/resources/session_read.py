"""Page another root session without attaching to it or taking execution ownership.

Only the result is bounded (groups and bytes): every call reads and projects the whole
target transcript off the event loop, so serve runs at most one such read at a time.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import weakref
from dataclasses import replace
from functools import partial

from ....core.checkpoints import ConversationIntegrityError
from ....core.session import validate_session_id
from ....core.session_files import (
    SessionError,
    SessionNotFoundError,
    child_directory,
    session_root,
)
from ....core.store import ConversationStore, ObserverSnapshotChanged
from ....project_registry import ProjectRegistryError
from ....read_page import InvalidParams, Key, TooLarge, page_result, resume
from ....storage_format import StorageFormatError, open_reader
from ...resource_protocol import Access, ResourceMethod, require_params

FEATURE = "session_history_read_v1"
_METHOD = "session_history_read"
READ_DEADLINE_SECONDS = 10
RETRY_AFTER_MS = 2000
_ORDER = (Key("group", int), Key("snapshot", str))
# One in-flight whole-transcript read per serve process, keyed by its event loop.
_in_flight: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Future] = weakref.WeakKeyDictionary()


def _page(store, request_id, features, source, start, backward, limit, max_bytes):
    from ...event_contract import shape_approval_history_rows
    from ...protocol import (
        ErrorDescriptor,
        FrameCodec,
        invalid_resource_params,
        resource_error,
    )
    from ..history_projection import (
        _history_endpoint,
        _history_group,
        history_projection,
    )

    row_features = {
        "include_delivery_id": {"delivery_id", "user_message_event", "user_message_delivery_id"} <= features,
        "fidelity": "session_history_fidelity" in features,
        "content_read": "content_read" in features,
        "approval_origin": "tool_approval_origin" in features,
    }
    projection = history_projection(store, served_turn_end="served_turn_end" in features, **row_features)
    current = len(projection.entry_groups) - 1
    if start.position is None:
        start = replace(start, pinned=(current,))
    end = start.pinned[0]

    def snapshot():
        # Appends keep the groups below ``end``; a rewrite or replaced file does not.
        identity = (source, end, _history_endpoint(projection, end), store.repair_generation)
        return hashlib.sha256(repr(identity).encode()).hexdigest()

    last = current if projection.entry_groups[current] else current - 1
    if start.position is None:
        first = last + 1 if backward else 0
    else:
        group, token = start.position
        if end > current or token != snapshot():
            raise resource_error(ErrorDescriptor.HISTORY_SNAPSHOT_STALE)
        if not group <= end or (backward and group == 0):
            raise invalid_resource_params()
        first = group if backward else group + 1
    indexes = range(first - 1, -1, -1) if backward else range(first, last + 1)
    token = snapshot()
    validator = {"end": end, "entry_id": _history_endpoint(projection, end), "repair_generation": store.repair_generation}
    groups = [
        shape_approval_history_rows(
            list(_history_group(projection, index, end=current, backward=backward, validator=validator, **row_features)[0]),
            include_restart_reason="approval_restart_reason" in features,
        )
        for index in indexes[:limit]
    ]

    def shape(items, cursor):
        ordered = reversed(items) if backward else items
        return {"messages": [row for group in ordered for row in group], "next_cursor": cursor}

    codec = FrameCodec()
    try:
        return page_result(
            groups, lambda count: (indexes[count], token), start, more=len(indexes) > limit, total=None,
            fits=lambda page: codec.response_size(request_id, page) <= max_bytes, shape=shape,
        )
    except TooLarge:
        raise resource_error(ErrorDescriptor.RESULT_TOO_LARGE) from None


def _identity(stat):
    return stat.st_dev, stat.st_ino


def _read(home, manager, session_id, project_id, request_id, features, start, backward, limit, max_bytes):
    from ...protocol import ErrorDescriptor, resource_error

    def check_project():
        if manager.read_metadata(session_id).project_id != project_id:
            raise resource_error(ErrorDescriptor.SESSION_NOT_FOUND)

    def transcript(directory_fd):
        return _identity(os.stat("conversation.jsonl", dir_fd=directory_fd, follow_symlinks=False))

    continuation = start.position is not None
    try:
        with open_reader(home), session_root(manager.sessions_dir) as root:
            for attempt in range(2):
                try:
                    with child_directory(root, session_id) as directory:
                        check_project()
                        source = transcript(directory)
                        with ConversationStore(
                            manager.sessions_dir, session_id=session_id,
                            _read_only=True, _observer_snapshot=True, _must_exist=True,
                            _storage_lease=manager._storage_lease,
                        ) as store:
                            check_project()
                            with child_directory(root, session_id) as current:
                                identity = _identity(os.fstat(directory))
                                if identity != _identity(os.fstat(store.directory_fd)) or identity != _identity(os.fstat(current)):
                                    raise ObserverSnapshotChanged("session directory changed")
                            if source != transcript(directory):
                                raise ObserverSnapshotChanged("transcript source changed")
                            page = _page(store, request_id, features, source, start, backward, limit, max_bytes)
                            check_project()
                            return page
                except ObserverSnapshotChanged:
                    if continuation:
                        raise resource_error(ErrorDescriptor.HISTORY_SNAPSHOT_STALE) from None
                    if attempt:
                        raise
                except FileNotFoundError:
                    # An existing session with missing files is not an absent session.
                    os.stat(session_id, dir_fd=root, follow_symlinks=False)
                    raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE) from None
    except (SessionNotFoundError, FileNotFoundError):
        raise resource_error(ErrorDescriptor.SESSION_NOT_FOUND) from None
    except (OSError, SessionError, ConversationIntegrityError, StorageFormatError):
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE) from None


def _launch_project_id(runtime):
    from ...protocol import ErrorDescriptor, resource_error

    try:
        return runtime.launch_project_id()
    except (ProjectRegistryError, OSError, ValueError, StorageFormatError):
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE) from None


def _read_idle(home, runtime, session_id, request_id, features, start, backward, limit, max_bytes):
    from ...protocol import ErrorDescriptor, resource_error

    scope = _launch_project_id(runtime)
    page = _read(home, runtime.manager, session_id, scope, request_id, features, start, backward, limit, max_bytes)
    if _launch_project_id(runtime) != scope:
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)
    return page


def _read_scope(client, session_id):
    from ...protocol import ErrorDescriptor, invalid_resource_params, resource_error

    runtime = client.server.runtime
    home = client.server.home
    opened = runtime.opened
    if opened is None:
        generation = runtime.attachment_generation

        def validate():
            # A session can attach and detach during the read.
            if (
                client.server.runtime is not runtime or runtime.opened is not None
                or runtime.attachment_generation != generation
            ):
                raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)

        return partial(_read_idle, home, runtime, session_id), validate

    if session_id == runtime.session_id:
        raise invalid_resource_params()  # the attached session pages through session_history
    project_id = opened.metadata.project_id

    def validate():
        if runtime.opened is not opened or opened.metadata.project_id != project_id:
            raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)

    return partial(_read, home, runtime.manager, session_id, project_id), validate


def _retrieve(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()  # a timed-out caller no longer awaits this result


async def _handle(client, request_id, params):
    from ...protocol import (
        MAX_FRAME_BYTES,
        ErrorDescriptor,
        invalid_resource_params,
        resource_error,
    )
    from ...session_replay import history_features
    from ..history import HISTORY_FIRST_PAGE_ENTRIES

    require_params(params, "session_id", optional=("cursor", "direction", "limit", "max_bytes"))
    session_id = params["session_id"]
    direction = params.get("direction", "backward")
    limit = params.get("limit", 50)
    max_bytes = params.get("max_bytes", MAX_FRAME_BYTES)
    if (
        not isinstance(session_id, str) or not session_id.isascii() or len(session_id) > 255
        or direction not in ("backward", "forward")
        or type(limit) is not int or not 1 <= limit <= HISTORY_FIRST_PAGE_ENTRIES
        or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_FRAME_BYTES
    ):
        raise invalid_resource_params()
    try:
        validate_session_id(session_id)
        start = resume(params.get("cursor"), (_METHOD, session_id, direction), (0,), _ORDER)
    except (SessionError, InvalidParams):
        raise invalid_resource_params() from None
    reader, validate_scope = _read_scope(client, session_id)
    loop = asyncio.get_running_loop()
    if loop in _in_flight:
        error = resource_error(ErrorDescriptor.OPERATION_BUSY)
        error.data["retry_after_ms"] = RETRY_AFTER_MS
        raise error
    features = history_features(client.features)

    read_page = partial(reader, request_id, features, start, direction == "backward", limit, max_bytes)
    read = asyncio.ensure_future(client.connections.run_blocking(read_page))
    _in_flight[loop] = read
    # The slot frees only when the worker thread is done, even after a timeout or cancel.
    read.add_done_callback(lambda done: (_retrieve(done), _in_flight.pop(loop, None)))
    try:
        async with asyncio.timeout(READ_DEADLINE_SECONDS):
            page = await asyncio.shield(read)
    except TimeoutError:
        raise resource_error(ErrorDescriptor.STATE_READ_UNAVAILABLE) from None
    validate_scope()
    return page


METHODS = (ResourceMethod(_METHOD, Access.READ, _handle),)
