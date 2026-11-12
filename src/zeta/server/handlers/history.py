"""Serve session_history paging and request handling."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace

from ...core.store import ObserverSnapshotChanged
from ...core.store._history_index import InvalidHistoryPageRequest
from ..protocol import MAX_REQUEST_ID_BYTES, ErrorDescriptor, FrameCodec, ProtocolError
from ..runtime import ServerRuntime
from . import child_run_lifecycle
from .history_projection import (
    HistoryProjection,
    _cache,
    _CachedProjection,
    _generation,
    _history_endpoint,
    _history_group,
    _history_group_fits,
    _publish_cache,
    history_projection,
    history_projection_from_groups,
)

HISTORY_FIRST_PAGE_ENTRIES = 128
_STABLE_HISTORY_ATTEMPTS = 16
_cache_creation_lock = threading.Lock()
_MISSING_VALIDATOR = object()


def _stable_history_read[T](
    produce: Callable[[bool], object], read: Callable[[object], T], *, has_end: bool
) -> T:
    """Build one page from a store snapshot that does not move during the read.

    ``produce(final)`` returns the store to read from. For the child route it
    refreshes an observer reader and can raise ``ObserverSnapshotChanged`` when
    the child transcript moves mid-refresh; a continuation (``has_end``) surfaces
    that move as the documented ``-32009`` stale error. A newest-page read
    retries acquisition and generation drift off-loop a bounded number of times,
    and its final attempt (``produce(True)``) returns a consistent snapshot to
    serve instead of raising.
    """
    if has_end:
        return read(produce(False))
    for attempt in range(_STABLE_HISTORY_ATTEMPTS):
        final = attempt == _STABLE_HISTORY_ATTEMPTS - 1
        try:
            store = produce(final)
        except ObserverSnapshotChanged:
            if final:
                raise
            continue
        generation = _generation(store)
        result = read(store)
        if generation == _generation(store) or final:
            return result
    raise AssertionError("unreachable")


def child_history(
    produce_reader: Callable[[bool], object], params: dict, *, fidelity: bool
) -> dict:
    """Serve one child history page from a stable observer snapshot.

    ``produce_reader(final)`` refreshes and returns the child reader store; the
    shared stable-read owner retries a validator-less page and serves a
    consistent snapshot, so only an end-bound continuation can return ``-32009``.
    """

    def read(reader: object) -> dict:
        runtime = SimpleNamespace(opened=SimpleNamespace(store=reader))
        return history(
            runtime, params, backward_enabled=True, fidelity=fidelity, child_runs=True
        )

    return _stable_history_read(
        produce_reader, read, has_end=params["before"] is not None
    )


def history_validator(
    expected: tuple[int, str | None, str | None] | None = None,
    *,
    supplied: object = _MISSING_VALIDATOR,
) -> dict[str, object]:
    emitted = (
        {
            "end": expected[0],
            "entry_id": expected[1],
            "repair_generation": expected[2],
        }
        if expected is not None
        else None
    )
    if supplied is _MISSING_VALIDATOR:
        assert emitted is not None
        return emitted
    if not isinstance(supplied, dict) or set(supplied) not in (
        {"end", "entry_id"},
        {"end", "entry_id", "repair_generation"},
    ):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "history_validator is invalid"
        )
    if (
        type(supplied.get("end")) is not int
        or supplied["end"] < 0
        or (
            supplied.get("entry_id") is not None
            and not isinstance(supplied.get("entry_id"), str)
        )
        or (
            "repair_generation" in supplied
            and supplied.get("repair_generation") is not None
            and not isinstance(supplied.get("repair_generation"), str)
        )
    ):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "history_validator is invalid"
        )
    if emitted is None:
        return dict(supplied)
    if supplied != emitted:
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "session history changed; restart from the newest page",
        )
    return emitted


def _projected_history_validator(
    projection: HistoryProjection,
    repair_generation: str | None,
    supplied: object,
) -> tuple[int, dict[str, object]]:
    current_end = len(projection.entry_groups) - 1
    if supplied is _MISSING_VALIDATOR:
        end = current_end
    else:
        supplied = history_validator(supplied=supplied)
        end = supplied["end"]
        if end > current_end:
            raise ProtocolError(
                ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
                "session history changed; restart from the newest page",
            )
    return end, history_validator(
        (end, _history_endpoint(projection, end), repair_generation),
        supplied=supplied,
    )


def _fit_history_groups(
    projection: HistoryProjection,
    indexes: range,
    *,
    end: int,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    backward: bool,
    validator: dict[str, object] | None = None,
    child_runs: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[int],
    list[tuple[dict[str, object], ...]],
    HistoryProjection,
]:
    groups = list(projection.groups)
    sizes = list(projection.body_sizes)
    cache_groups = list(projection.groups)
    cache_sizes = list(projection.body_sizes)
    rows: list[dict[str, object]] = []
    selected: list[int] = []
    response_groups: list[tuple[dict[str, object], ...]] = []
    body_size = 0
    codec = FrameCodec()
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    for index in indexes:
        while index >= len(groups):
            next_index = len(groups)
            group, size, forward, forward_size = _history_group(
                projection,
                next_index,
                end=end,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
                backward=backward,
                validator=validator,
            )
            cache_groups.append(forward)
            cache_sizes.append(forward_size)
            groups.append(group)
            sizes.append(size)
        cursor = index if backward else min(index + 1, end)
        raw_group = projection.entry_groups[index]
        reserve = child_run_lifecycle.baseline_reserve(groups[index]) if backward and child_runs else 0
        while True:
            fits, group, size = _history_group_fits(
                codec,
                request_id,
                groups[index],
                sizes[index],
                body_size=body_size + reserve,
                rows=rows,
                cursor=cursor,
                end=end,
                backward=backward,
                validator=validator,
                raw_group=raw_group,
                projection=projection,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
            )
            if fits or rows or not reserve:
                break
            reserve = 0
        if not fits:
            break
        groups[index], sizes[index] = group, size
        rows.extend(group)
        body_size += int(bool(rows[: -len(group)] and group)) + size + reserve if group else 0
        selected.append(index)
        response_groups.append(group)
    return (
        rows,
        selected,
        response_groups,
        HistoryProjection(
            projection.entry_groups,
            projection.approval_calls,
            projection.entry_index,
            tuple(cache_groups),
            tuple(cache_sizes),
            projection.frontier,
        ),
    )


def _forward_page_history(
    projection: HistoryProjection,
    offset: int,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    child_runs: bool = False,
) -> tuple[dict[str, object], HistoryProjection]:
    entry_count = len(projection.entry_groups) - 1
    if offset > entry_count:
        return {"messages": [], "next_offset": None}, projection
    stop = min(HISTORY_FIRST_PAGE_ENTRIES, entry_count) if offset == 0 else entry_count
    if stop == entry_count:
        stop += 1
    rows, selected, _response_groups, expanded = _fit_history_groups(
        projection,
        range(offset, stop),
        end=entry_count,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        backward=False,
        child_runs=child_runs,
    )
    next_offset = min(selected[-1] + 1, entry_count) if selected else offset
    return {
        "messages": rows,
        "next_offset": next_offset if next_offset < entry_count else None,
    }, expanded


def _backward_page_history(
    projection: HistoryProjection,
    before: int,
    validator: dict[str, object],
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    first_page: bool = False,
    child_runs: bool = False,
    snapshot_checked: bool = False,
) -> tuple[dict[str, object], HistoryProjection]:
    end = len(projection.entry_groups) - 1
    stable_end = validator["end"]
    if not snapshot_checked and (
        not isinstance(stable_end, int)
        or stable_end > end
        or validator.get("entry_id") != _history_endpoint(projection, stable_end)
    ):
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "session history changed; restart from the newest page",
        )
    cursor_end = end if snapshot_checked else stable_end
    if before < 0 or before > cursor_end + int(first_page):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "before must not exceed history end"
        )
    rows, selected, response_groups, expanded = _fit_history_groups(
        projection,
        range(before - 1, -1, -1),
        end=end,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        backward=True,
        validator=validator,
        child_runs=child_runs,
    )
    selected.reverse()
    response_groups.reverse()
    rows = [row for group in response_groups for row in group]
    boundary = selected[0] if selected else 0
    return {
        "messages": rows,
        "next_before": boundary if boundary > 0 else None,
        "history_validator": validator,
    }, expanded


def page_history(
    projection: HistoryProjection,
    cursor: int | None,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    backward: bool = False,
    validator: dict[str, object] | None = None,
    first_page: bool = False,
    child_runs: bool = False,
) -> tuple[dict[str, object], HistoryProjection]:
    assert cursor is not None
    if backward:
        assert validator is not None
        return _backward_page_history(
            projection,
            cursor,
            validator,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            first_page=first_page,
            child_runs=child_runs,
        )
    return _forward_page_history(
        projection,
        cursor,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        child_runs=child_runs,
    )


def _publish_projected_page(
    store: object,
    generation: tuple[int, bytes],
    result: dict,
    expanded: HistoryProjection,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
) -> dict:
    with _cache_creation_lock:
        cached = _cache(
            store,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
            child_runs=child_runs,
        )
        if (
            cached is not None
            and cached.generation == generation
            and len(cached.projection.groups) < len(expanded.groups)
        ):
            _publish_cache(
                store,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
                served_turn_end=served_turn_end,
                child_runs=child_runs,
                cached=_CachedProjection(generation, expanded, cached.closed),
            )
    return result


def _backward_history_page(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
    supplied_validator: object = _MISSING_VALIDATOR,
) -> tuple[dict, int, str | None]:
    before = params["before"]
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    )
    end, validator = _projected_history_validator(
        projection,
        store.repair_generation,
        supplied_validator
        if supplied_validator is not _MISSING_VALIDATOR
        else params.get("history_validator", _MISSING_VALIDATOR),
    )
    if before is None:
        cursor, first_page = end + 1, True
    else:
        cursor, first_page = before, False
    result, expanded = page_history(
        projection,
        cursor,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        backward=True,
        validator=validator,
        first_page=first_page,
        child_runs=child_runs,
    )
    return _publish_projected_page(
        store,
        generation,
        result,
        expanded,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    ), projection.frontier, projection.frontier_row_id


def replay_history(
    runtime: ServerRuntime,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
    supplied_validator: object = _MISSING_VALIDATOR,
) -> tuple[dict, int, str | None]:
    """Return the newest backward page and its stable frontier row."""
    store = runtime.opened.store
    return _stable_history_read(
        lambda _final: store,
        lambda _store: _backward_history_page(
            runtime,
            {"before": None},
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
            child_runs=child_runs,
            supplied_validator=supplied_validator,
        ),
        has_end=supplied_validator is not _MISSING_VALIDATOR,
    )


def _projected_forward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
) -> dict:
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    )
    result, expanded = page_history(
        projection,
        params.get("offset", 0),
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        child_runs=child_runs,
    )
    return _publish_projected_page(
        store,
        generation,
        result,
        expanded,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    )


def _projected_backward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
) -> dict:
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    )
    before = params["before"]
    end, validator = _projected_history_validator(
        projection,
        store.repair_generation,
        params.get("history_validator", _MISSING_VALIDATOR),
    )
    if before is None:
        cursor, first_page = end + 1, True
    else:
        cursor, first_page = before, False
    result, expanded = page_history(
        projection,
        cursor,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        backward=True,
        validator=validator,
        first_page=first_page,
        child_runs=child_runs,
    )
    result["history_validator"] = validator
    return _publish_projected_page(
        store,
        generation,
        result,
        expanded,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
        child_runs=child_runs,
    )


def _backward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
) -> dict:
    before = params["before"]
    validator = params.get("history_validator")
    requested_end = validator.get("end") if isinstance(validator, dict) else None
    store = runtime.opened.store
    read_page = None if child_runs else getattr(store, "history_index_page", None)
    try:
        page = read_page(
            {
                "backward": True,
                "before": before,
                **({"end": requested_end} if before is not None else {}),
            }
        ) if read_page is not None else None
    except InvalidHistoryPageRequest as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    if page is None:
        return _projected_backward_history_request(
            runtime,
            params,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
            child_runs=child_runs,
        )
    validator = history_validator(
        (page.total, page.endpoint, page.repair_generation),
        supplied=(
            _MISSING_VALIDATOR if before is None else params["history_validator"]
        ),
    )
    projection = history_projection_from_groups(
        page.groups, page.approval_calls, served_turn_end=served_turn_end
    )
    local_before = len(page.groups)
    result, _expanded = _backward_page_history(
        projection,
        local_before,
        validator,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        first_page=before is None,
        snapshot_checked=True,
    )
    boundary = result["next_before"]
    result["next_before"] = page.start + boundary if boundary is not None else (
        page.start if page.start > 0 else None
    )
    result["history_validator"] = validator
    return result


def _forward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    child_runs: bool = False,
) -> dict:
    store = runtime.opened.store
    read_page = None if child_runs else getattr(store, "history_index_page", None)
    offset = params.get("offset", 0)
    try:
        page = read_page({"offset": offset}) if read_page is not None else None
    except InvalidHistoryPageRequest as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    if page is None:
        return _projected_forward_history_request(
            runtime,
            params,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
            child_runs=child_runs,
        )
    projection = history_projection_from_groups(
        page.groups, page.approval_calls, served_turn_end=served_turn_end
    )
    result, _expanded = page_history(
        projection,
        0,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
    )
    local_next = result["next_offset"]
    consumed = len(page.groups) if local_next is None else local_next
    next_offset = page.start + consumed
    result["next_offset"] = next_offset if next_offset <= page.total else None
    return result


def _validate_history_params(params: dict) -> None:
    if set(params) - {"session_id", "before", "history_validator"}:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "invalid backward session history parameters",
        )
    before = params["before"]
    if before is not None and (type(before) is not int or before < 0):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "before must be null or a nonnegative integer",
        )
    validator = params.get("history_validator")
    if before is None and "history_validator" in params:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "history_validator is only valid for continuations",
        )
    if before is not None:
        history_validator(supplied=validator)


def history(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool = False,
    backward_enabled: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
    served_turn_end: bool = False,
    child_runs: bool = False,
) -> dict:
    backward_keys = {"before", "history_validator"}
    if backward_keys.intersection(params) and not backward_enabled:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "backward session history requires negotiation",
        )
    if "before" in params:
        _validate_history_params(params)
        return _stable_history_read(
            lambda _final: runtime.opened.store,
            lambda _store: _backward_history_request(
                runtime,
                params,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
                served_turn_end=served_turn_end,
                child_runs=child_runs,
            ),
            has_end=params["before"] is not None,
        )
    if backward_keys.intersection(params):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "invalid session history parameters"
        )
    offset = params.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "offset must be a nonnegative integer"
        )
    return _stable_history_read(
        lambda _final: runtime.opened.store,
        lambda _store: _forward_history_request(
            runtime,
            params,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
            child_runs=child_runs,
        ),
        has_end=False,
    )
