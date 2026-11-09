"""Serve session_history paging and request handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..protocol import MAX_REQUEST_ID_BYTES, ErrorDescriptor, FrameCodec, ProtocolError
from ..runtime import ServerRuntime
from .history_projection import (
    HistoryProjection,
    _cache,
    _cache_creation_lock,
    _CachedProjection,
    _generation,
    _history_endpoint,
    _history_group,
    _history_group_fits,
    _publish_cache,
    history_projection,
)

if TYPE_CHECKING:
    from ...core.store import ConversationStore

HISTORY_FIRST_PAGE_ENTRIES = 128


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
        fits, group, size = _history_group_fits(
            codec,
            request_id,
            groups[index],
            sizes[index],
            body_size=body_size,
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
        if not fits:
            break
        groups[index], sizes[index] = group, size
        rows.extend(group)
        body_size += int(bool(rows[: -len(group)] and group)) + size if group else 0
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
        ),
    )


def _forward_page_history(
    projection: HistoryProjection,
    offset: int,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
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
) -> tuple[dict[str, object], HistoryProjection]:
    end = len(projection.entry_groups) - 1
    stable_end = validator["end"]
    if (
        not isinstance(stable_end, int)
        or stable_end > end
        or validator.get("entry_id") != _history_endpoint(projection, stable_end)
    ):
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "session history changed; restart from the newest page",
        )
    if before < 0 or before > stable_end + int(first_page):
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
        )
    return _forward_page_history(
        projection,
        cursor,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
    )


def _publish_history_page(
    store: ConversationStore,
    generation: tuple[int, bytes],
    result: dict,
    expanded: HistoryProjection,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
) -> dict:
    with _cache_creation_lock:
        cached = _cache(
            store,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
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
                cached=_CachedProjection(generation, expanded),
            )
    return result


def _backward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
) -> dict:
    before = params["before"]
    validator = params.get("history_validator")
    if before is None:
        validator = {"end": 0, "entry_id": None}
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
    )
    end = len(projection.entry_groups) - 1
    if before is None:
        validator = {"end": end, "entry_id": _history_endpoint(projection, end)}
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
    )
    if _generation(store) != generation:
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "session history changed; restart from the newest page",
        )
    return _publish_history_page(
        store,
        generation,
        result,
        expanded,
        include_delivery_id,
        fidelity,
        content_read,
        served_turn_end,
    )


def _forward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
) -> dict:
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
    )
    result, expanded = page_history(
        projection,
        params.get("offset", 0),
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
    )
    if _generation(store) != generation:
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "session history changed; restart from the newest page",
        )
    return _publish_history_page(
        store,
        generation,
        result,
        expanded,
        include_delivery_id,
        fidelity,
        content_read,
        served_turn_end,
    )


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
    if before is not None and (
        not isinstance(validator, dict)
        or set(validator) != {"end", "entry_id"}
        or type(validator.get("end")) is not int
        or validator.get("end") < 0
        or (
            validator.get("entry_id") is not None
            and not isinstance(validator.get("entry_id"), str)
        )
    ):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "history_validator is invalid"
        )


def history(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool = False,
    backward_enabled: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
    served_turn_end: bool = False,
) -> dict:
    backward_keys = {"before", "history_validator"}
    if backward_keys.intersection(params) and not backward_enabled:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "backward session history requires negotiation",
        )
    if "before" in params:
        _validate_history_params(params)
        return _backward_history_request(
            runtime,
            params,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
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
    return _forward_history_request(
        runtime,
        params,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
    )
