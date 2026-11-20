"""Read-only projections of the skill and agent profile catalogs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ....agent.presets import AgentPreset
from ....memory.safety import REDACTED_SECRET, redact_secrets
from ....project_schema import MAX_MEMORY_FILE_SIZE
from ....read_page import TooLarge, offset_page
from ....skills import agent_catalog, catalog
from ....skills.discovery import is_slash_safe_name
from ...resource_protocol import Access, ResourceMethod, require_params

FEATURE = "library_read_v1"
_SCOPES = ("effective", "home", "project")
_BODY_BYTES = MAX_MEMORY_FILE_SIZE  # same 128 KiB cap as project memory text


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _items(home, project, scope, *, profiles, capture=None):
    owner = agent_catalog if profiles else catalog
    if scope == "effective":
        if profiles:
            return owner.discover_session_agents(home=home, project_dir=project, capture=capture).agents
        return owner.discover_session_skills(home=home, project_dir=project, capture=capture).skills
    root = home if scope == "home" else project / ".zeta"
    discover = owner.discover_agents if profiles else owner.discover_skills
    return discover(root, source=scope, capture=capture)


def _fields(item):
    fields = {"name": item.name, "description": item.description}
    if isinstance(item, AgentPreset):
        fields.update(
            tools=item.tools, model=item.model, allow_delegation=item.allow_delegation,
            accepts_follow_ups=item.accepts_follow_ups, explicit_finish=item.explicit_finish,
            max_turns=item.max_turns,
        )
    else:
        fields["keywords"] = item.keywords
    return fields


def _public_text(value):
    """Redact YAML/JSON string fields, bounding YAML alias expansion on the wire."""
    from ...protocol import MAX_FRAME_BYTES, ErrorDescriptor, closed_error

    redacted = False
    nodes = size = 0

    def text(value):
        nonlocal redacted, size
        size += len(value)
        if size > MAX_FRAME_BYTES:
            raise closed_error(ErrorDescriptor.RESULT_TOO_LARGE)
        safe = redact_secrets(value)
        redacted |= safe != value
        return safe

    def visit(value, depth=0):
        nonlocal nodes, redacted
        nodes += 1
        if depth > 32 or nodes > 4096:
            raise closed_error(ErrorDescriptor.RESULT_TOO_LARGE)
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            return [visit(item, depth + 1) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                raw_key = str(key)
                safe_key = text(raw_key)
                probe = f"{raw_key}: value"
                if redact_secrets(probe) != probe:
                    redacted = True
                    result[safe_key] = REDACTED_SECRET
                else:
                    result[safe_key] = visit(item, depth + 1)
            return result
        if value is None or type(value) in (bool, int) or (type(value) is float and math.isfinite(value)):
            return value
        return text(str(value))

    return visit(value), redacted


def _catalog(items, filenames, offset, limit, fits):
    from ...protocol import ErrorDescriptor, closed_error

    def project(item):
        fields, _ = _public_text(_fields(item))
        filename = filenames.get((item.source, item.name))
        return {
            **fields, "source": item.source, "editable": item.source != "packaged",
            "scope": "effective" if item.source == "packaged" else item.source,
            "filename": redact_secrets(filename) if filename is not None else None,
            "diagnostics": [],
        }

    def shape(page, next_offset):
        data = {"items": page}
        if next_offset is not None:
            data["next_offset"] = next_offset
        return {"revision": _digest(data), "data": data}

    try:
        return offset_page(sorted(items, key=lambda item: item.name), offset, limit,
                           project=project, fits=fits, shape=shape)
    except TooLarge:
        raise closed_error(ErrorDescriptor.RESULT_TOO_LARGE) from None


def _detail(item, document, fits):
    from ...protocol import ErrorDescriptor, closed_error, invalid_resource_params

    if item is None:
        raise invalid_resource_params(reason="resource_not_found")
    metadata, body = (
        (document.metadata, document.body) if document is not None
        else (_fields(item), item.prompt_suffix or item.preamble)
    )
    safe_body = redact_secrets(body)
    data, redacted = _public_text({
        "name": item.name, "frontmatter": metadata,
        "source": item.source, "editable": item.source != "packaged",
    })
    data["redacted"] = redacted or safe_body != body
    data["body"] = safe_body
    encoded = safe_body.encode()
    if len(encoded) > _BODY_BYTES:
        data["body"] = encoded[:_BODY_BYTES].decode(errors="ignore")
        data["truncated"] = True
    revision = hashlib.sha256(document.raw).hexdigest() if document is not None else _digest(data)
    result = {"revision": revision, "data": data}
    if not fits(result):
        raise closed_error(ErrorDescriptor.RESULT_TOO_LARGE)
    return result


def _handler(*, profiles, detail):
    async def handle(client, request_id, params):
        from ...protocol import ErrorDescriptor, closed_error, invalid_resource_params

        if detail:
            require_params(params, "name", "scope")
            name = params["name"]
            if (
                not isinstance(name, str) or not is_slash_safe_name(name)
                or name in (".", "..") or "\\" in name or "\x00" in name
            ):
                raise invalid_resource_params()
        else:
            require_params(params, optional=("scope", "offset", "limit"))
            name = None
        scope = params.get("scope", "effective")
        offset, limit = params.get("offset", 0), params.get("limit", 50)
        if scope not in _SCOPES or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise invalid_resource_params()
        home = client.server.home
        opened = client.server.runtime.opened
        project = Path(opened.metadata.cwd) if opened is not None and scope != "home" else None
        if scope == "project" and project is None:
            raise closed_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)

        def fits(result):
            return client.codec.response_fits(request_id, result)

        def read():
            documents = {}
            filenames = {}

            def capture(item, document):
                key = (item.source, item.name)
                if name is None:
                    filenames[key] = document.path.relative_to(document.root).as_posix()
                elif item.name == name:
                    documents[item.source] = document

            items = _items(home, project, scope, profiles=profiles, capture=capture)
            if name is None:
                return _catalog(items, filenames, offset, limit, fits)
            item = next((item for item in items if item.name == name), None)
            return _detail(item, documents.get(item.source) if item is not None else None, fits)

        return await client.connections.run_blocking(read)

    return handle


METHODS = (
    ResourceMethod("skill_catalog", Access.READ, _handler(profiles=False, detail=False)),
    ResourceMethod("skill_read", Access.READ, _handler(profiles=False, detail=True)),
    ResourceMethod("agent_profile_catalog", Access.READ, _handler(profiles=True, detail=False)),
    ResourceMethod("agent_profile_read", Access.READ, _handler(profiles=True, detail=True)),
)
