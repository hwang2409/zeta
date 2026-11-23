"""Controller writes of skill and agent profile files through the one library owner."""

from __future__ import annotations

from ....memory.safety import REDACTED_SECRET, redact_secrets
from ...resource_protocol import (
    Access,
    ResourceMethod,
    require_params,
    require_revision,
)
from .library import project_root, valid_name

FEATURE = "library_write_v1"


def _handler(*, profiles, delete):
    async def handle(client, request_id, params):
        from ....skills import resource_store
        from ...protocol import (
            ErrorDescriptor,
            closed_error,
            invalid_resource_params,
            resource_error,
            stale_resource_revision,
        )

        if delete:
            require_params(params, "name", "scope", "revision")
        else:
            require_params(params, "name", "scope", "mutation", optional=("revision",))
            mutation = params["mutation"]
            if not isinstance(mutation, dict) or mutation.keys() != {"document"} or not isinstance(mutation["document"], str):
                raise invalid_resource_params()
            # A read masks secrets; saving that masked text back would replace the real secret.
            if REDACTED_SECRET in mutation["document"]:
                raise invalid_resource_params(reason="contains_redaction_marker")
        name, scope, revision = params["name"], params["scope"], params.get("revision")
        if revision is not None:
            require_revision(revision)
        if not valid_name(name):
            raise invalid_resource_params()
        if scope == "effective":  # the catalog's scope for packaged items
            raise invalid_resource_params(reason="packaged_read_only")
        if scope not in ("home", "project"):
            raise invalid_resource_params()
        if scope == "home":
            base = client.server.home
        else:
            project = project_root(client)
            if project is None:
                raise closed_error(ErrorDescriptor.STATE_READ_UNAVAILABLE)
            base = project / ".zeta"
        root = base / ("agents" if profiles else "skills")

        def run():
            try:
                if delete:
                    resource_store.delete(root, profiles=profiles, name=name, revision=revision)
                    return {"data": {"name": name, "scope": scope}}
                document = mutation["document"]
                written = resource_store.write(
                    root, profiles=profiles, name=name, revision=revision, document=document,
                )
            except resource_store.LibraryError as exc:
                if exc.code == "stale_revision":
                    raise stale_resource_revision("library") from None
                if exc.code == "busy":
                    raise resource_error(ErrorDescriptor.RESOURCE_BUSY) from None
                detail = redact_secrets(exc.detail) if exc.detail is not None else None
                raise invalid_resource_params(reason=exc.code, detail=detail) from None
            # The body is the user's own file and is stored as written; it never comes back.
            return {"revision": written, "data": {
                "name": name, "scope": scope, "redacted": redact_secrets(document) != document,
            }}

        return await client.connections.run_blocking(run)

    return handle


METHODS = (
    ResourceMethod("skill_write", Access.CONTROLLER_WRITE, _handler(profiles=False, delete=False)),
    ResourceMethod("skill_delete", Access.CONTROLLER_WRITE, _handler(profiles=False, delete=True)),
    ResourceMethod("agent_profile_write", Access.CONTROLLER_WRITE, _handler(profiles=True, delete=False)),
    ResourceMethod("agent_profile_delete", Access.CONTROLLER_WRITE, _handler(profiles=True, delete=True)),
)
