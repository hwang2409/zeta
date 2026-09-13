"""The model may save and inspect drafts, but can never arm them."""

from __future__ import annotations

from ...automations.authoring import listing, resolve_job, show
from ...automations.store import SQLiteStore
from ...core.session import SessionError, SessionManager, env_home
from ...types import StructuredToolResult
from ..registry import ToolRegistry, text_block


async def _automation(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    home = env_home()
    try:
        with SQLiteStore(home) as store:
            action = arguments.get("action")
            name = arguments.get("name")
            if action == "list":
                text = listing(store)
            elif action == "show" and isinstance(name, str):
                text = show(store, name)
            elif action == "draft" and isinstance(name, str):
                provider = model = None
                try:
                    metadata = (
                        SessionManager(home)
                        .open(registry.session_store.session_id)
                        .metadata
                    )
                    provider, model = metadata.provider, metadata.model
                except SessionError:
                    pass
                job = resolve_job(
                    name,
                    arguments.get("job"),
                    cwd=str(registry.cwd),
                    home=home,
                    provider=provider,
                    model=model,
                )
                state = store.draft(job)
                text = (
                    show(store, name)
                    + f"\nSaved inert draft r{state.revision}. The human must run /automations approve {name}."
                )
            else:
                raise ValueError(
                    "use list, show with name, or draft with name and job; arming is human-only"
                )
        return {
            "content": [text_block(text)],
            "isError": False,
            "structuredContent": None,
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {
            "content": [text_block(str(exc))],
            "isError": True,
            "structuredContent": None,
        }


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "automation",
        _automation,
        requires_approval=False,
        description=(
            "Set up a recurring agent run when the user asks for something on a "
            "schedule (\"every morning\", \"every 5 minutes\", \"each weekday\"). "
            "draft saves the job; list and show inspect existing ones. Drafts are "
            "inert and cannot run: the human arms an exact revision with "
            "/automations approve, which is where tool permissions and the Slack "
            "recipient are granted. Tell the user to run it after drafting. "
            "Editing a job suspends future runs until it is approved again."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "show", "draft"]},
                "name": {"type": "string"},
                "job": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "trigger": {
                            "type": "object",
                            "description": "Schedule: kind=schedule, cron=five numeric fields, timezone=IANA zone (default America/Toronto). Poll: kind=poll, condition=natural language, interval_seconds>=300 (default 300).",
                        },
                        "servers": {"type": "array", "items": {"type": "string"}},
                        "allow": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact tool or tool(subject glob); MCP names are server__tool.",
                        },
                        "deliver": {
                            "type": "string",
                            "description": "slack:@name, slack:#channel, or slack:<Slack ID>",
                        },
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "cwd": {"type": "string"},
                    },
                    "required": ["prompt", "trigger", "servers", "allow", "deliver"],
                    "additionalProperties": False,
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    )
