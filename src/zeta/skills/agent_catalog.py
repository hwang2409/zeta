"""Discover and load markdown sub-agent definitions for one session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..agent.presets import AGENT_PRESETS, AgentPreset
from ..models.catalog import known_model_names
from .discovery import (
    MarkdownDocument,
    contained_path,
    discover_markdown,
    discover_session_items,
    read_markdown,
    snapshot_metadata,
    warn_discovery,
)

AgentMeta = AgentPreset

_CLAUDE_TOOL_NAMES = {
    "Read": "read",
    "Edit": "edit",
    "Write": "write",
    "Bash": "bash",
    "WebFetch": "fetch",
    "WebSearch": "websearch",
    "TodoWrite": "todo",
}
_ZETA_TOOL_NAMES = frozenset(
    {
        "agent",
        "agent_output",
        "agent_status",
        "bash",
        "edit",
        "fetch",
        "read",
        "skill",
        "todo",
        "websearch",
        "write",
    }
)


@dataclass(frozen=True, slots=True)
class AgentCatalog:
    """The immutable agent definitions mounted into one session."""

    agents: tuple[AgentPreset, ...]
    notices: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> AgentCatalog:
        return cls(())

    def find(self, name: str) -> AgentPreset:
        for agent in self.agents:
            if agent.name == name:
                return agent
        available = ", ".join(agent.name for agent in self.agents) or "none"
        raise ValueError(f"unknown agent preset {name!r}; available presets: {available}")

    def names(self) -> list[str]:
        return [agent.name for agent in self.agents]

    def description(self) -> str:
        choices = "; ".join(
            f"{agent.name}: {agent.selection_guidance}" for agent in self.agents
        )
        return f"Choose one of: {choices}." if choices else "No agent presets are available."

    def to_snapshot(self) -> list[dict[str, object]]:
        """Return agent metadata without embedding prompt bodies."""

        snapshots = []
        for agent in self.agents:
            snapshot = snapshot_metadata(
                name=agent.name,
                description=agent.selection_guidance,
                source=agent.source,
                path=agent.path,
                root_name="agents_root",
                root=agent.agents_root,
            )
            snapshot.update(
                {
                    "turn_cap": agent.turn_cap,
                    "tools": list(agent.tool_names)
                    if agent.tool_names is not None
                    else None,
                    "model": agent.model,
                    "preamble": agent.preamble,
                }
            )
            snapshots.append(snapshot)
        return snapshots

    @classmethod
    def from_snapshot(cls, value: object) -> AgentCatalog:
        if type(value) is not list:
            raise ValueError("agent catalog snapshot must be a list")
        agents: list[AgentPreset] = []
        for item in value:
            if type(item) is not dict:
                raise ValueError("agent catalog snapshot entries must be mappings")
            name = item.get("name")
            description = item.get("description")
            turn_cap = item.get("turn_cap")
            tools = item.get("tools")
            model = item.get("model")
            preamble = item.get("preamble", "")
            source = item.get("source", "")
            path = item.get("path")
            agents_root = item.get("agents_root")
            if agents_root is None and type(path) is str:
                agents_root = str(Path(path).parent)
            if (
                type(name) is not str
                or type(description) is not str
                or type(turn_cap) is not int
                or turn_cap < 1
                or (
                    tools is not None
                    and (
                        type(tools) is not list
                        or any(type(tool) is not str for tool in tools)
                    )
                )
                or (model is not None and type(model) is not str)
                or type(preamble) is not str
                or type(source) is not str
                or (path is not None and type(path) is not str)
                or (agents_root is not None and type(agents_root) is not str)
            ):
                raise ValueError("agent catalog snapshot entry is invalid")
            agents.append(
                AgentPreset(
                    name=name,
                    turn_cap=turn_cap,
                    tool_names=frozenset(tools) if tools is not None else None,
                    preamble=preamble,
                    prompt_suffix="",
                    selection_guidance=description,
                    model=model,
                    source=source,
                    path=Path(path) if path is not None else None,
                    agents_root=Path(agents_root) if agents_root is not None else None,
                )
            )
        return cls(tuple(agents))


def _agents_dir(root: Path) -> Path:
    return root / "agents"


def _parse_tools(
    value: object, path: Path, notices: list[str] | None
) -> frozenset[str] | None:
    if value is None:
        return None
    if type(value) is str:
        names = [name.strip() for name in value.split(",")]
    elif type(value) is list and all(type(name) is str for name in value):
        names = [name.strip() for name in value]
    else:
        raise ValueError(f"agent {path} frontmatter tools must be a list or string")
    if not names or any(not name for name in names):
        raise ValueError(f"agent {path} frontmatter tools must contain names")

    supported: set[str] = set()
    for name in names:
        zeta_name = _CLAUDE_TOOL_NAMES.get(name)
        if zeta_name is None and name in _ZETA_TOOL_NAMES:
            zeta_name = name
        if zeta_name is None:
            notice = warn_discovery(
                path,
                f"unknown tool name {name!r}; skipped from allowlist",
                "agent",
            )
            if notices is not None:
                notices.append(notice)
            continue
        supported.add(zeta_name)
    return frozenset(supported)


def _build_agent(
    document: MarkdownDocument, source: str, notices: list[str] | None
) -> AgentPreset:
    metadata = document.metadata
    name = metadata["name"]
    description = metadata["description"]
    model = metadata.get("model")
    if model is not None and (type(model) is not str or not model.strip()):
        raise ValueError(f"agent {document.path} frontmatter model must be a nonempty string")
    if model is not None and model not in known_model_names():
        notice = warn_discovery(
            document.path,
            f"unknown model {model!r}; agent will inherit the parent model",
            "agent",
        )
        if notices is not None:
            notices.append(notice)
        model = None
    assert isinstance(name, str)
    assert isinstance(description, str)
    return AgentPreset(
        name=name,
        turn_cap=AGENT_PRESETS["general"].turn_cap,
        tool_names=_parse_tools(metadata.get("tools"), document.path, notices),
        preamble="",
        prompt_suffix=document.body,
        selection_guidance=description,
        model=model if isinstance(model, str) else None,
        source=source,
        path=document.entry_path.absolute(),
        agents_root=document.root,
    )


def discover_agents(
    root: Path,
    *,
    source: str,
    notices: list[str] | None = None,
    agents_dir: Path | None = None,
) -> list[AgentPreset]:
    """Discover one tier of flat agent definitions."""

    agents_dir = agents_dir or _agents_dir(root)
    return discover_markdown(
        agents_dir,
        source=source,
        kind="agent",
        build=lambda document: _build_agent(document, source, notices),
        notices=notices,
    )


def discover_session_agents(
    *, home: str | Path | None = None, project_dir: str | Path | None = None
) -> AgentCatalog:
    """Discover packaged, home, and project agents for one session."""

    agents, notices = discover_session_items(
        home=home,
        project_dir=project_dir,
        packaged=lambda _notices: [
            replace(preset, source="packaged") for preset in AGENT_PRESETS.values()
        ],
        discover=lambda root, source, notices: discover_agents(
            root, source=source, notices=notices
        ),
        warn_override=lambda agent: warn_discovery(
            agent.path or Path(agent.name),
            "custom definition overrides packaged preset",
            "agent",
        ),
    )
    return AgentCatalog(agents, notices)


def discover_packaged_agents() -> AgentCatalog:
    return discover_session_agents()


def load_agent(meta: AgentPreset) -> str:
    """Load an agent body at execution time from its pinned path."""

    if meta.path is None:
        return meta.prompt_suffix
    root = meta.agents_root or meta.path.parent
    resolved = contained_path(meta.path, root, "agent")
    try:
        _, body = read_markdown(resolved, "agent")
    except FileNotFoundError as exc:
        raise ValueError(f"agent definition {meta.path} no longer exists") from exc
    return body


__all__ = [
    "AgentCatalog",
    "AgentMeta",
    "discover_agents",
    "discover_packaged_agents",
    "discover_session_agents",
    "load_agent",
]
