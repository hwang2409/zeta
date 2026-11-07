"""Discover and load markdown sub-agent definitions for one session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..agent.presets import AGENT_PRESETS, AgentPreset
from ..models.catalog import known_model_names
from .discovery import (
    MarkdownDocument,
    discover_markdown,
    discover_session_items,
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


@dataclass(frozen=True, slots=True)
class AgentCatalog:
    """The immutable agent definitions mounted into one session."""

    agents: tuple[AgentPreset, ...]
    notices: tuple[str, ...] = ()
    bodies_committed: bool = True

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
        """Return complete agent definitions from one catalog generation."""

        if not self.bodies_committed:
            raise ValueError("agent catalog does not contain committed profile bodies")
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
                    "tools": list(agent.tool_names)
                    if agent.tool_names is not None
                    else None,
                    "model": agent.model,
                    "preamble": agent.preamble,
                    "prompt_suffix": agent.prompt_suffix,
                    "allow_delegation": agent.allow_delegation,
                    "accepts_follow_ups": agent.accepts_follow_ups,
                    "explicit_finish": agent.explicit_finish,
                }
            )
            snapshots.append(snapshot)
        return snapshots

    @classmethod
    def from_snapshot(cls, value: object) -> AgentCatalog:
        if type(value) is not list:
            raise ValueError("agent catalog snapshot must be a list")
        agents: list[AgentPreset] = []
        bodies_committed = True
        for item in value:
            if type(item) is not dict:
                raise ValueError("agent catalog snapshot entries must be mappings")
            name = item.get("name")
            description = item.get("description")
            tools = item.get("tools")
            model = item.get("model")
            preamble = item.get("preamble", "")
            if "prompt_suffix" in item:
                prompt_suffix = item["prompt_suffix"]
            else:
                prompt_suffix = ""
                bodies_committed = False
            allow_delegation = item.get("allow_delegation", True)
            accepts_follow_ups = item.get("accepts_follow_ups", False)
            explicit_finish = item.get("explicit_finish", False)
            source = item.get("source", "")
            path = item.get("path")
            agents_root = item.get("agents_root")
            if agents_root is None and type(path) is str:
                agents_root = str(Path(path).parent)
            if (
                type(name) is not str
                or type(description) is not str
                or (
                    tools is not None
                    and (
                        type(tools) is not list
                        or any(type(tool) is not str for tool in tools)
                    )
                )
                or (model is not None and type(model) is not str)
                or type(preamble) is not str
                or type(prompt_suffix) is not str
                or type(allow_delegation) is not bool
                or type(accepts_follow_ups) is not bool
                or type(explicit_finish) is not bool
                or type(source) is not str
                or (path is not None and type(path) is not str)
                or (agents_root is not None and type(agents_root) is not str)
            ):
                raise ValueError("agent catalog snapshot entry is invalid")
            agents.append(
                AgentPreset(
                    name=name,
                    tool_names=frozenset(tools) if tools is not None else None,
                    preamble=preamble,
                    prompt_suffix=prompt_suffix,
                    selection_guidance=description,
                    model=model,
                    source=source,
                    path=Path(path) if path is not None else None,
                    agents_root=Path(agents_root) if agents_root is not None else None,
                    allow_delegation=allow_delegation,
                    accepts_follow_ups=accepts_follow_ups,
                    explicit_finish=explicit_finish,
                )
            )
        return cls(tuple(agents), bodies_committed=bodies_committed)


def _agents_dir(root: Path) -> Path:
    return root / "agents"


def _parse_tools(value: object, path: Path) -> frozenset[str] | None:
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

    return frozenset(_CLAUDE_TOOL_NAMES.get(name, name) for name in names)


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
    allow_delegation = metadata.get("allow_delegation", True)
    if type(allow_delegation) is not bool:
        raise ValueError(
            f"agent {document.path} frontmatter allow_delegation must be a boolean"
        )
    accepts_follow_ups = metadata.get("accepts_follow_ups", False)
    if type(accepts_follow_ups) is not bool:
        raise ValueError(
            f"agent {document.path} frontmatter accepts_follow_ups must be a boolean"
        )
    explicit_finish = metadata.get("explicit_finish", False)
    if type(explicit_finish) is not bool:
        raise ValueError(
            f"agent {document.path} frontmatter explicit_finish must be a boolean"
        )
    assert isinstance(name, str)
    assert isinstance(description, str)
    return AgentPreset(
        name=name,
        tool_names=_parse_tools(metadata.get("tools"), document.path),
        preamble="",
        prompt_suffix=document.body,
        selection_guidance=description,
        model=model if isinstance(model, str) else None,
        path=document.entry_path.absolute(),
        source=source,
        agents_root=document.root,
        allow_delegation=allow_delegation,
        accepts_follow_ups=accepts_follow_ups,
        explicit_finish=explicit_finish,
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


__all__ = [
    "AgentCatalog",
    "AgentMeta",
    "discover_agents",
    "discover_packaged_agents",
    "discover_session_agents",
]
