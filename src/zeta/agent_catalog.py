"""Discover and load markdown sub-agent definitions for one session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from .model_catalog import known_model_names
from .skills.catalog import is_slash_safe_name
from .tools.agent_presets import AGENT_PRESETS, AgentPreset

AgentMeta = AgentPreset

_logger = logging.getLogger(__name__)
_PACKAGED_AGENT_NAMES = frozenset(AGENT_PRESETS)


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
        return [
            {
                "name": agent.name,
                "description": agent.selection_guidance,
                "turn_cap": agent.turn_cap,
                "tools": list(agent.tool_names) if agent.tool_names is not None else None,
                "model": agent.model,
                "preamble": agent.preamble,
                "prompt_suffix": agent.prompt_suffix,
                "source": agent.source,
                "path": str(agent.path) if agent.path is not None else None,
                "agents_root": (
                    str(agent.agents_root) if agent.agents_root is not None else None
                ),
            }
            for agent in self.agents
        ]

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
            preamble = item.get("preamble")
            prompt_suffix = item.get("prompt_suffix", "")
            source = item.get("source", "")
            path = item.get("path")
            agents_root = item.get("agents_root")
            if (
                type(name) is not str
                or type(description) is not str
                or type(turn_cap) is not int
                or turn_cap < 1
                or (tools is not None and (
                    type(tools) is not list
                    or any(type(tool) is not str for tool in tools)
                ))
                or (model is not None and type(model) is not str)
                or type(preamble) is not str
                or type(prompt_suffix) is not str
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
                    prompt_suffix=prompt_suffix,
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


def _contained_path(path: Path, agents_root: Path) -> Path:
    try:
        resolved = path.resolve()
        resolved.relative_to(agents_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"agent document {path} resolves outside agents root {agents_root}"
        ) from exc
    return resolved


def _warn(path: Path, error: Exception | str) -> str:
    message = f"ignored agent {path}: {error}"
    _logger.warning(message)
    return message


def _read(path: Path) -> tuple[dict[str, object], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"agent {path} is missing YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"agent {path} has unterminated YAML frontmatter") from exc
    try:
        values = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise ValueError(f"agent {path} has invalid YAML frontmatter") from exc
    if not isinstance(values, dict):
        raise ValueError(  # noqa: TRY004 — malformed metadata is skipped
            f"agent {path} frontmatter must be a mapping"
        )
    missing = [key for key in ("name", "description") if key not in values]
    if missing:
        raise ValueError(f"agent {path} frontmatter is missing: {', '.join(missing)}")
    name = values["name"]
    if type(name) is not str or not name.strip():
        raise ValueError(f"agent {path} frontmatter name must be a nonempty string")
    if not is_slash_safe_name(name):
        raise ValueError(f"agent {path} frontmatter name must be tool-argument-safe")
    description = values["description"]
    if type(description) is not str or not description.strip():
        raise ValueError(f"agent {path} frontmatter description must be a nonempty string")
    tools = values.get("tools")
    if tools is not None and (
        type(tools) is not list
        or any(type(tool) is not str or not tool.strip() for tool in tools)
    ):
        raise ValueError(f"agent {path} frontmatter tools must be a list of strings")
    model = values.get("model")
    if model is not None and (type(model) is not str or not model.strip()):
        raise ValueError(f"agent {path} frontmatter model must be a nonempty string")
    return {
        "name": name,
        "description": description,
        "tools": tools,
        "model": model,
    }, "\n".join(lines[end + 1 :]).strip()


def discover_agents(
    root: Path,
    *,
    source: str,
    notices: list[str] | None = None,
    agents_dir: Path | None = None,
) -> list[AgentPreset]:
    agents_dir = agents_dir or _agents_dir(root)
    if not agents_dir.is_dir():
        return []
    try:
        agents_root = agents_dir.resolve()
        paths = sorted(agents_dir.glob("*.md"), key=lambda path: path.name)
    except (OSError, RuntimeError, UnicodeError) as exc:
        notice = _warn(agents_dir, exc)
        if notices is not None:
            notices.append(notice)
        return []
    discovered: list[AgentPreset] = []
    paths_by_name: dict[str, Path] = {}
    for path in paths:
        try:
            resolved = _contained_path(path, agents_root)
            metadata, body = _read(resolved)
        except (OSError, UnicodeError, ValueError, RecursionError, yaml.YAMLError) as exc:
            notice = _warn(path, exc)
            if notices is not None:
                notices.append(notice)
            continue
        name = metadata["name"]
        assert isinstance(name, str)
        previous = paths_by_name.get(name)
        if previous is not None:
            raise ValueError(f"duplicate agent name {name!r} in {previous} and {path}")
        paths_by_name[name] = path
        model = metadata["model"]
        if model is not None and model not in known_model_names():
            notice = _warn(path, f"unknown model {model!r}; agent will inherit the parent model")
            if notices is not None:
                notices.append(notice)
            model = None
        tools = metadata["tools"]
        assert tools is None or isinstance(tools, list)
        description = metadata["description"]
        assert isinstance(description, str)
        discovered.append(
            AgentPreset(
                name=name,
                turn_cap=AGENT_PRESETS["general"].turn_cap,
                tool_names=frozenset(tools) if tools is not None else None,
                preamble="",
                prompt_suffix=body,
                selection_guidance=description,
                model=model if isinstance(model, str) else None,
                source=source,
                path=path.absolute(),
                agents_root=agents_root,
            )
        )
    return discovered


def discover_session_agents(
    *, home: str | Path | None = None, project_dir: str | Path | None = None
) -> AgentCatalog:
    """Discover packaged, home, and project agents for one session."""

    notices: list[str] = []
    packaged = [
        AgentPreset(
            name=preset.name,
            turn_cap=preset.turn_cap,
            tool_names=preset.tool_names,
            preamble=preset.preamble,
            selection_guidance=preset.selection_guidance,
            model=preset.model,
            prompt_suffix=preset.prompt_suffix,
            source="packaged",
        )
        for preset in AGENT_PRESETS.values()
    ]
    tiers: list[list[AgentPreset]] = [packaged]
    if home is not None:
        tiers.append(discover_agents(Path(home), source="home", notices=notices))
    if project_dir is not None:
        tiers.append(
            discover_agents(
                Path(project_dir) / ".zeta",
                source="project",
                notices=notices,
            )
        )
    for tier in tiers[1:]:
        for agent in tier:
            if agent.name in _PACKAGED_AGENT_NAMES:
                notices.append(
                    _warn(agent.path or Path(agent.name), "custom definition overrides packaged preset")
                )
    selected: dict[str, AgentPreset] = {}
    for tier in tiers:
        for agent in tier:
            selected[agent.name] = agent
    agents = tuple(
        agent for tier in tiers for agent in tier if selected[agent.name] is agent
    )
    return AgentCatalog(agents, tuple(notices))


def discover_packaged_agents() -> AgentCatalog:
    return discover_session_agents()


def load_agent(meta: AgentPreset) -> str:
    if meta.path is None:
        return meta.prompt_suffix
    root = meta.agents_root or meta.path.parent
    return _read(_contained_path(meta.path, root))[1]


__all__ = [
    "AgentCatalog",
    "AgentMeta",
    "discover_agents",
    "discover_packaged_agents",
    "discover_session_agents",
    "load_agent",
]
