"""Load and render user-defined prompt and execution commands."""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

COMMAND_FILE_SIZE_LIMIT = 64 * 1024
INLINE_SHELL_RE = re.compile(r"!`([^`\n]+)`")

@dataclass(frozen=True, slots=True)
class InlineShellResult:
    """Result of preprocessing inline shell spans."""

    outputs: tuple[str, ...]
    canceled: bool = False


InlineShellRunner = Callable[
    [tuple[str, ...]], Awaitable[tuple[str, ...] | InlineShellResult]
]


@dataclass(frozen=True, slots=True)
class CustomCommand:
    """One prompt or exec command loaded from a markdown file."""

    name: str
    description: str
    body: str
    path: Path
    source: str
    kind: str = "prompt"
    timeout: float = 300.0
    background: bool = False

    def render(self, arguments: str) -> str:
        """Substitute the raw argument tail and positional arguments."""

        values = arguments.split()

        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if token == "ARGUMENTS":
                return arguments
            index = int(token)
            return values[index - 1] if index <= len(values) else ""

        return re.sub(r"\$(ARGUMENTS|[1-9])(?!\d)", replace, self.body)

    def render_exec(self, arguments: str) -> str:
        """Build a shell invocation with arguments kept outside the script."""

        argv = arguments.split()
        return " ".join(
            (
                f"ARGUMENTS={shlex.quote(arguments)}",
                "sh -c",
                shlex.quote(self.body),
                "zeta-macro",
                *(shlex.quote(value) for value in argv),
            )
        )


@dataclass(frozen=True, slots=True)
class CommandLoadResult:
    """Commands and non-fatal notices found during one load."""

    commands: tuple[CustomCommand, ...]
    notices: tuple[str, ...]


def render_custom_input(
    value: str, commands: Mapping[str, CustomCommand]
) -> str:
    """Expand a custom prompt command or turn an escaped slash literal."""

    if value.startswith("//"):
        return value[1:]
    first_line = value.split("\n", 1)[0]
    if not first_line.startswith("/"):
        return value
    parts = first_line[1:].split(maxsplit=1)
    if not parts:
        return value
    command = commands.get(parts[0])
    if command is None:
        return value
    arguments = parts[1] if len(parts) == 2 else ""
    return command.render(arguments)


def _prompt_command(
    value: str, commands: Mapping[str, CustomCommand]
) -> CustomCommand | None:
    first_line = value.split("\n", 1)[0]
    if not first_line.startswith("/") or first_line.startswith("//"):
        return None
    parts = first_line[1:].split(maxsplit=1)
    command = commands.get(parts[0]) if parts else None
    return command if command is not None and command.kind == "prompt" else None


def needs_inline_shell_resolution(
    value: str, commands: Mapping[str, CustomCommand]
) -> bool:
    """Return whether model resolution can wait for shell approval."""

    command = _prompt_command(value, commands)
    return command is not None and bool(
        INLINE_SHELL_RE.search(render_custom_input(value, commands))
    )


async def resolve_custom_input(
    value: str,
    commands: Mapping[str, CustomCommand],
    inline_shell: InlineShellRunner,
) -> str | None:
    """Expand a prompt command and resolve its inline shell spans."""

    rendered = render_custom_input(value, commands)
    if _prompt_command(value, commands) is None:
        return rendered
    matches = tuple(INLINE_SHELL_RE.finditer(rendered))
    if not matches:
        return rendered
    resolution = await inline_shell(tuple(match.group(1) for match in matches))
    if isinstance(resolution, InlineShellResult):
        if resolution.canceled:
            return None
        replacements = resolution.outputs
    else:
        replacements = resolution
    if len(replacements) != len(matches):
        raise ValueError("inline shell resolver returned the wrong result count")
    replacement_iter = iter(replacements)
    return INLINE_SHELL_RE.sub(lambda _match: next(replacement_iter), rendered)


def load_custom_commands(
    *,
    home: str | Path | None,
    project_dir: str | Path,
) -> CommandLoadResult:
    """Load project commands over home commands without touching other homes."""

    commands: list[CustomCommand] = []
    notices: list[str] = []
    directories: list[tuple[str, Path]] = []
    if home is not None:
        directories.append(("home", Path(home).resolve() / "commands"))
    directories.append(("project", Path(project_dir).resolve() / ".zeta" / "commands"))
    for source, directory in directories:
        for path in _markdown_files(directory, notices):
            command, notice = _read_command(path, source)
            if notice is not None:
                notices.append(notice)
            if command is not None:
                commands.append(command)
    return CommandLoadResult(tuple(commands), tuple(notices))


def _markdown_files(directory: Path, notices: list[str]) -> list[Path]:
    try:
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.md"))
    except OSError as exc:
        notices.append(f"could not scan custom command directory {directory}: {exc}")
        return []


def _read_command(path: Path, source: str) -> tuple[CustomCommand | None, str | None]:
    try:
        if path.stat().st_size > COMMAND_FILE_SIZE_LIMIT:
            raise ValueError(f"file exceeds {COMMAND_FILE_SIZE_LIMIT} byte limit")
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_document(text, path)
        name = path.stem
        if not name or any(character.isspace() for character in name):
            raise ValueError("filename stem must be one nonempty word")
        description = metadata.get("description", "")
        if type(description) is not str:
            raise ValueError("description must be a string")
        kind = metadata.get("kind", "prompt")
        if type(kind) is not str or kind not in {"prompt", "exec"}:
            raise ValueError(f"unknown command kind: {kind!r}")
        timeout_value = 300.0
        background = False
        if kind == "exec":
            timeout = metadata.get("timeout", 300.0)
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError, OverflowError):
                timeout_value = 0.0
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout_value)
                or timeout_value <= 0
            ):
                raise ValueError("timeout must be a positive finite number")
            background_value = metadata.get("background", False)
            if type(background_value) is not bool:
                raise ValueError("background must be a boolean")
            background = background_value
        if not body:
            raise ValueError("prompt body is empty")
    except (
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
        yaml.YAMLError,
    ) as exc:
        return None, f"ignored custom command {path}: {exc}"
    return CustomCommand(
        name,
        description.strip(),
        body,
        path.resolve(),
        source,
        kind,
        timeout_value,
        background,
    ), None


def _split_document(text: str, path: Path) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        try:
            end = next(
                index
                for index, line in enumerate(lines[1:], 1)
                if line.strip() == "---"
            )
        except StopIteration as exc:
            raise ValueError("frontmatter is unterminated") from exc
        metadata = yaml.safe_load("\n".join(lines[1:end]))
        if not isinstance(metadata, dict):
            raise ValueError(f"frontmatter in {path} must be a mapping")
        body = "\n".join(lines[end + 1 :]).strip()
        return metadata, body
    return {}, text.strip()
