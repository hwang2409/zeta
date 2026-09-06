"""Assemble the system prompt and project instruction files for one session.

The final system prompt has three ordered sources:

1. The packaged identity + skill index from :mod:`zeta.prompts`.
2. ``AGENTS.md`` files walked from ``cwd`` up to the repository root (or
   ``$HOME`` when outside a repository), concatenated with the nearest file
   last so deeper monorepo instructions override outer ones.
3. Optional user overrides from ``~/.zeta/SYSTEM.md`` /
   ``~/.zeta/APPEND_SYSTEM.md`` and the matching ``--system-prompt`` /
   ``--append-system-prompt`` CLI flags.

Override precedence:

- Within one channel, the CLI flag wins over the file.
- If a replacement override is present (flag or file), the packaged identity
  and the walked instruction files are dropped and the append override is
  ignored. This keeps a hand-authored system prompt intact.

Every walked file goes through :func:`html.escape` before it is wrapped in a
``<zeta-project-instructions>`` block, so raw ``</zeta-project-instructions>``
inside the file cannot break out of the container.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from ..prompts import load_identity

AGENTS_FILENAME = "AGENTS.md"
CLAUDE_FILENAME = "CLAUDE.md"
SYSTEM_FILENAME = "SYSTEM.md"
APPEND_SYSTEM_FILENAME = "APPEND_SYSTEM.md"

# Cap the walked instruction bytes to keep the cached system prefix bounded.
# 512 KiB is roughly 128k tokens — comfortably below any provider context
# window, and well above realistic AGENTS.md totals.
CONTEXT_BYTE_CAP = 512 * 1024


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """The composed system prompt, the files that fed it, and any notices."""

    system_prompt: str
    files: tuple[Path, ...]
    notices: tuple[str, ...] = field(default_factory=tuple)


def discover_repo_root(cwd: str | Path | None = None) -> Path:
    """Resolve the git worktree root, or use cwd when it is not a repository."""

    directory = Path(cwd or Path.cwd()).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return directory
    root = getattr(result, "stdout", "").strip()
    return Path(root).expanduser().resolve() if root else directory


def _present(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


class PromptArgumentError(ValueError):
    """The ``--system-prompt`` / ``--append-system-prompt`` value cannot be loaded."""


def resolve_prompt_argument(value: str | None) -> str | None:
    """Turn a ``TEXT-OR-@FILE`` CLI value into a plain string.

    - ``None`` passes through untouched (the flag was not supplied).
    - ``"@/absolute/path"`` or ``"@relative/path"`` reads the file's UTF-8
      content and returns it. Missing files raise
      :class:`PromptArgumentError` so the CLI can surface a clean error
      instead of failing during session start.
    - Any other value is returned verbatim as the prompt text.
    - ``"@"`` alone or an empty value is a caller error.
    """

    if value is None:
        return None
    if not value:
        raise PromptArgumentError("system-prompt value must be nonempty")
    if not value.startswith("@"):
        return value
    reference = value[1:]
    if not reference:
        raise PromptArgumentError("system-prompt @path is missing a filename")
    path = Path(reference).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptArgumentError(f"system-prompt file not found: {path}") from exc
    except OSError as exc:
        raise PromptArgumentError(
            f"system-prompt file unreadable: {path}: {exc}"
        ) from exc


def _format_section(path: Path, content: str) -> str:
    source = escape(str(path), quote=True)
    return (
        f'<zeta-project-instructions source="{source}">\n'
        f"{escape(content, quote=True)}\n"
        "</zeta-project-instructions>"
    )


def _default_stop_at(cwd: Path, home: Path) -> Path:
    """Bound the walk at ``$HOME`` when cwd is inside it; otherwise cwd."""

    try:
        cwd.relative_to(home)
    except ValueError:
        return cwd
    return home


def _walk_up_agents_files(cwd: Path, stop_at: Path) -> list[Path]:
    """Return AGENTS.md files from the outermost bound in to cwd, nearest-last."""

    # Only walk when cwd is inside the bound. When cwd sits outside stop_at
    # (mismatched caller, or repo_root != cwd's actual ancestor), fall back
    # to just cwd so we neither leak upward nor silently pick nothing.
    try:
        cwd.relative_to(stop_at)
    except ValueError:
        dirs = [cwd]
    else:
        dirs = []
        current = cwd
        while True:
            dirs.append(current)
            if current == stop_at:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
        dirs.reverse()
    files: list[Path] = []
    for directory in dirs:
        agents = directory / AGENTS_FILENAME
        if _present(agents):
            files.append(agents)
    return files


def load_project_context(
    *,
    cwd: str | Path | None = None,
    repo_root: str | Path | None = None,
    zeta_home: str | Path | None = None,
    system_override: str | None = None,
    system_append: str | None = None,
    byte_cap: int = CONTEXT_BYTE_CAP,
) -> ProjectContext:
    """Load the composed system prompt for one session.

    ``system_override`` and ``system_append`` come from CLI flags after any
    ``@file`` shorthand has been resolved by the caller. When either is
    ``None`` this function falls back to ``~/.zeta/SYSTEM.md`` and
    ``~/.zeta/APPEND_SYSTEM.md`` respectively; the resulting content is used
    as-is (no further escape, since the user is authoring their own prompt).
    """

    working_dir = Path(cwd or Path.cwd()).expanduser().resolve()
    home = Path(zeta_home or Path.home() / ".zeta").expanduser().resolve()
    if repo_root is not None:
        stop_at = Path(repo_root).expanduser().resolve()
    else:
        stop_at = _default_stop_at(working_dir, Path.home().expanduser().resolve())

    if system_override is None:
        system_override = _read_optional(home / SYSTEM_FILENAME)
    if system_append is None:
        system_append = _read_optional(home / APPEND_SYSTEM_FILENAME)

    notices: list[str] = []
    loaded: list[Path] = []

    if system_override is not None:
        sections: list[str] = [system_override]
    else:
        sections = [load_identity()]
        candidates: list[Path] = []
        home_agents = home / AGENTS_FILENAME
        if _present(home_agents):
            candidates.append(home_agents)
        candidates.extend(_walk_up_agents_files(working_dir, stop_at))
        # Always attempt to include the repository-root AGENTS.md (or fall
        # back to CLAUDE.md for repos that never migrated). When cwd is
        # inside stop_at the walk already picked this up; the dedupe below
        # drops the duplicate. When cwd sits outside stop_at (unusual, but
        # supported for tests and edge callers), this keeps the anchor.
        stop_agents = stop_at / AGENTS_FILENAME
        if _present(stop_agents):
            candidates.append(stop_agents)
        else:
            claude = stop_at / CLAUDE_FILENAME
            if _present(claude):
                candidates.append(claude)

        seen: set[Path] = set()
        deduped: list[Path] = []
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)

        # The cap governs walked instruction bytes so a long packaged
        # identity does not push every file over. Iterate nearest-first so
        # the cap drops OUTER (more general) files and preserves the
        # nearest (most-specific) instructions per the documented
        # precedence.
        instructions_bytes = 0
        skipped: list[Path] = []
        kept: list[tuple[Path, str]] = []
        for path in reversed(deduped):
            content = path.read_text(encoding="utf-8")
            section = _format_section(path, content)
            section_bytes = len(section.encode("utf-8"))
            if instructions_bytes + section_bytes > byte_cap:
                skipped.append(path)
                continue
            kept.append((path, section))
            instructions_bytes += section_bytes
        for path, section in reversed(kept):
            sections.append(section)
            loaded.append(path)
        if skipped:
            joined = ", ".join(str(path) for path in reversed(skipped))
            notices.append(
                f"context · walked instructions exceeded {byte_cap} byte cap; "
                f"skipped {joined}"
            )

        if system_append is not None:
            sections.append(system_append)

    return ProjectContext(
        "\n\n".join(sections),
        tuple(loaded),
        tuple(notices),
    )


__all__ = [
    "AGENTS_FILENAME",
    "APPEND_SYSTEM_FILENAME",
    "CLAUDE_FILENAME",
    "CONTEXT_BYTE_CAP",
    "SYSTEM_FILENAME",
    "ProjectContext",
    "PromptArgumentError",
    "discover_repo_root",
    "load_project_context",
    "resolve_prompt_argument",
]
