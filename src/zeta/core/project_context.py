"""Assemble the system prompt and project instruction files for one session.

The final system prompt has three ordered sources:

1. The user-editable ``~/.zeta/AGENTS.md`` identity + skill index from
   :mod:`zeta.prompts`. The file is seeded from the packaged identity when it
   is missing.
2. ``AGENTS.md`` files walked from ``cwd`` up to the repository root (or
   ``$HOME`` when outside a repository), concatenated with the nearest file
   last so deeper monorepo instructions override outer ones.
3. Optional user overrides from ``~/.zeta/SYSTEM.md`` /
   ``~/.zeta/APPEND_SYSTEM.md`` and the matching ``--system-prompt`` /
   ``--append-system-prompt`` CLI flags.

Override precedence:

- Within one channel, the CLI flag wins over the file.
- If a replacement override is present (flag or file), the base identity and
  the walked instruction files are dropped and the append override is ignored.
  This keeps a hand-authored system prompt intact.

Every walked file goes through :func:`html.escape` before it is wrapped in a
``<zeta-project-instructions>`` block, so raw ``</zeta-project-instructions>``
inside the file cannot break out of the container.

Home identity seeding uses a hard link; filesystems without hard-link support
fall back to the packaged identity.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from ..project_registry import ProjectRegistry, ProjectRegistryError
from ..prompts import load_identity, load_packaged_identity
from ..skills import SkillCatalog
from .process_env import subprocess_env

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
    # Structured location of the owned project-memory block within
    # ``system_prompt``.  Recorded at assembly time so a resume can replace the
    # owned component by offset instead of searching for a forgeable marker.
    memory_offset: int | None = None
    memory_length: int | None = None
    memory_project_id: str | None = None
    memory_digest: str | None = None
    has_override: bool = False


def _git_env() -> dict[str, str]:
    """Environment for discovery, never allowing ambient repository selection."""
    env = subprocess_env({"GIT_CONFIG_NOSYSTEM": "1"})
    for name in tuple(env):
        if name.startswith("GIT_") and name not in {"GIT_CONFIG_NOSYSTEM"}:
            del env[name]
    return env


_GIT_TIMEOUT = 2.0


def discover_project_root(cwd: str | Path | None = None) -> Path | None:
    """Resolve the git worktree root, or return None outside a repository."""

    directory = Path(cwd or Path.cwd()).expanduser().resolve()
    env = _git_env()
    try:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-C", str(directory), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, env=env, timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    root = getattr(result, "stdout", "").strip()
    if not root:
        return None
    # Linked worktrees report their own checkout as --show-toplevel.  The
    # common git directory identifies the main repository, which is the single
    # project identity shared by all worktrees.
    try:
        common = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(directory), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True, capture_output=True, text=True, env=env, timeout=_GIT_TIMEOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return Path(root).expanduser().resolve()
    common_path = Path(common).expanduser().resolve() if common else None
    if common_path is not None:
        try:
            listing = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-C", str(directory), "worktree", "list", "--porcelain"],
                check=True, capture_output=True, text=True, env=env, timeout=_GIT_TIMEOUT,
            ).stdout.splitlines()
            if listing and listing[0].startswith("worktree "):
                return Path(listing[0][len("worktree "):]).expanduser().resolve()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return Path(root).expanduser().resolve()


def discover_repo_root(cwd: str | Path | None = None) -> Path:
    """Resolve the git worktree root, or use cwd when it is not a repository."""

    directory = Path(cwd or Path.cwd()).expanduser().resolve()
    return discover_project_root(directory) or directory


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


def _load_home_identity(home: Path) -> tuple[str, str | None]:
    """Load or atomically seed the user-editable base identity."""

    path = home / AGENTS_FILENAME
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        packaged = load_packaged_identity()
        temporary: Path | None = None
        identity = packaged
        seed_notice: str | None = None
        try:
            home.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=home,
                prefix=f".{AGENTS_FILENAME}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(packaged)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                identity = path.read_text(encoding="utf-8")
        except OSError as exc:
            seed_notice = (
                f"context · could not seed {path}: {exc}; using packaged identity"
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_notice = f"context · could not clean up temporary file {temporary}: {exc}"
                    seed_notice = (
                        f"{seed_notice}; {cleanup_notice}"
                        if seed_notice is not None
                        else cleanup_notice
                    )
        return identity, seed_notice


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


PROJECT_MEMORY_START = "<zeta-project-memory>"
PROJECT_MEMORY_END = "</zeta-project-memory>"


def _render_memory_block(project_id: str, sections: list[str]) -> str:
    """Render the owned project-memory envelope.

    Even with no admitted sections this emits the full delimiter pair and the
    ``project-id`` line, so its exact encoded cost is known before any memory
    is admitted to the optional budget.
    """

    body = "\n\n".join(sections)
    return (
        PROJECT_MEMORY_START
        + "\nproject-id: "
        + project_id
        + "\n"
        + (body + "\n" if sections else "")
        + PROJECT_MEMORY_END
    )


def _owned_block_digest(block: str) -> str:
    """Hash the exact owned block so a refresh can confirm it before replacing.

    The digest guards the offsets recorded at assembly time against a
    persisted prompt that no longer matches; it never scans the flattened
    prompt for a marker, so a forged envelope in identity or append text can
    never be selected.
    """

    return hashlib.sha256(block.encode("utf-8")).hexdigest()


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
    catalog: SkillCatalog,
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

    home_identity, seed_notice = _load_home_identity(home)
    notices: list[str] = []
    if seed_notice is not None:
        notices.append(seed_notice)
    loaded: list[Path] = []
    memory_index: int | None = None
    memory_block: str | None = None
    memory_project_id: str | None = None

    if system_override is not None:
        sections: list[str] = [system_override]
    else:
        sections = [load_identity(catalog=catalog, identity=home_identity)]
        candidates: list[Path] = []
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
        home_agents = (home / AGENTS_FILENAME).resolve()
        for path in candidates:
            resolved = path.resolve()
            if resolved == home_agents:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)

        # Only optional walked instructions and project memory consume this
        # budget.  The home identity and append section are mandatory and are
        # never sliced to make room for optional context.
        optional_budget = max(0, byte_cap)
        instructions_bytes = 0
        skipped: list[Path] = []
        kept: list[tuple[Path, str]] = []
        for path in reversed(deduped):
            content = path.read_text(encoding="utf-8")
            section = _format_section(path, content)
            section_bytes = len(section.encode("utf-8"))
            if instructions_bytes + section_bytes > optional_budget:
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

        # Project memory is deliberately separate from transcripts and bounded.
        # A malformed or unsafe optional registry must not prevent a session from
        # starting; surface it as a startup notice instead.
        try:
            registry = ProjectRegistry(home / "projects")
            project = (
                registry.find_for_directory(working_dir)
                if registry.root.exists()
                else None
            )
            if project is not None:
                memory_sections: list[str] = []
                # Budget the complete envelope on every trial admission: the
                # delimiters, the ``project-id`` line, and every inter-section
                # separator are all counted, so admitted memory can never push
                # the owned block past the optional budget.
                budget_for_memory = max(0, optional_budget - instructions_bytes)
                empty_block = _render_memory_block(project.project_id, [])
                if len(empty_block.encode("utf-8")) > budget_for_memory:
                    # Not even the empty envelope fits.  Omit the owned block
                    # entirely and leave offset/length/digest unset rather than
                    # appending a zero-content envelope that overruns the cap.
                    notices.append(
                        f"context · project memory exceeded {byte_cap} byte cap; "
                        "omitted the memory block"
                    )
                else:
                    for name, content in registry.load_memory(project.project_id):
                        path = registry.root / project.project_id / "memory" / name
                        section = _format_section(path, content)
                        candidate = _render_memory_block(
                            project.project_id, memory_sections + [section]
                        )
                        if len(candidate.encode("utf-8")) <= budget_for_memory:
                            memory_sections.append(section)
                            loaded.append(path)
                        else:
                            notices.append(
                                f"context · project memory exceeded {byte_cap} byte cap; skipped {name}"
                            )
                    memory_block = _render_memory_block(
                        project.project_id, memory_sections
                    )
                    memory_index = len(sections)
                    memory_project_id = project.project_id
                    sections.append(memory_block)
        except (ProjectRegistryError, OSError) as exc:
            notices.append(f"context · project memory unavailable: {exc}")

        if system_append is not None:
            sections.append(system_append)

    # Deliberately do not slice the assembled prompt: mandatory identity and
    # append content must remain intact, while optional content was admitted
    # only after budgeting its complete encoded envelope.
    prompt = "\n\n".join(sections)
    memory_offset: int | None = None
    memory_length: int | None = None
    memory_digest: str | None = None
    if memory_index is not None and memory_block is not None:
        # Record the owned block's byte-less character span within the joined
        # prompt.  ``memory_index`` is never 0 (the identity section precedes
        # it), so the "\n\n" separator always contributes two characters.
        prefix = "\n\n".join(sections[:memory_index])
        memory_offset = len(prefix) + (2 if memory_index else 0)
        memory_length = len(memory_block)
        memory_digest = _owned_block_digest(memory_block)
    return ProjectContext(
        prompt,
        tuple(loaded),
        tuple(notices),
        memory_offset=memory_offset,
        memory_length=memory_length,
        memory_project_id=memory_project_id,
        memory_digest=memory_digest,
        has_override=system_override is not None or system_append is not None,
    )


def refresh_project_memory(
    system_prompt: str,
    *,
    home: Path,
    cwd: Path | None = None,
    project_id: str | None = None,
    memory_offset: int | None = None,
    memory_length: int | None = None,
    memory_digest: str | None = None,
) -> str:
    """Replace the single owned memory block by structured offset, never search.

    The owned block is located by the byte-less character span recorded at
    assembly time (``memory_offset`` / ``memory_length``), validated by the
    hash of the exact owned block (``memory_digest``).  The flattened prompt is
    never scanned for a marker pair, so a forged ``<zeta-project-memory>``
    envelope planted in identity or append text can never be selected.

    Sessions that predate structured components (missing any of offset, length,
    or digest) are left byte-identical -- the triple is treated as all-or-none
    and marker search is deliberately not attempted for them.  A ``project_id``
    of ``None`` also leaves the prompt unchanged.
    """
    del cwd  # runtime cwd must never re-select a project on resume
    if project_id is None:
        return system_prompt
    # The structured span is all-or-none: without every component (offset,
    # length, and the digest that proves the persisted prompt still matches)
    # there is nothing safe to replace, so the prompt is returned unchanged.
    if memory_offset is None or memory_length is None or memory_digest is None:
        return system_prompt
    start = memory_offset
    end = memory_offset + memory_length
    if start < 0 or memory_length < 0 or end > len(system_prompt):
        return system_prompt
    owned = system_prompt[start:end]
    if _owned_block_digest(owned) != memory_digest:
        return system_prompt
    try:
        registry = ProjectRegistry(home / "projects")
        project = registry.show_project(project_id)
        if project is None:
            return system_prompt
        entries = registry.load_memory(project.project_id)
    except (ProjectRegistryError, OSError):
        return system_prompt
    prefix = system_prompt[:start]
    suffix = system_prompt[end:]
    kept: list[str] = []
    for name, content in entries:
        section = _format_section(
            registry.root / project.project_id / "memory" / name, content
        )
        candidate = _render_memory_block(project.project_id, kept + [section])
        if len((prefix + candidate + suffix).encode()) > CONTEXT_BYTE_CAP:
            break
        kept.append(section)
    # Only the owned span is replaced; the prefix and suffix stay byte-identical,
    # so any forged envelope elsewhere in the prompt is preserved untouched.
    return prefix + _render_memory_block(project.project_id, kept) + suffix


__all__ = [
    "AGENTS_FILENAME",
    "APPEND_SYSTEM_FILENAME",
    "CLAUDE_FILENAME",
    "CONTEXT_BYTE_CAP",
    "SYSTEM_FILENAME",
    "ProjectContext",
    "PromptArgumentError",
    "discover_project_root",
    "discover_repo_root",
    "load_project_context",
    "resolve_prompt_argument",
]
