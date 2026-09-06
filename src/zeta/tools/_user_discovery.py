"""External tool discovery for ``~/.zeta/tools`` and ``<project>/.zeta/tools``.

Two scopes stacked over the built-in registry (registered by
:func:`~zeta.tools.registry._register_discovered_tools` during
``ToolRegistry.__init__``):

* ``user`` -- ``~/.zeta/tools/*.py``. Loaded and registered directly. A user
  tool that shadows a built-in or replaces one wins on name collision; a dim
  notice records the shadow so it is visible at session start. User modules
  run with the same trust as built-ins because the user planted them in their
  own home.

* ``project`` -- ``<project>/.zeta/tools/*.py``. TRUST BOUNDARY: this is code
  from a cloned repository. The modules are NOT imported at discovery time
  and are NOT registered. Session start records their file paths and shows a
  loud warning ("untrusted project tools present; run /tools trust"). The
  user runs ``/tools trust`` in-session to import and register them, at which
  point they participate exactly like user tools (same registry validation,
  approval policy, structured-result contract, and ZETA-73 allow/deny
  lists). This mirrors the ZETA-73 project-settings trust decision: a
  hostile checkout must not silently execute code or grant itself
  privileges. Session-scoped trust only; a durable ``/trust`` mechanism is a
  future ticket.

Precedence: project tools may replace USER tools on trust (loud "shadows
user tool" notice) but may NEVER replace BUILT-IN names -- an attempt is
rejected at the discovery seam with a loud REJECTED notice naming the
module and tool name, and the built-in stays intact. The rest of the
project module's tools still register. This closes the "trusted project
tool hijacks an always_allow built-in name" gap.

Failure modes fail OPEN: a malformed module (bad Python, missing/invalid
``register``, exception during registration, or import error) is reported
via a dim notice naming the file. Import errors are isolated per file so
one broken tool does not stop the rest. When a module's ``register()``
raises AFTER re-registering names that already existed, the prior
``ToolDefinition`` for each affected name is restored so a partial-failed
project module cannot silently unregister a pre-existing user tool.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolDefinition, ToolRegistry

TOOLS_DIRNAME = "tools"
USER_SCOPE = "user"
PROJECT_SCOPE = "project"

_MODULE_NAMESPACE = "_zeta_external_tools"


@dataclass(frozen=True, slots=True)
class PendingProjectTool:
    """A project-scoped tool module discovered but not yet trusted."""

    path: Path
    module_stem: str


class ExternalToolDiscovery:
    """Mutable per-session record of external tool discovery."""

    def __init__(
        self,
        *,
        notices: tuple[str, ...],
        warnings: tuple[str, ...],
        pending_project_tools: tuple[PendingProjectTool, ...],
        builtin_names: frozenset[str] = frozenset(),
        user_names: frozenset[str] = frozenset(),
    ) -> None:
        self._notices: list[str] = list(notices)
        self._warnings: list[str] = list(warnings)
        self._pending: list[PendingProjectTool] = list(pending_project_tools)
        self._builtin_names: frozenset[str] = builtin_names
        self._user_names: frozenset[str] = user_names

    @property
    def notices(self) -> tuple[str, ...]:
        return tuple(self._notices)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def pending_project_tools(self) -> tuple[PendingProjectTool, ...]:
        return tuple(self._pending)


def apply_external_tools(
    registry: ToolRegistry,
    *,
    home: str | Path | None,
    project_dir: str | Path | None,
) -> ExternalToolDiscovery:
    """Load ``~/.zeta/tools`` into ``registry`` and stage project-scope tools.

    ``home`` and ``project_dir`` follow the ZETA-73 convention: each is the
    directory that directly contains the scope's ``tools/`` folder
    (``~/.zeta`` and ``<repo>/.zeta`` in production). ``None`` disables the
    scope.
    """

    builtin_names = registry.registered_names
    notices: list[str] = []
    warnings: list[str] = []
    user_dir = _resolve_tools_dir(home)
    project_tools_dir = _resolve_tools_dir(project_dir)
    if user_dir is not None:
        user_added: set[str] = set()
        for path in _module_files(user_dir):
            added = _load_module_file(
                registry,
                path,
                USER_SCOPE,
                builtin_names=builtin_names,
                user_names=frozenset(user_added),
                other_names=frozenset(),
                other_label="",
                notices=notices,
            )
            user_added.update(added)
    pending: list[PendingProjectTool] = []
    if project_tools_dir is not None and project_tools_dir != user_dir:
        pending = _collect_pending_project_tools(project_tools_dir)
    if pending:
        names = ", ".join(sorted({tool.module_stem for tool in pending}))
        warnings.append(
            f"tools · untrusted project tools present: {names}; "
            "run /tools trust to enable them for this session"
        )
    user_names = frozenset(registry.registered_names - builtin_names)
    return ExternalToolDiscovery(
        notices=tuple(notices),
        warnings=tuple(warnings),
        pending_project_tools=tuple(pending),
        builtin_names=builtin_names,
        user_names=user_names,
    )


def trust_project_tools(
    registry: ToolRegistry,
    discovery: ExternalToolDiscovery,
) -> tuple[tuple[str, ...], tuple[PendingProjectTool, ...]]:
    """Import and register any pending project tools, returning notices.

    Returns ``(notices, trusted)`` where ``trusted`` lists the modules that
    the caller must now consider active. The discovery's pending list is
    cleared regardless of per-module success -- a malformed module is
    reported once and stays dropped.

    Project modules may NEVER register a name owned by a built-in tool: any
    such attempt is rejected with a loud notice and the rest of the module's
    tools register normally.
    """

    if not discovery._pending:
        return (), ()
    builtin_names = discovery._builtin_names
    user_names = discovery._user_names
    project_added: set[str] = set()
    notices: list[str] = []
    for pending in list(discovery._pending):
        added = _load_module_file(
            registry,
            pending.path,
            PROJECT_SCOPE,
            builtin_names=builtin_names,
            user_names=user_names,
            other_names=frozenset(project_added),
            other_label="project tool",
            notices=notices,
        )
        project_added.update(added)
    trusted = tuple(discovery._pending)
    discovery._pending.clear()
    return tuple(notices), trusted


def _resolve_tools_dir(base: str | Path | None) -> Path | None:
    if base is None:
        return None
    try:
        resolved = Path(base).expanduser().resolve(strict=False)
    except OSError:
        return None
    candidate = resolved / TOOLS_DIRNAME
    if not candidate.is_dir():
        return None
    return candidate


def _collect_pending_project_tools(
    tools_dir: Path,
) -> list[PendingProjectTool]:
    return [
        PendingProjectTool(path=path, module_stem=path.stem)
        for path in _module_files(tools_dir)
    ]


def _module_files(tools_dir: Path) -> list[Path]:
    files = []
    try:
        for path in sorted(tools_dir.iterdir()):
            if not path.is_file() or path.suffix != ".py":
                continue
            if path.name == "__init__.py" or path.name.startswith("_"):
                continue
            files.append(path)
    except OSError:
        return []
    return files


def _load_module_file(
    registry: ToolRegistry,
    path: Path,
    scope: str,
    *,
    builtin_names: frozenset[str],
    user_names: frozenset[str],
    other_names: frozenset[str],
    other_label: str,
    notices: list[str],
) -> tuple[str, ...]:
    display = _display_path(path)
    try:
        module = _import_module_from_file(path)
    except Exception as exc:  # noqa: BLE001 - discovery must fail open
        notices.append(
            f"tools · ignored {scope} tool {display}: import failed: {_short(exc)}"
        )
        return ()
    register = getattr(module, "register", None)
    if not callable(register):
        notices.append(
            f"tools · ignored {scope} tool {display}: no callable register(registry)"
        )
        return ()
    added: list[str] = []
    snapshots: dict[str, ToolDefinition | None] = {}
    rejected_builtins: list[str] = []
    reject_builtins = scope == PROJECT_SCOPE
    original_register = registry.register

    def spy(name: str, *args: object, **kwargs: object) -> object:
        if reject_builtins and name in builtin_names:
            if name not in rejected_builtins:
                rejected_builtins.append(name)
            return None
        snapshots.setdefault(name, registry._tools.get(name))
        added.append(name)
        return original_register(name, *args, **kwargs)

    registry.register = spy  # type: ignore[method-assign]
    registry.register_tool = spy  # type: ignore[method-assign]
    register_error: BaseException | None = None
    try:
        register(registry)
    except Exception as exc:  # noqa: BLE001 - discovery must fail open
        register_error = exc
        for name, prior in snapshots.items():
            if prior is None:
                registry._tools.pop(name, None)
            else:
                registry._tools[name] = prior
    finally:
        try:
            del registry.register
        except AttributeError:
            pass
        try:
            del registry.register_tool
        except AttributeError:
            pass
    for name in rejected_builtins:
        notices.append(
            f"tools · REJECTED {scope} tool '{name}' from {display}: "
            "cannot replace built-in tool"
        )
    if register_error is not None:
        notices.append(
            f"tools · ignored {scope} tool {display}: "
            f"register() raised: {_short(register_error)}"
        )
        return ()
    if not added and not rejected_builtins:
        notices.append(
            f"tools · {scope} tool {display} registered no tools"
        )
        return ()
    for name in added:
        phrase = _shadow_phrase(
            name,
            builtin_names=builtin_names,
            user_names=user_names,
            other_names=other_names,
            other_label=other_label,
        )
        if phrase is None:
            continue
        notices.append(
            f"tools · {scope} tool '{name}' from {display} shadows {phrase}"
        )
    return tuple(added)


def _shadow_phrase(
    name: str,
    *,
    builtin_names: frozenset[str],
    user_names: frozenset[str],
    other_names: frozenset[str],
    other_label: str,
) -> str | None:
    if name in builtin_names:
        return "built-in"
    if name in user_names:
        return "user tool"
    if other_label and name in other_names:
        return other_label
    return None


def _import_module_from_file(path: Path) -> object:
    module_name = f"{_MODULE_NAMESPACE}.{path.stem}_{abs(hash(str(path))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _display_path(path: Path) -> str:
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return str(path)
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def _short(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    text = text.replace("\n", " ")
    if len(text) > 200:
        text = text[:200] + "…"
    return text


__all__ = [
    "PROJECT_SCOPE",
    "TOOLS_DIRNAME",
    "USER_SCOPE",
    "ExternalToolDiscovery",
    "PendingProjectTool",
    "apply_external_tools",
    "trust_project_tools",
]
