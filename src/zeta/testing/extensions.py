"""Author test kit for zeta extensions.

Lets an extension author load their extension exactly as a session would
(through the real :func:`~zeta.extensions.discovery.apply_extensions` path) and
call the tools it registers, without standing up a whole session::

    from zeta.testing.extensions import load_extension

    async with load_extension("path/to/my-extension") as kit:
        result = await kit.call_tool("my_tool", {"q": "hi"})
        assert result["isError"] is False

A project-scope extension is staged but not activated until ``kit.trust()`` is
called, mirroring the real trust boundary.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..extensions.discovery import (
    USER_SCOPE,
    apply_extensions,
    trust_project_extensions,
)
from ..extensions.host import PROJECT_SCOPE, ExtensionHost
from ..protocol.types import StructuredToolResult, ToolCall
from ..skills import SkillCatalog
from ..tools import ToolRegistry


@dataclass(slots=True)
class ExtensionKit:
    """A loaded extension plus the registry its tools landed in."""

    host: ExtensionHost
    registry: ToolRegistry
    _root: Path

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def trust(self) -> None:
        """Activate a staged project-scope extension (the trust action)."""

        trust_project_extensions(self.host)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> StructuredToolResult:
        """Execute a registered tool and return its structured result."""

        call = ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments or {})
        return await self.registry.execute(call)

    async def aclose(self) -> None:
        await self.registry.close()
        shutil.rmtree(self._root, ignore_errors=True)


def load_extension(
    extension_dir: str | Path,
    *,
    scope: str = USER_SCOPE,
    skill_catalog: SkillCatalog | None = None,
) -> ExtensionKit:
    """Load one extension directory into a fresh registry and return a kit.

    The directory is placed under a throwaway scope root so discovery sees it
    exactly as it would in ``~/.zeta/extensions`` (user) or
    ``<project>/.zeta/extensions`` (project). Use the kit as an async context
    manager (or ``await kit.aclose()``) to close the registry and remove the
    throwaway root.
    """

    source = Path(extension_dir).expanduser().resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="zeta-ext-kit-"))
    cwd = root / "cwd"
    cwd.mkdir()
    registry = ToolRegistry(cwd, skill_catalog=skill_catalog or SkillCatalog.empty())
    builtin_names = registry.registered_names

    if scope == PROJECT_SCOPE:
        home = root / "home"
        home.mkdir()
        extensions_dir = root / "project" / ".zeta" / "extensions"
        project_dir: Path | None = root / "project"
    else:
        home = root / "home"
        extensions_dir = home / "extensions"
        project_dir = None
    extensions_dir.mkdir(parents=True)
    shutil.copytree(source, extensions_dir / source.name)

    host = apply_extensions(
        registry,
        home=home,
        project_dir=project_dir,
        builtin_names=builtin_names,
    )
    return ExtensionKit(host=host, registry=registry, _root=root)


__all__ = ["ExtensionKit", "load_extension"]
