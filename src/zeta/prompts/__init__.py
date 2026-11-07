"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from importlib.resources import files

from ..runtime.prompt import PromptFragment, owned_span
from ..skills import SkillCatalog


def load_identity_fragment(
    *,
    catalog: SkillCatalog,
    identity: str | None = None,
) -> PromptFragment:
    """Render identity text with its generated skill index ownership."""

    content = load_packaged_identity() if identity is None else identity
    prefix = f"{content.rstrip()}\n\n"
    index = catalog.index()
    text = f"{prefix}{index}\n"
    return PromptFragment(text, {"skill_index": owned_span(text, len(prefix), len(index))})


def load_identity(
    *,
    catalog: SkillCatalog,
    identity: str | None = None,
) -> str:
    """Load identity content with the skill index for one session."""

    return load_identity_fragment(catalog=catalog, identity=identity).text


def load_packaged_identity() -> str:
    """Load the packaged identity without the session skill index."""

    return files(__package__).joinpath("identity.md").read_text(encoding="utf-8")


def load_runtime_guidance() -> str:
    """Load harness-owned guidance that applies to every composed prompt."""

    return files(__package__).joinpath("runtime.md").read_text(encoding="utf-8")


def load_skill(name: str, *, catalog: SkillCatalog) -> str:
    """Load one skill from a session catalog."""

    return catalog.load(name)


__all__ = [
    "load_identity",
    "load_identity_fragment",
    "load_packaged_identity",
    "load_runtime_guidance",
    "load_skill",
]
