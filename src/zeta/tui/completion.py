"""Prompt-toolkit completion for slash commands and filesystem paths."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from ..core.commands.path_completion import complete_paths
from ..core.slash import SlashCommandRegistry
from ..skills.invocation import dollar_completion_prefix
from .models import match_models


class SlashCompleter(Completer):
    """Complete slash commands, and model names after ``/model``."""

    def __init__(
        self,
        registry: SlashCommandRegistry,
        *,
        model_choices: Callable[[], Sequence[str]] | None = None,
        current_model: Callable[[], str] | None = None,
    ) -> None:
        self.registry = registry
        self._model_choices = model_choices
        self._current_model = current_model

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        before_cursor = document.text_before_cursor
        if not before_cursor.startswith("/"):
            return
        if any(character.isspace() for character in before_cursor):
            yield from self._model_completions(before_cursor)
            return
        prefix = before_cursor[1:]
        for name, description, source in self.registry.completion_entries:
            if not name.startswith(prefix):
                continue
            meta = description
            if source:
                meta = f"[{source}] {description}".strip()
            yield Completion(
                name,
                start_position=-len(prefix),
                display=f"/{name}",
                display_meta=meta,
            )

    def _model_completions(self, before_cursor: str) -> Iterator[Completion]:
        """Offer provider models while the argument to ``/model`` is being typed."""

        if self._model_choices is None or "\n" in before_cursor:
            return
        parts = before_cursor[1:].split(maxsplit=1)
        argument = parts[1] if len(parts) == 2 else ""
        if parts[0] != "model" or any(character.isspace() for character in argument):
            return
        current = self._current_model() if self._current_model is not None else None
        for name in match_models(argument, self._model_choices()):
            yield Completion(
                name,
                start_position=-len(argument),
                display=name,
                display_meta="current" if name == current else "",
            )


class DollarSkillCompleter(Completer):
    """Complete only registered skills for an active ``$`` token."""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self.registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        prefix = dollar_completion_prefix(document.text_before_cursor)
        if prefix is None:
            return
        for skill in self.registry.skill_entries:
            if not skill.name.startswith(prefix):
                continue
            meta = skill.description
            if skill.source:
                meta = f"[{skill.source}] {skill.description}".strip()
            yield Completion(
                f"{skill.name} ",
                start_position=-len(prefix),
                display=f"${skill.name}",
                display_meta=meta,
            )


class PathCompleter(Completer):
    """Complete one filesystem directory level for an ``@`` token."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        rows, _truncated = complete_paths(
            document.text_before_cursor, self.base_dir
        )
        for row in rows:
            yield Completion(
                row.text,
                start_position=row.start_position,
                display=row.display,
                display_meta=row.kind,
            )


class ComposerCompleter(Completer):
    """Route slash commands and ``@`` paths to their focused completer."""

    def __init__(
        self,
        registry: SlashCommandRegistry,
        base_dir: str | Path | None = None,
        *,
        model_choices: Callable[[], Sequence[str]] | None = None,
        current_model: Callable[[], str] | None = None,
    ) -> None:
        self.slash = SlashCompleter(
            registry, model_choices=model_choices, current_model=current_model
        )
        self.dollar_skill = DollarSkillCompleter(registry)
        self.path = PathCompleter(base_dir)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        before_cursor = document.text_before_cursor
        if before_cursor.startswith("/"):
            yield from self.slash.get_completions(document, complete_event)
        yield from self.dollar_skill.get_completions(document, complete_event)
        yield from self.path.get_completions(document, complete_event)


__all__ = [
    "ComposerCompleter",
    "DollarSkillCompleter",
    "PathCompleter",
    "SlashCompleter",
]
