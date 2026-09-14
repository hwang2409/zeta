"""Prompt-toolkit completion for slash commands and filesystem paths."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from ..core.slash import SlashCommandRegistry

PATH_COMPLETION_LIMIT = 50
PATH_SCAN_LIMIT = 1000
_PATH_TOKEN_RE = re.compile(
    r'(?<!\S)@(?:"(?P<quoted>[^"\n]*)|(?P<unquoted>[^\s]*))$'
)


class SlashCompleter(Completer):
    """Complete slash commands with descriptions and custom-source badges."""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self.registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        before_cursor = document.text_before_cursor
        if not before_cursor.startswith("/") or any(
            character.isspace() for character in before_cursor
        ):
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


class PathCompleter(Completer):
    """Complete one filesystem directory level for an ``@`` token."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        match = _PATH_TOKEN_RE.search(document.text_before_cursor)
        if match is None:
            return
        quoted = match.group("quoted") is not None
        path_text = match.group("quoted") or match.group("unquoted") or ""
        directory, display_prefix, basename = self._completion_context(path_text)
        show_hidden = basename.startswith(".")

        try:
            entries = os.scandir(directory)
        except OSError:
            return
        with entries:
            matches: list[tuple[bool, str]] = []
            for index, entry in enumerate(entries):
                if index >= PATH_SCAN_LIMIT:
                    break
                name = entry.name
                if not name.startswith(basename) or (
                    name.startswith(".") and not show_hidden
                ):
                    continue
                try:
                    is_directory = entry.is_dir()
                except OSError:
                    continue
                matches.append((is_directory, name))

        matches.sort(key=lambda item: (not item[0], item[1]))
        for is_directory, name in matches[:PATH_COMPLETION_LIMIT]:
            candidate = self._candidate(display_prefix, name, is_directory)
            completion_text = self._quoted_candidate(candidate, quoted, is_directory)
            yield Completion(
                completion_text,
                start_position=-len(path_text),
                display=f"@{completion_text}",
                display_meta="directory" if is_directory else "file",
            )

    def _completion_context(self, path_text: str) -> tuple[Path, str, str]:
        if path_text == "~":
            return Path.home(), "~/", ""
        slash = path_text.rfind("/")
        if slash < 0:
            return self.base_dir, "", path_text
        display_prefix = path_text[: slash + 1]
        directory_text = path_text[:slash] or "/"
        directory = (
            Path.home()
            if directory_text == "~"
            else Path(directory_text).expanduser()
        )
        if not directory.is_absolute():
            directory = self.base_dir / directory
        return directory, display_prefix, path_text[slash + 1 :]

    @staticmethod
    def _candidate(display_prefix: str, name: str, is_directory: bool) -> str:
        if display_prefix == "/":
            candidate = f"/{name}"
        else:
            candidate = f"{display_prefix}{name}"
        return f"{candidate}/" if is_directory else candidate

    @staticmethod
    def _quoted_candidate(candidate: str, quoted: bool, is_directory: bool) -> str:
        if quoted:
            closing_quote = "" if is_directory else '"'
            return f"{candidate}{closing_quote}"
        if not any(character.isspace() for character in candidate):
            return candidate
        closing_quote = "" if is_directory else '"'
        return f'"{candidate}{closing_quote}'


class ComposerCompleter(Completer):
    """Route slash commands and ``@`` paths to their focused completer."""

    def __init__(
        self, registry: SlashCommandRegistry, base_dir: str | Path | None = None
    ) -> None:
        self.slash = SlashCompleter(registry)
        self.path = PathCompleter(base_dir)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        before_cursor = document.text_before_cursor
        if before_cursor.startswith("/") and not any(
            character.isspace() for character in before_cursor
        ):
            yield from self.slash.get_completions(document, complete_event)
            return
        yield from self.path.get_completions(document, complete_event)


__all__ = ["ComposerCompleter", "PathCompleter", "SlashCompleter"]
