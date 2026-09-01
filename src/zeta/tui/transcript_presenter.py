"""Transcript presentation for the full-screen terminal UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from ..types import StreamEvent, StreamEventType
from .layout import CONTENT_MARGIN
from .render import render_event
from .transcript import (
    ToolLifecycleKey,
    TranscriptWidget,
    _ToolUnit,
    _TranscriptUnit,
    _event_tool_lifecycle_key,
)


@dataclass(frozen=True)
class ToolEventPresentation:
    visible_output: bool = False
    stop_after_tool: bool = False


class TranscriptPresenter:
    """Own transcript output and tool-card presentation for the TUI."""

    def __init__(
        self,
        transcript: TranscriptWidget,
        console: Console,
        full_screen_active: Callable[[], bool],
        print_callback: Callable[[RenderableType | None], None],
    ) -> None:
        self.transcript = transcript
        self.console = console
        self._full_screen_active = full_screen_active
        self._print_callback = print_callback
        self._printed_units = False
        self._assistant_unit_open = False
        self._active_tool_calls: set[ToolLifecycleKey] = set()
        self._background_tool_ids: set[ToolLifecycleKey] = set()
        self._pending_tool_renders: list[RenderableType] = []
        self._tool_region_units: dict[ToolLifecycleKey, _ToolUnit] = {}
        self._tool_region: Live | None = None
        self._thinking_live: Live | None = None
        self._thinking_unit: _TranscriptUnit | None = None
        self._assistant_live: Live | None = None
        self._assistant_unit: _TranscriptUnit | None = None
        self._assistant_message_units: list[_TranscriptUnit] = []
        self._assistant_message_region: list[_TranscriptUnit] = []

    @property
    def tool_region(self) -> Live | None:
        return self._tool_region

    @property
    def has_active_agent(self) -> bool:
        return self.transcript.has_active_agent or any(
            unit.active_card for unit in self._tool_region_units.values()
        )

    def _append(self, renderable: RenderableType) -> None:
        self.transcript.append(renderable)

    def append_blank(self) -> None:
        self.transcript.append_blank()

    def print(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        self._print_callback(renderable)

    def print_unit(
        self, renderable: RenderableType | None, *, user: bool = False
    ) -> _TranscriptUnit | None:
        if renderable is None:
            return None
        if self._printed_units:
            if self._full_screen_active():
                self.append_blank()
            else:
                self.console.print()
        self.print(renderable)
        self._printed_units = True
        if self._full_screen_active() and self.transcript._units:
            unit = self.transcript._units[-1]
            if user:
                self.transcript.mark_user(unit)
            return unit
        return None

    def print_user(self, renderable: RenderableType | None) -> None:
        self.print_unit(renderable, user=True)

    def print_assistant(self, renderable: RenderableType | None) -> bool:
        if renderable is None:
            return False
        self.update_assistant(renderable)
        plain = getattr(renderable, "plain", None)
        return plain is None or bool(plain.strip())

    def update_assistant(self, rendered: RenderableType) -> None:
        """Replace the one mutable unit used by an in-flight assistant message."""

        if self._full_screen_active():
            if self._assistant_unit is None:
                unit_start = len(self.transcript._units)
                self._assistant_unit = self.print_unit(rendered)
                self._assistant_message_region.extend(self.transcript._units[unit_start:])
                if self._assistant_unit is not None:
                    self._assistant_message_units.append(self._assistant_unit)
            else:
                self._assistant_unit = self.transcript.replace(
                    self._assistant_unit, rendered
                )
        else:
            if self._assistant_live is None:
                self._assistant_live = Live(
                    Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN)),
                    console=self.console,
                    transient=True,
                    refresh_per_second=20,
                )
                self._assistant_live.start()
            else:
                self._assistant_live.update(
                    Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))
                )
        self._assistant_unit_open = True

    def finish_assistant(
        self,
        rendered: RenderableType,
        *,
        preserve_inline: bool = False,
    ) -> None:
        """Commit the completed assistant message into its existing unit."""

        if self._full_screen_active():
            if self._assistant_unit is None:
                self._assistant_unit = self.print_unit(rendered)
            else:
                self._assistant_unit = self.transcript.replace(
                    self._assistant_unit, rendered
                )
        else:
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            if preserve_inline:
                self.print_unit(rendered)
        self._assistant_unit = None
        self._assistant_unit_open = False

    def finish_assistant_message(self, rendered: RenderableType | None) -> None:
        """Render the complete assistant message in one transcript unit."""

        if self._full_screen_active():
            if rendered is not None:
                unit = next(
                    (
                        candidate
                        for candidate in self._assistant_message_units
                        if candidate in self.transcript._units
                    ),
                    None,
                )
                if unit is None:
                    unit = self.print_unit(rendered)
                else:
                    unit = self.transcript.replace(unit, rendered)
                for candidate in self._assistant_message_units:
                    if candidate is not unit:
                        self.transcript.remove(candidate)
            else:
                for candidate in self._assistant_message_region:
                    self.transcript.remove(candidate)
        else:
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            if rendered is not None:
                self.print_unit(rendered)
        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self._assistant_unit = None
        self._assistant_unit_open = False

    def reset_assistant_unit(self) -> None:
        self._assistant_unit_open = False
        self._assistant_unit = None

    def reset_assistant_message(self) -> None:
        """Forget the units owned by an incomplete assistant message."""

        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self.reset_assistant_unit()

    def start_thinking(self, rendered: Text) -> None:
        self.reset_assistant_unit()
        if self._full_screen_active():
            self._thinking_unit = self.print_unit(rendered)
        else:
            self._thinking_live = Live(
                Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN)),
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )
            self._thinking_live.start()

    def update_thinking(self, rendered: Text) -> None:
        if self._full_screen_active():
            if self._thinking_unit is None:
                self._thinking_unit = self.print_unit(rendered)
            else:
                self._thinking_unit = self.transcript.replace(
                    self._thinking_unit, rendered
                )
        elif self._thinking_live is not None:
            self._thinking_live.update(
                Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))
            )

    def finish_thinking(self, rendered: Text) -> None:
        if self._full_screen_active():
            if self._thinking_unit is None:
                self._thinking_unit = self.print_unit(rendered)
            else:
                self._thinking_unit = self.transcript.replace(
                    self._thinking_unit, rendered
                )
            self._thinking_unit = None
        elif self._thinking_live is not None:
            self._thinking_live.stop()
            self._thinking_live = None
            self.print_unit(rendered)
        self.reset_assistant_unit()

    def update_tool_region(self, event: StreamEvent) -> bool:
        rendered = render_event(event)
        if not isinstance(rendered, Text):
            return False
        if self._full_screen_active():
            lifecycle_key = _event_tool_lifecycle_key(event)
            if event.tool_call is not None:
                if lifecycle_key is not None:
                    self.transcript.update_tool(lifecycle_key, rendered, event)
            return bool(event.delta and event.delta.strip())
        call = event.tool_call
        lifecycle_key = _event_tool_lifecycle_key(event)
        if (
            call is not None
            and lifecycle_key is not None
            and lifecycle_key not in self._tool_region_units
        ):
            self._tool_region_units[lifecycle_key] = _ToolUnit(call, rendered, event)
        unit = (
            self._tool_region_units.get(lifecycle_key)
            if lifecycle_key is not None
            else None
        )
        if unit is None:
            return False
        if self._tool_region is None:
            self._tool_region = Live(
                self._tool_region_renderable(),
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )
        unit.update(rendered, event)
        self._tool_region.update(self._tool_region_renderable())
        return bool(event.delta and event.delta.strip())

    def _tool_region_renderable(self) -> Padding | Group:
        renderables = [unit.renderable for unit in self._tool_region_units.values()]
        content: RenderableType = Group(*renderables) if len(renderables) > 1 else renderables[0]
        return Padding(content, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))

    def refresh_active_agents(self) -> None:
        """Refresh elapsed time without adding child events to the parent store."""

        for unit in self.transcript._tools.values():
            unit.refresh()
        if self._tool_region is not None:
            for unit in self._tool_region_units.values():
                unit.refresh()
            self._tool_region.update(self._tool_region_renderable())

    def handle_tool_event(
        self,
        event: StreamEvent,
        *,
        aborted: bool,
    ) -> ToolEventPresentation | None:
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            self.reset_assistant_unit()
            lifecycle_key = _event_tool_lifecycle_key(event)
            if lifecycle_key is not None:
                self._active_tool_calls.add(lifecycle_key)
            rendered = render_event(event)
            if rendered is None:
                return ToolEventPresentation()
            if self._full_screen_active() and event.tool_call is not None:
                if self._printed_units:
                    self.append_blank()
                self.transcript.start_tool(
                    event.tool_call.id,
                    event.tool_call,
                    rendered,
                    event,
                )
                self._printed_units = True
            else:
                self.print_unit(rendered)
            if not self._full_screen_active() and event.tool_call is not None:
                if lifecycle_key is not None:
                    self._tool_region_units[lifecycle_key] = _ToolUnit(
                        event.tool_call, rendered, event
                    )
            return ToolEventPresentation(visible_output=True)
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            return ToolEventPresentation(
                visible_output=self.update_tool_region(event)
            )
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return None

        structured = (
            event.tool_result.structured_content
            if event.tool_result is not None
            else None
        )
        is_background_start = (
            structured is not None and structured.get("status") == "running"
        )
        if event.tool_call is not None:
            lifecycle_key = _event_tool_lifecycle_key(event)
            if lifecycle_key is not None:
                self._active_tool_calls.discard(lifecycle_key)
            if is_background_start:
                if lifecycle_key is not None:
                    self._background_tool_ids.add(lifecycle_key)
                if self._full_screen_active():
                    if lifecycle_key is not None:
                        self.transcript.mark_tool_background(lifecycle_key)
                path = structured.get("child_session_path") if structured else None
                if isinstance(path, str) and path:
                    if self._full_screen_active():
                        if lifecycle_key is not None:
                            self.transcript.set_tool_child_session_path(
                                lifecycle_key, path
                            )
                    else:
                        unit = self._tool_region_units.get(lifecycle_key)
                        if unit is not None:
                            unit.card.set_child_session_path(path)
                return ToolEventPresentation(visible_output=True)
        rendered = render_event(event)
        if not self._full_screen_active() and event.tool_call is not None:
            lifecycle_key = _event_tool_lifecycle_key(event)
            unit = self._tool_region_units.get(lifecycle_key)
            if unit is not None and rendered is not None:
                unit.finish(rendered, event)
                rendered = unit.renderable
        if rendered is not None:
            if self._full_screen_active() and event.tool_call is not None:
                lifecycle_key = _event_tool_lifecycle_key(event)
                if lifecycle_key is not None:
                    self.transcript.finish_tool(lifecycle_key, rendered, event)
            else:
                self._pending_tool_renders.append(rendered)
        if event.tool_call is not None:
            lifecycle_key = _event_tool_lifecycle_key(event)
            if lifecycle_key is not None:
                self._background_tool_ids.discard(lifecycle_key)
        if not self._active_tool_calls:
            self.commit_tool_region()
        return ToolEventPresentation(
            visible_output=rendered is not None,
            stop_after_tool=aborted,
        )

    def commit_tool_region(self) -> None:
        final_renders = self._pending_tool_renders
        self._pending_tool_renders = []
        if self._tool_region is not None:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
            if self._background_tool_ids:
                self._tool_region.update(self._tool_region_renderable())
            else:
                if final_renders:
                    self._tool_region.update(Group(*final_renders))
                self._tool_region.stop()
                self._tool_region = None
        else:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
        for rendered in final_renders:
            self.print(rendered)
        self.reset_assistant_unit()

    def discard_tool_region(self) -> None:
        self._pending_tool_renders.clear()
        if self._full_screen_active():
            self.transcript.discard_tools()
        if self._tool_region is not None:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
            if not self._background_tool_ids:
                self._tool_region.stop()
                self._tool_region = None
        else:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }

    def clear_active_tool_calls(self) -> None:
        self._active_tool_calls.clear()

    def clear(self) -> None:
        """Reset presentation state before rebuilding the transcript."""

        self._tool_region = None
        self._tool_region_units.clear()
        self._pending_tool_renders.clear()
        self._active_tool_calls.clear()
        self._thinking_live = None
        self._assistant_live = None
        self._thinking_unit = None
        self._assistant_unit = None
        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self._assistant_unit_open = False
        self._printed_units = False
        self.transcript.clear()
