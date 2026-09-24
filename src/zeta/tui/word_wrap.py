"""Word-boundary wrapping for the prompt-toolkit composer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.containers import Window, WindowAlign
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import _CHAR_CACHE, Screen, WritePosition
from prompt_toolkit.mouse_events import MouseEvent


@dataclass(frozen=True)
class _WrapCell:
    source_char: str | None
    source_col: int | None
    display_cell: Any
    width: int
    is_escape: bool = False


@dataclass(frozen=True)
class _WrapPlan:
    rows: tuple[tuple[_WrapCell, ...], ...]
    overflow: bool = False
    row_start_columns: tuple[int, ...] = ()

    def row_for_source_col(self, source_col: int) -> int:
        for row_index, row in enumerate(self.rows):
            for cell in row:
                if cell.source_col == source_col:
                    return row_index
        return len(self.rows) - 1


def _prefix_width(
    get_line_prefix: Callable[[int, int], Any] | None,
    lineno: int,
    wrap_count: int,
) -> int:
    if get_line_prefix is None:
        return 0
    return fragment_list_width(to_formatted_text(get_line_prefix(lineno, wrap_count)))


def _build_wrap_cells(
    line: StyleAndTextTuples,
    cursor_col: int | None,
    is_input: bool,
) -> tuple[list[_WrapCell], _WrapCell | None]:
    has_cursor_cell = False
    document_line = line
    if is_input and line:
        style, text, *_ = line[-1]
        has_cursor_cell = (
            style == "" and text == " " and (len(line) > 1 or cursor_col in (None, 0))
        )
        if has_cursor_cell:
            document_line = line[:-1]

    cells: list[_WrapCell] = []
    source_col = 0
    for style, text, *_ in document_line:
        if "[ZeroWidthEscape]" in style:
            cells.append(
                _WrapCell(
                    source_char=None,
                    source_col=None,
                    display_cell=text,
                    width=0,
                    is_escape=True,
                )
            )
            continue
        for source_char in text:
            display_cell = _CHAR_CACHE[source_char, style]
            cells.append(
                _WrapCell(
                    source_char=source_char,
                    source_col=source_col,
                    display_cell=display_cell,
                    width=display_cell.width,
                )
            )
            source_col += 1

    cursor_cell = None
    if has_cursor_cell:
        display_cell = _CHAR_CACHE[" ", ""]
        cursor_cell = _WrapCell(
            source_char=" ",
            source_col=source_col,
            display_cell=display_cell,
            width=display_cell.width,
        )
    return cells, cursor_cell


def _build_wrap_plan(
    line: StyleAndTextTuples,
    lineno: int,
    width: int,
    get_line_prefix: Callable[[int, int], Any] | None,
    *,
    initial_x: int | None = None,
    cursor_col: int | None = None,
    is_input: bool = True,
    wrap_lines: bool = True,
    source_start: int = 0,
) -> _WrapPlan:
    def make_plan(rows: list[list[_WrapCell]], overflow: bool = False) -> _WrapPlan:
        row_tuples = tuple(tuple(row) for row in rows)
        row_start_columns = tuple(
            next(
                (cell.source_col for cell in row if cell.source_col is not None),
                source_start,
            )
            for row in row_tuples
        )
        return _WrapPlan(
            row_tuples,
            overflow=overflow,
            row_start_columns=row_start_columns,
        )

    cells, cursor_cell = _build_wrap_cells(line, cursor_col, is_input)
    cells = [
        cell
        for cell in cells
        if cell.source_col is None or cell.source_col >= source_start
    ]
    if cursor_cell is not None and cursor_cell.source_col < source_start:
        cursor_cell = None
    if width <= 0:
        return make_plan([cells], overflow=True)

    first_prefix_width = (
        _prefix_width(get_line_prefix, lineno, 0)
        if initial_x is None
        else initial_x
    )
    if first_prefix_width >= width:
        return make_plan([cells], overflow=True)

    rows: list[list[_WrapCell]] = []
    row_start = 0
    row_x = first_prefix_width
    last_break_end: int | None = None
    wrap_count = 0
    index = 0
    while index < len(cells):
        cell = cells[index]
        if wrap_lines and cell.width and row_x + cell.width > width:
            break_at = last_break_end if last_break_end is not None else index
            if break_at > row_start:
                rows.append(cells[row_start:break_at])
                row_start = break_at
                wrap_count += 1
                row_x = _prefix_width(get_line_prefix, lineno, wrap_count) + sum(
                    cell.width for cell in cells[break_at:index]
                )
                if row_x >= width:
                    return make_plan(rows, overflow=True)
                last_break_end = None
                continue

        if cell.source_char is not None and cell.source_char in " \t":
            last_break_end = index + 1
        row_x += cell.width
        index += 1

    rows.append(cells[row_start:])
    if not rows:
        rows.append([])

    if cursor_cell is not None:
        last_row = rows[-1]
        last_row_x = (
            first_prefix_width
            if len(rows) == 1
            else _prefix_width(get_line_prefix, lineno, len(rows) - 1)
        ) + sum(cell.width for cell in last_row)
        if wrap_lines and last_row_x + cursor_cell.width > width:
            rows.append([cursor_cell])
        else:
            last_row.append(cursor_cell)

    return make_plan(rows)


def _word_wrap_height(
    line: StyleAndTextTuples,
    lineno: int,
    width: int,
    get_line_prefix: Callable[[int, int], Any] | None,
    slice_stop: int | None = None,
    *,
    cursor_col: int | None = None,
) -> int:
    """Return visual rows from the shared word-wrap plan."""

    plan = _build_wrap_plan(
        line,
        lineno,
        width,
        get_line_prefix,
        cursor_col=cursor_col,
    )
    if plan.overflow:
        return 10**8
    if slice_stop is not None:
        return plan.row_for_source_col(slice_stop) + 1
    return len(plan.rows)


class WordWrapWindow(Window):
    """A prompt-toolkit window that wraps the composer at word boundaries."""

    MAX_COMPOSER_ROWS = 8

    def preferred_height(self, width: int, max_available_height: int) -> Dimension:
        """Report the word-wrapped composer height to the layout engine."""

        total_margin_width = self._get_total_margin_width()
        content_width = max(1, width - total_margin_width)
        content = self.content.create_content(content_width, height=1)
        height = 0
        for lineno in range(content.line_count):
            cursor_col = (
                content.cursor_position.x
                if content.cursor_position.y == lineno
                else None
            )
            height += _word_wrap_height(
                content.get_line(lineno),
                lineno,
                content_width,
                self.get_line_prefix,
                cursor_col=cursor_col,
            )
            if height >= self.MAX_COMPOSER_ROWS:
                break

        height = min(max(height, 1), self.MAX_COMPOSER_ROWS)
        return Dimension(min=1, preferred=height, max=height)

    def write_to_screen(
        self,
        screen: Screen,
        mouse_handlers: MouseHandlers,
        write_position: WritePosition,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        super().write_to_screen(
            screen,
            mouse_handlers,
            write_position,
            parent_style,
            erase_bg,
            z_index,
        )

        for (
            y,
            x_min,
            x_max,
            target_y,
            target_x,
            target_row,
            target_col,
        ) in self._soft_wrap_mouse_targets:
            if mouse_handlers.mouse_handlers[target_y].get(target_x) is None:
                continue

            def translate(
                mouse_event: MouseEvent,
                *,
                target_row: int = target_row,
                target_col: int = target_col,
            ) -> object:
                result = self.content.mouse_handler(
                    MouseEvent(
                        position=Point(x=target_col, y=target_row),
                        event_type=mouse_event.event_type,
                        button=mouse_event.button,
                        modifiers=mouse_event.modifiers,
                    )
                )
                if result == NotImplemented:
                    return self._mouse_handler(mouse_event)
                return result

            mouse_handlers.set_mouse_handler_for_range(
                x_min, x_max, y, y + 1, translate
            )

    def _copy_body(
        self,
        ui_content: UIContent,
        new_screen: Screen,
        write_position: WritePosition,
        move_x: int,
        width: int,
        vertical_scroll: int = 0,
        horizontal_scroll: int = 0,
        wrap_lines: bool = False,
        highlight_lines: bool = False,
        vertical_scroll_2: int = 0,
        always_hide_cursor: bool = False,
        has_focus: bool = False,
        align: WindowAlign = WindowAlign.LEFT,
        get_line_prefix: Callable[[int, int], Any] | None = None,
    ) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
        """Copy content while moving a whole word to the next visual row.

        This private-API override mirrors prompt-toolkit 3.0.53's
        ``Window._copy_body``. Revisit it if the pinned prompt-toolkit version
        changes.
        """

        xpos = write_position.xpos + move_x
        ypos = write_position.ypos
        line_count = ui_content.line_count
        new_buffer = new_screen.data_buffer
        empty_char = _CHAR_CACHE["", ""]
        visible_line_to_row_col: dict[int, tuple[int, int]] = {}
        rowcol_to_yx: dict[tuple[int, int], tuple[int, int]] = {}
        self._soft_wrap_mouse_targets: list[
            tuple[int, int, int, int, int, int, int]
        ] = []

        def copy_line(
            line: StyleAndTextTuples,
            lineno: int,
            x: int,
            y: int,
            is_input: bool = False,
        ) -> tuple[int, int]:
            if is_input:
                current_rowcol_to_yx = rowcol_to_yx
            else:
                current_rowcol_to_yx = {}

            if is_input and get_line_prefix:
                prompt = to_formatted_text(get_line_prefix(lineno, 0))
                x, y = copy_line(prompt, lineno, x, y, is_input=False)

            cursor_col = (
                ui_content.cursor_position.x
                if is_input
                and ui_content.cursor_position
                and ui_content.cursor_position.y == lineno
                else None
            )
            source_start = 0
            if horizontal_scroll and is_input:
                h_scroll = horizontal_scroll
                for cell in _build_wrap_cells(line, cursor_col, True)[0]:
                    if h_scroll <= 0:
                        break
                    h_scroll -= cell.width
                    if cell.source_col is not None:
                        source_start = cell.source_col + 1
                x -= h_scroll

            if align == WindowAlign.CENTER:
                line_width = fragment_list_width(line)
                if line_width < width:
                    x += (width - line_width) // 2
            elif align == WindowAlign.RIGHT:
                line_width = fragment_list_width(line)
                if line_width < width:
                    x += width - line_width

            first_row_x = x
            plan = _build_wrap_plan(
                line,
                lineno,
                width,
                get_line_prefix if is_input else None,
                initial_x=x,
                cursor_col=cursor_col,
                is_input=is_input,
                wrap_lines=wrap_lines,
                source_start=source_start,
            )

            x = first_row_x
            pending_tail: tuple[int, int, int, int] | None = None
            for row_index, row in enumerate(plan.rows):
                if row_index:
                    y += 1
                    x = 0
                    if is_input and get_line_prefix:
                        prompt = to_formatted_text(get_line_prefix(lineno, row_index))
                        x, y = copy_line(prompt, lineno, x, y, is_input=False)
                    if y >= write_position.height:
                        return x, y

                if row_index:
                    visible_line_to_row_col[y] = (
                        lineno,
                        plan.row_start_columns[row_index],
                    )

                new_buffer_row = new_buffer[y + ypos]
                for cell in row:
                    if cell.is_escape:
                        new_screen.zero_width_escapes[y + ypos][x + xpos] += (
                            cell.display_cell
                        )
                        continue

                    if x >= 0 and y >= 0 and x < width:
                        new_buffer_row[x + xpos] = cell.display_cell
                        if cell.width > 1:
                            for i in range(1, cell.width):
                                new_buffer_row[x + xpos + i] = empty_char
                        elif cell.width == 0:
                            for previous_width in [2, 1]:
                                if (
                                    x - previous_width >= 0
                                    and new_buffer_row[x + xpos - previous_width].width
                                    == previous_width
                                ):
                                    previous_char = new_buffer_row[
                                        x + xpos - previous_width
                                    ]
                                    new_buffer_row[x + xpos - previous_width] = (
                                        _CHAR_CACHE[
                                            previous_char.char + cell.display_cell.char,
                                            previous_char.style,
                                        ]
                                    )
                        if cell.source_col is not None:
                            current_rowcol_to_yx[lineno, cell.source_col] = (
                                y + ypos,
                                x + xpos,
                            )
                            if pending_tail is not None:
                                tail_y, tail_start, tail_end, target_col = pending_tail
                                self._soft_wrap_mouse_targets.append(
                                    (
                                        tail_y,
                                        tail_start,
                                        tail_end,
                                        y + ypos,
                                        x + xpos,
                                        lineno,
                                        target_col,
                                    )
                                )
                                pending_tail = None
                    x += cell.width

                if wrap_lines and row_index + 1 < len(plan.rows) and x < width:
                    pending_tail = (
                        y + ypos,
                        max(x, 0) + xpos,
                        width + xpos,
                        plan.row_start_columns[row_index + 1],
                    )

            if (
                is_input
                and cursor_col is not None
                and (lineno, cursor_col) not in current_rowcol_to_yx
            ):
                current_rowcol_to_yx[lineno, cursor_col] = (y + ypos, x + xpos)

            return x, y

        def copy() -> int:
            y = -vertical_scroll_2
            lineno = vertical_scroll
            while y < write_position.height and lineno < line_count:
                line = ui_content.get_line(lineno)
                visible_line_to_row_col[y] = (lineno, horizontal_scroll)
                x = 0
                x, y = copy_line(line, lineno, x, y, is_input=True)
                lineno += 1
                y += 1
            return y

        copy()

        def cursor_pos_to_screen_pos(row: int, col: int) -> Point:
            try:
                y, x = rowcol_to_yx[row, col]
            except KeyError:
                return Point(x=0, y=0)
            return Point(x=x, y=y)

        if ui_content.cursor_position:
            screen_cursor_position = cursor_pos_to_screen_pos(
                ui_content.cursor_position.y, ui_content.cursor_position.x
            )
            if has_focus:
                new_screen.set_cursor_position(self, screen_cursor_position)
                if always_hide_cursor:
                    new_screen.show_cursor = False
                else:
                    new_screen.show_cursor = ui_content.show_cursor
                self._highlight_digraph(new_screen)
            if highlight_lines:
                self._highlight_cursorlines(
                    new_screen,
                    screen_cursor_position,
                    xpos,
                    ypos,
                    width,
                    write_position.height,
                )

        if has_focus and ui_content.cursor_position:
            self._show_key_processor_key_buffer(new_screen)

        if ui_content.menu_position:
            new_screen.set_menu_position(
                self,
                cursor_pos_to_screen_pos(
                    ui_content.menu_position.y,
                    ui_content.menu_position.x,
                ),
            )

        new_screen.height = max(new_screen.height, ypos + write_position.height)
        return visible_line_to_row_col, rowcol_to_yx

    def _scroll_when_linewrapping(
        self, ui_content: UIContent, width: int, height: int
    ) -> None:
        # UIContent's built-in height calculation assumes character wrapping.
        # This temporary method keeps scrolling and render-info row counts in
        # sync with the word-boundary copy logic above.
        def get_height_for_line(
            lineno: int,
            line_width: int,
            prefix: Callable[[int, int], Any] | None,
            slice_stop: int | None = None,
        ) -> int:
            cursor_col = (
                ui_content.cursor_position.x
                if lineno == ui_content.cursor_position.y
                else None
            )
            return _word_wrap_height(
                ui_content.get_line(lineno),
                lineno,
                line_width,
                prefix,
                slice_stop,
                cursor_col=cursor_col,
            )

        ui_content.get_height_for_line = get_height_for_line
        super()._scroll_when_linewrapping(ui_content, width, height)
