//! Native text input using GPUI's UTF-16/IME contract and grapheme editing.
use gpui::{
    actions, canvas, div, fill, point, prelude::*, px, rgb, rgba, size, App, Bounds, ClipboardItem,
    Context, CursorStyle, ElementInputHandler, EntityInputHandler, EventEmitter, FocusHandle,
    Focusable, KeyBinding, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    Point, ScrollHandle, SharedString, StyledText, TextLayout, TextRun, UTF16Selection,
    UnderlineStyle, Window,
};
use std::ops::Range;
use unicode_segmentation::UnicodeSegmentation;

actions!(
    text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        ShowCharacterPalette,
        Paste,
        Cut,
        Copy,
        Up,
        Down,
        SelectUp,
        SelectDown,
        Newline,
        Submit,
    ]
);

pub struct Composer {
    focus_handle: FocusHandle,
    pub content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    // At a soft wrap, true places the caret at the start of the next row.
    cursor_downstream: bool,
    marked_range: Option<Range<usize>>,
    layout: Option<TextLayout>,
    scroll: ScrollHandle,
    reveal_cursor: bool,
    pub enabled: bool,
    pub images_enabled: bool,
    pub images: Vec<zeta_gui::session::ImageAttachment>,
    pub image_error: Option<String>,
    is_selecting: bool,
}

impl Composer {
    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx)
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            let prev = self.previous_boundary(self.cursor_offset());
            if self.cursor_offset() == prev {
                window.play_system_bell();
                return;
            }
            self.select_to(prev, cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            let next = self.next_boundary(self.cursor_offset());
            if self.cursor_offset() == next {
                window.play_system_bell();
                return;
            }
            self.select_to(next, cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle, cx);
        self.is_selecting = true;

        self.move_to_position(event.position, event.modifiers.shift, cx);
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.move_to_position(event.position, true, cx);
        }
    }

    fn show_character_palette(
        &mut self,
        _: &ShowCharacterPalette,
        window: &mut Window,
        _: &mut Context<Self>,
    ) {
        window.show_character_palette();
    }

    fn add_images(
        &mut self,
        images: Result<Vec<zeta_gui::session::ImageAttachment>, String>,
        cx: &mut Context<Self>,
    ) {
        if !self.enabled || !self.images_enabled {
            return;
        }
        match images {
            Ok(images)
                if self.images.len() + images.len() <= 4
                    && self
                        .images
                        .iter()
                        .chain(&images)
                        .map(|item| item.size)
                        .sum::<usize>()
                        <= zeta_gui::session::MAX_IMAGE_BYTES =>
            {
                self.images.extend(images);
                self.image_error = None;
            }
            Ok(_) => self.image_error = Some("attach at most 4 images, totaling 512 KiB".into()),
            Err(error) => self.image_error = Some(error),
        }
        cx.notify();
    }

    fn drop_images(&mut self, paths: &gpui::ExternalPaths, _: &mut Window, cx: &mut Context<Self>) {
        if self.enabled && self.images_enabled {
            if self.images.len() + paths.0.len() > 4 {
                self.image_error = Some("attach at most 4 images, totaling 512 KiB".into());
                cx.notify();
                return;
            }
            self.add_images(
                paths
                    .0
                    .iter()
                    .map(|path| zeta_gui::session::ImageAttachment::from_path(path))
                    .collect(),
                cx,
            );
        }
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else {
            return;
        };
        if self.enabled && self.images_enabled {
            for entry in item.entries() {
                match entry {
                    gpui::ClipboardEntry::Image(image) => {
                        self.add_images(
                            zeta_gui::session::ImageAttachment::from_bytes(
                                format!("pasted-image.{}", image.format.extension()),
                                &image.bytes,
                            )
                            .map(|image| vec![image]),
                            cx,
                        );
                        return;
                    }
                    gpui::ClipboardEntry::ExternalPaths(paths) => {
                        self.drop_images(paths, window, cx);
                        return;
                    }
                    _ => {}
                }
            }
        }
        if let Some(text) = item.text() {
            self.replace_text_in_range(None, &text, window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }
    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx)
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selection_reversed = false;
        self.selected_range = offset..offset;
        self.cursor_downstream = true;
        self.reveal_cursor = true;
        cx.notify()
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn caret_for_mouse_position(&self, position: Point<Pixels>) -> (usize, bool) {
        if self.content.is_empty() {
            return (0, true);
        }

        let Some(layout) = &self.layout else {
            return (0, true);
        };
        let (index, downstream) = closest_caret(layout, position);
        (index.min(self.content.len()), downstream)
    }

    fn move_to_position(&mut self, position: Point<Pixels>, select: bool, cx: &mut Context<Self>) {
        let (offset, downstream) = self.caret_for_mouse_position(position);
        if select {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx);
        }
        self.cursor_downstream = downstream;
    }

    fn position_for_index(&self, layout: &TextLayout, index: usize) -> Option<Point<Pixels>> {
        let position = layout.position_for_index(index)?;
        if index != self.cursor_offset() || self.cursor_downstream {
            let mut start = 0;
            for line in layout.line_layouts() {
                for boundary in line.wrap_boundaries() {
                    let glyph = &line.runs()[boundary.run_ix].glyphs[boundary.glyph_ix];
                    if start + glyph.index == index {
                        return Some(point(
                            layout.bounds().left(),
                            position.y + layout.line_height(),
                        ));
                    }
                }
                start += line.len() + 1;
            }
        }
        Some(position)
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset
        } else {
            self.selected_range.end = offset
        };
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        self.cursor_downstream = true;
        self.reveal_cursor = true;
        cx.notify()
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        utf16_offset(&self.content, offset)
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;

        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }

        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range_utf16: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range_utf16.start)..self.offset_from_utf16(range_utf16.end)
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .rev()
            .find_map(|(idx, _)| (idx < offset).then_some(idx))
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .find_map(|(idx, _)| (idx > offset).then_some(idx))
            .unwrap_or(self.content.len())
    }

    pub fn reset(&mut self) {
        self.images.clear();
        self.image_error = None;
        self.content = "".into();
        self.selected_range = 0..0;
        self.selection_reversed = false;
        self.marked_range = None;
        self.layout = None;
        self.scroll.set_offset(point(px(0.), px(0.)));
        self.cursor_downstream = true;
        self.reveal_cursor = true;
        self.is_selecting = false;
    }
}

impl EntityInputHandler for Composer {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        if !self.enabled {
            return None;
        }
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.enabled {
            return;
        }
        let range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        self.content =
            (self.content[0..range.start].to_owned() + new_text + &self.content[range.end..])
                .into();
        self.selected_range = range.start + new_text.len()..range.start + new_text.len();
        self.marked_range.take();
        self.selection_reversed = false;
        self.cursor_downstream = true;
        self.reveal_cursor = true;
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.enabled {
            return;
        }
        let range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        self.content =
            (self.content[0..range.start].to_owned() + new_text + &self.content[range.end..])
                .into();
        if !new_text.is_empty() {
            self.marked_range = Some(range.start..range.start + new_text.len());
        } else {
            self.marked_range = None;
        }
        self.selected_range = new_selected_range_utf16
            .as_ref()
            .map(|selection| {
                let start = utf16_offset(new_text, selection.start);
                let end = utf16_offset(new_text, selection.end);
                range.start + start..range.start + end
            })
            .unwrap_or_else(|| range.start + new_text.len()..range.start + new_text.len());

        self.selection_reversed = false;
        self.cursor_downstream = true;
        self.reveal_cursor = true;
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let range = self.range_from_utf16(&range_utf16);
        let layout = self.layout.as_ref()?;
        let start = self.position_for_index(layout, range.start)?;
        let end = self.position_for_index(layout, range.end)?;
        // Native input methods anchor to the first visual row of a selection.
        let right = if end.y == start.y {
            end.x
        } else {
            bounds.right()
        };
        Some(Bounds::from_corners(start, point(right, start.y + px(22.))))
    }

    fn character_index_for_point(
        &mut self,
        point: gpui::Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        self.layout.as_ref()?;
        Some(self.offset_to_utf16(self.caret_for_mouse_position(point).0))
    }
}

impl EventEmitter<Submit> for Composer {}

impl Composer {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: "".into(),
            placeholder: "write a message...".into(),
            selected_range: 0..0,
            selection_reversed: false,
            cursor_downstream: true,
            marked_range: None,
            layout: None,
            scroll: ScrollHandle::new(),
            reveal_cursor: false,
            is_selecting: false,
            enabled: true,
            images_enabled: false,
            images: Vec::new(),
            image_error: None,
        }
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        if self.enabled && self.marked_range.is_none() {
            cx.emit(Submit);
        }
    }

    fn newline(&mut self, _: &Newline, window: &mut Window, cx: &mut Context<Self>) {
        self.replace_text_in_range(None, "\n", window, cx);
    }

    fn vertical(&mut self, down: bool, select: bool, cx: &mut Context<Self>) {
        let Some(position) = self
            .layout
            .as_ref()
            .and_then(|layout| self.position_for_index(layout, self.cursor_offset()))
        else {
            return;
        };
        self.move_to_position(
            position + point(px(0.), if down { px(33.) } else { px(-11.) }),
            select,
            cx,
        );
    }

    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(false, false, cx);
    }
    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(true, false, cx);
    }
    fn select_up(&mut self, _: &SelectUp, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(false, true, cx);
    }
    fn select_down(&mut self, _: &SelectDown, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(true, true, cx);
    }
}

// TextLayout exposes only containing-glyph hit testing. Use its shaped lines for
// nearest insertion boundaries, retaining which side of a soft wrap was hit.
fn closest_caret(layout: &TextLayout, position: Point<Pixels>) -> (usize, bool) {
    let height = layout.line_height();
    let mut y = (position.y - layout.bounds().top())
        .max(px(0.))
        .min(layout.bounds().size.height - height / 2.);
    let mut start = 0;
    for line in layout.line_layouts() {
        let line_height = height * (line.wrap_boundaries().len() + 1) as f32;
        if y < line_height {
            let row = (y / height) as usize;
            let mut index = line
                .closest_index_for_position(point(position.x - layout.bounds().left(), y), height)
                .unwrap_or_else(|index| index);
            // GPUI falls through to len after the final glyph's start. Include
            // its midpoint locally, using unwrapped coordinates for the last row.
            if index == line.len() {
                if let Some(last) = line.runs().iter().rev().find_map(|run| run.glyphs.last()) {
                    let row_start_x = if row > 0 {
                        let boundary = line.wrap_boundaries()[row - 1];
                        line.runs()[boundary.run_ix].glyphs[boundary.glyph_ix]
                            .position
                            .x
                    } else {
                        px(0.)
                    };
                    let x = position.x - layout.bounds().left() + row_start_x;
                    if x >= last.position.x
                        && x <= (last.position.x + line.unwrapped_layout.width) / 2.
                    {
                        index = last.index;
                    }
                }
            }
            let downstream = row > 0 && {
                let boundary = line.wrap_boundaries()[row - 1];
                index == line.runs()[boundary.run_ix].glyphs[boundary.glyph_ix].index
            };
            return (start + index, downstream);
        }
        y -= line_height;
        start += line.len() + 1;
    }
    (layout.len(), false)
}

fn utf16_offset(text: &str, offset: usize) -> usize {
    let mut count = 0;
    text.char_indices()
        .find_map(|(index, ch)| {
            if count >= offset {
                Some(index)
            } else {
                count += ch.len_utf16();
                None
            }
        })
        .unwrap_or(text.len())
}

impl Focusable for Composer {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, Some("Composer")),
        KeyBinding::new("delete", Delete, Some("Composer")),
        KeyBinding::new("left", Left, Some("Composer")),
        KeyBinding::new("right", Right, Some("Composer")),
        KeyBinding::new("shift-left", SelectLeft, Some("Composer")),
        KeyBinding::new("shift-right", SelectRight, Some("Composer")),
        KeyBinding::new("up", Up, Some("Composer")),
        KeyBinding::new("down", Down, Some("Composer")),
        KeyBinding::new("shift-up", SelectUp, Some("Composer")),
        KeyBinding::new("shift-down", SelectDown, Some("Composer")),
        KeyBinding::new("cmd-a", SelectAll, Some("Composer")),
        KeyBinding::new("cmd-v", Paste, Some("Composer")),
        KeyBinding::new("cmd-c", Copy, Some("Composer")),
        KeyBinding::new("cmd-x", Cut, Some("Composer")),
        KeyBinding::new("home", Home, Some("Composer")),
        KeyBinding::new("end", End, Some("Composer")),
        KeyBinding::new("ctrl-cmd-space", ShowCharacterPalette, Some("Composer")),
        KeyBinding::new("enter", Submit, Some("Composer")),
        KeyBinding::new("shift-enter", Newline, Some("Composer")),
    ]);
}

impl Render for Composer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = crate::appearance(window).palette();
        let input = cx.entity();
        let display = if self.content.is_empty() {
            self.placeholder.clone()
        } else {
            self.content.clone()
        };
        let style = window.text_style();
        let mut boundaries = vec![0, display.len()];
        if !self.content.is_empty() {
            boundaries.extend([self.selected_range.start, self.selected_range.end]);
            if let Some(marked) = &self.marked_range {
                boundaries.extend([marked.start, marked.end]);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        let runs = boundaries
            .windows(2)
            .map(|range| TextRun {
                len: range[1] - range[0],
                font: style.font(),
                color: rgb(if self.content.is_empty() {
                    p.muted
                } else {
                    p.text
                })
                .into(),
                background_color: self
                    .selected_range
                    .contains(&range[0])
                    .then(|| rgba(0x6e8eaa60).into()),
                underline: self
                    .marked_range
                    .as_ref()
                    .is_some_and(|marked| marked.contains(&range[0]))
                    .then_some(UnderlineStyle {
                        color: Some(style.color),
                        thickness: px(1.),
                        wavy: false,
                    }),
                strikethrough: None,
            })
            .collect();
        let text = StyledText::new(display).with_runs(runs);
        let layout = text.layout().clone();
        div()
            .id("composer-input")
            .on_drop(cx.listener(Self::drop_images))
            .children(self.images.iter().enumerate().map(|(index, image)| {
                div()
                    .id(format!("draft-image-{index}"))
                    .min_h(px(40.))
                    .px_2()
                    .py_2()
                    .bg(rgb(p.code_chip))
                    .child(format!("{} · {} bytes · remove", image.name, image.size))
                    .cursor_pointer()
                    .on_click(cx.listener(move |view, _, _, cx| {
                        if view.enabled {
                            view.images.remove(index);
                            cx.notify();
                        }
                    }))
            }))
            .when_some(self.image_error.clone(), |view, error| {
                view.child(div().text_color(rgb(p.error)).child(error))
            })
            .key_context("Composer")
            .track_focus(&self.focus_handle)
            .tab_index(0)
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::up))
            .on_action(cx.listener(Self::down))
            .on_action(cx.listener(Self::select_up))
            .on_action(cx.listener(Self::select_down))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::show_character_palette))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::newline))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .w_full()
            .min_h(px(72.))
            .max_h(px(220.))
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .line_height(px(22.))
            .p_3()
            .border_1()
            .border_color(rgb(p.border))
            .text_color(rgb(p.text))
            .child(
                div().relative().w_full().child(text).child(
                    canvas(
                        move |_, _, _| (),
                        move |bounds, (), window, cx| {
                            input.update(cx, |input, cx| {
                                window.handle_input(
                                    &input.focus_handle,
                                    ElementInputHandler::new(bounds, cx.entity()),
                                    cx,
                                );
                                let cursor =
                                    input.position_for_index(&layout, input.cursor_offset());
                                if let Some(cursor) = cursor {
                                    if input.reveal_cursor {
                                        input.reveal_cursor = false;
                                        let viewport = input.scroll.bounds();
                                        let adjustment = if cursor.y < viewport.top() + px(13.) {
                                            viewport.top() + px(13.) - cursor.y
                                        } else if cursor.y + px(22.) > viewport.bottom() - px(13.) {
                                            viewport.bottom() - px(13.) - cursor.y - px(22.)
                                        } else {
                                            px(0.)
                                        };
                                        if adjustment != px(0.) {
                                            let offset = input.scroll.offset();
                                            input
                                                .scroll
                                                .set_offset(point(offset.x, offset.y + adjustment));
                                            cx.notify();
                                        }
                                    }
                                    if input.enabled
                                        && input.focus_handle.is_focused(window)
                                        && input.selected_range.is_empty()
                                    {
                                        window.paint_quad(fill(
                                            Bounds::new(cursor, size(px(1.), px(22.))),
                                            rgb(p.accent),
                                        ));
                                    }
                                }
                                input.layout = Some(layout);
                            });
                        },
                    )
                    .absolute()
                    .size_full()
                    .top_0()
                    .left_0(),
                ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_caret(
        view: &mut Composer,
        layout: &TextLayout,
        index: usize,
        row: usize,
        x: Pixels,
        window: &mut Window,
        cx: &mut Context<Composer>,
    ) {
        assert_eq!(view.cursor_offset(), index);
        let expected = layout.bounds().origin + point(x, layout.line_height() * row as f32);
        assert_eq!(view.position_for_index(layout, index), Some(expected));
        let range = view.range_to_utf16(&(index..index));
        assert_eq!(
            view.bounds_for_range(range, layout.bounds(), window, cx)
                .unwrap()
                .origin,
            expected
        );
    }

    #[gpui::test]
    fn clicks_use_nearest_boundaries_and_preserve_wrap_row(cx: &mut gpui::TestAppContext) {
        let window = cx.open_window(size(px(240.), px(400.)), |_, cx| Composer::new(cx));
        window
            .update(cx, |view, window, cx| {
                view.replace_text_in_range(None, &"abcdefghij".repeat(6), window, cx);
            })
            .unwrap();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
        window
            .update(cx, |view, window, cx| {
                let layout = view.layout.as_ref().unwrap().clone();
                let wrapped = layout.wrapped_text();
                let rows: Vec<_> = wrapped.split('\n').collect();
                assert!(rows.len() >= 3);
                let wrap = rows[0].len();
                let second_wrap = wrap + rows[1].len();
                let line = layout.line_layouts().remove(0);
                let glyph_x = |index| line.unwrapped_layout.x_for_index(index);
                let origin = layout.bounds().origin;
                let height = layout.line_height();
                // Both halves of a glyph, including the first glyph after wrapping.
                for (row, start) in [(0, 0), (1, wrap)] {
                    let width = glyph_x(start + 1) - glyph_x(start);
                    for (fraction, index, x) in [(0.25, start, px(0.)), (0.75, start + 1, width)] {
                        let position =
                            origin + point(width * fraction, height * (row as f32 + 0.5));
                        assert_eq!(view.caret_for_mouse_position(position).0, index);
                        assert_eq!(
                            view.character_index_for_point(position, window, cx),
                            Some(index)
                        );
                        view.on_mouse_down(
                            &MouseDownEvent {
                                position,
                                button: MouseButton::Left,
                                ..Default::default()
                            },
                            window,
                            cx,
                        );
                        assert_caret(view, &layout, index, row, x, window, cx);
                    }
                }
                // The same byte index has two valid visual positions.
                let end_x = glyph_x(wrap);
                let last_width = end_x - glyph_x(wrap - 1);
                view.move_to_position(
                    origin + point(end_x - last_width * 0.25, height / 2.),
                    false,
                    cx,
                );
                assert_caret(view, &layout, wrap, 0, end_x, window, cx);
                view.move_to_position(origin + point(px(0.), height * 1.5), false, cx);
                assert_caret(view, &layout, wrap, 1, px(0.), window, cx);
                view.vertical(true, false, cx);
                assert_caret(view, &layout, second_wrap, 2, px(0.), window, cx);
                view.vertical(false, true, cx);
                assert_caret(view, &layout, wrap, 1, px(0.), window, cx);
                assert!(view.selection_reversed);
                view.vertical(false, false, cx);
                assert_caret(view, &layout, 0, 0, px(0.), window, cx);
                // A drag and a shift-click retain the target row, too.
                view.on_mouse_move(
                    &MouseMoveEvent {
                        position: origin + point(px(0.), height * 1.5),
                        ..Default::default()
                    },
                    window,
                    cx,
                );
                assert_caret(view, &layout, wrap, 1, px(0.), window, cx);
                view.on_mouse_down(
                    &MouseDownEvent {
                        position: origin + point(px(0.), height * 2.5),
                        button: MouseButton::Left,
                        modifiers: gpui::Modifiers {
                            shift: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    window,
                    cx,
                );
                assert_caret(view, &layout, second_wrap, 2, px(0.), window, cx);
            })
            .unwrap();
    }

    #[gpui::test]
    fn final_glyph_clicks_use_nearest_boundary(cx: &mut gpui::TestAppContext) {
        let window = cx.open_window(size(px(240.), px(400.)), |_, cx| Composer::new(cx));
        for text in ["abcd".to_owned(), "abcdefghij".repeat(6)] {
            window
                .update(cx, |view, window, cx| {
                    view.reset();
                    view.replace_text_in_range(None, &text, window, cx);
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
            window
                .update(cx, |view, window, cx| {
                    let layout = view.layout.as_ref().unwrap().clone();
                    let wrapped = layout.wrapped_text();
                    let rows: Vec<_> = wrapped.split('\n').collect();
                    let line = layout.line_layouts().remove(0);
                    let mut start = 0;
                    for (row, text) in rows.iter().enumerate() {
                        assert!(text.len() > 1);
                        let end = start + text.len();
                        let row_x = line.unwrapped_layout.x_for_index(start);
                        let left = line.unwrapped_layout.x_for_index(end - 1) - row_x;
                        let right = line.unwrapped_layout.x_for_index(end) - row_x;
                        for (fraction, index, x) in [(0.25, end - 1, left), (0.75, end, right)] {
                            let position = layout.bounds().origin
                                + point(
                                    left + (right - left) * fraction,
                                    layout.line_height() * (row as f32 + 0.5),
                                );
                            view.on_mouse_down(
                                &MouseDownEvent {
                                    position,
                                    button: MouseButton::Left,
                                    ..Default::default()
                                },
                                window,
                                cx,
                            );
                            assert_caret(view, &layout, index, row, x, window, cx);
                        }
                        start = end;
                    }
                })
                .unwrap();
        }
    }

    #[gpui::test]
    fn vertical_movement_uses_final_glyph_midpoint(cx: &mut gpui::TestAppContext) {
        let text = "aa\na\u{10400}\naa";
        let window = cx.open_window(size(px(240.), px(400.)), |_, cx| Composer::new(cx));
        window
            .update(cx, |view, window, cx| {
                view.replace_text_in_range(None, text, window, cx);
            })
            .unwrap();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
        window
            .update(cx, |view, window, cx| {
                let layout = view.layout.as_ref().unwrap().clone();
                let lines = layout.line_layouts();
                let left = lines[1].unwrapped_layout.x_for_index(1);
                let right = lines[1].unwrapped_layout.width;
                for (source, row, down) in [(2, 0, true), (text.len(), 2, false)] {
                    // GPUI's test font gives supplementary letters double width.
                    // The source caret lands at the final glyph's midpoint.
                    let x = lines[row].unwrapped_layout.width;
                    assert_eq!(x, (left + right) / 2.);
                    view.move_to(source, cx);
                    assert_caret(view, &layout, source, row, x, window, cx);
                    view.vertical(down, false, cx);
                    assert_caret(view, &layout, 4, 1, left, window, cx);
                }
            })
            .unwrap();
    }

    #[gpui::test]
    fn empty_rows_and_trailing_newlines_keep_exact_caret_rows(cx: &mut gpui::TestAppContext) {
        let window = cx.open_window(size(px(240.), px(400.)), |_, cx| Composer::new(cx));
        for text in ["", "\n", "a\n\nb\n\n"] {
            window
                .update(cx, |view, window, cx| {
                    view.reset();
                    view.replace_text_in_range(None, text, window, cx);
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
            window
                .update(cx, |view, window, cx| {
                    let layout = view.layout.as_ref().unwrap().clone();
                    let mut start = 0;
                    let mut starts = Vec::new();
                    for (row, line) in text.split('\n').enumerate() {
                        starts.push(start);
                        let position = layout.bounds().origin
                            + point(px(0.), layout.line_height() * (row as f32 + 0.5));
                        view.move_to_position(position, false, cx);
                        assert_caret(view, &layout, start, row, px(0.), window, cx);
                        assert_eq!(
                            view.character_index_for_point(position, window, cx),
                            Some(start)
                        );
                        start += line.len() + 1;
                    }
                    for row in (0..starts.len().saturating_sub(1)).rev() {
                        view.vertical(false, false, cx);
                        assert_caret(view, &layout, starts[row], row, px(0.), window, cx);
                    }
                    for (row, index) in starts.iter().enumerate().skip(1) {
                        view.vertical(true, false, cx);
                        assert_caret(view, &layout, *index, row, px(0.), window, cx);
                    }
                })
                .unwrap();
        }
    }

    #[gpui::test]
    fn wrapped_prompt_supports_caret_selection_and_scrolling(cx: &mut gpui::TestAppContext) {
        cx.update(bind_keys);
        let window = cx.open_window(size(px(240.), px(400.)), |_, cx| Composer::new(cx));
        window
            .update(cx, |view, window, cx| window.focus(&view.focus_handle, cx))
            .unwrap();
        let draw = |cx: &mut gpui::TestAppContext| {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        };
        let prompt = "abcdefghij".repeat(60);
        cx.simulate_input(window.into(), &prompt);
        draw(cx);
        draw(cx); // Apply the caret's scroll request.
        window
            .update(cx, |view, window, cx| {
                let layout = view.layout.as_ref().unwrap().clone();
                assert!(layout.wrapped_text().lines().count() > 10);
                assert!(view.scroll.max_offset().y > px(0.));
                let caret = layout.position_for_index(prompt.len()).unwrap();
                assert!(caret.x <= view.scroll.bounds().right());
                assert!(caret.y >= view.scroll.bounds().top());
                assert!(caret.y + px(22.) <= view.scroll.bounds().bottom());
                let suffix = prompt.len() - 3;
                let position = layout.position_for_index(suffix).unwrap() + point(px(1.), px(11.));
                assert_eq!(view.caret_for_mouse_position(position).0, suffix);
                view.on_mouse_down(
                    &MouseDownEvent {
                        position,
                        button: MouseButton::Left,
                        ..Default::default()
                    },
                    window,
                    cx,
                );
                assert_eq!(view.cursor_offset(), suffix);
                view.select_to(prompt.len(), cx);
                view.copy(&Copy, window, cx);
                assert_eq!(
                    cx.read_from_clipboard().unwrap().text().unwrap(),
                    &prompt[suffix..]
                );
                let range = view.range_to_utf16(&(suffix..suffix));
                let ime_bounds = view
                    .bounds_for_range(range, layout.bounds(), window, cx)
                    .unwrap();
                assert_eq!(
                    ime_bounds.origin,
                    layout.position_for_index(suffix).unwrap()
                );
                view.move_to(suffix, cx);
                view.vertical(false, true, cx);
                assert!(view.cursor_offset() < suffix);
                assert!(view.selection_reversed);
                view.home(&Home, window, cx);
            })
            .unwrap();
        draw(cx);
        draw(cx);
        window
            .update(cx, |view, _, _| {
                assert_eq!(view.scroll.offset().y, px(0.));
                assert_eq!(view.cursor_offset(), 0);
            })
            .unwrap();
    }

    #[gpui::test]
    fn native_input_edits_multiline_unicode_and_pastes(cx: &mut gpui::TestAppContext) {
        cx.update(bind_keys);
        let window = cx.add_window(|_, cx| Composer::new(cx));
        let view = window.root(cx).unwrap();
        window
            .update(cx, |view, window, cx| window.focus(&view.focus_handle, cx))
            .unwrap();
        cx.simulate_input(window.into(), "Hello! Καλημέρα");
        cx.simulate_keystrokes(window.into(), "home right shift-right");
        cx.simulate_input(window.into(), "E");
        cx.simulate_keystrokes(window.into(), "end shift-enter");
        cx.update(|cx| cx.write_to_clipboard(ClipboardItem::new_string("second\nthird".into())));
        cx.simulate_keystrokes(window.into(), "cmd-v up home");
        view.read_with(cx, |view, _| {
            assert_eq!(view.content.as_ref(), "HEllo! Καλημέρα\nsecond\nthird");
            assert_eq!(view.selected_range, 0..0);
        });
        cx.simulate_keystrokes(window.into(), "cmd-a backspace");
        view.read_with(cx, |view, _| assert!(view.content.is_empty()));
    }

    #[gpui::test]
    fn ime_replacement_uses_utf16_relative_to_marked_text(cx: &mut gpui::TestAppContext) {
        let window = cx.add_window(|_, cx| Composer::new(cx));
        window
            .update(cx, |view, window, cx| {
                view.replace_text_in_range(None, "ab𝄞", window, cx);
                view.replace_and_mark_text_in_range(None, "日本", Some(1..2), window, cx);
                assert_eq!(
                    view.selected_text_range(false, window, cx).unwrap().range,
                    5..6
                );
                assert_eq!(view.marked_text_range(window, cx), Some(4..6));
                view.replace_text_in_range(None, "日本語", window, cx);
                assert_eq!(view.content.as_ref(), "ab𝄞日本語");
                assert!(view.marked_range.is_none());
                view.replace_text_in_range(Some(2..4), "!", window, cx);
                assert_eq!(view.content.as_ref(), "ab!日本語");
                view.move_to(2, cx);
                view.replace_text_in_range(None, "e\u{301}", window, cx);
                view.backspace(&Backspace, window, cx);
                assert_eq!(view.content.as_ref(), "ab!日本語");
            })
            .unwrap();
    }
}

#[cfg(test)]
mod attachment_tests {
    use super::*;

    #[gpui::test]
    fn pasted_image_names_follow_clipboard_format(cx: &mut gpui::TestAppContext) {
        let window = cx.add_window(|_, cx| Composer::new(cx));
        for (format, bytes, name) in [
            (
                gpui::ImageFormat::Png,
                b"\x89PNG\r\n\x1a\n".as_slice(),
                "pasted-image.png",
            ),
            (
                gpui::ImageFormat::Jpeg,
                b"\xff\xd8\xff".as_slice(),
                "pasted-image.jpg",
            ),
            (
                gpui::ImageFormat::Gif,
                b"GIF89a".as_slice(),
                "pasted-image.gif",
            ),
            (
                gpui::ImageFormat::Webp,
                b"RIFF\x04\x00\x00\x00WEBP".as_slice(),
                "pasted-image.webp",
            ),
        ] {
            window
                .update(cx, |composer, window, cx| {
                    composer.images_enabled = true;
                    composer.images.clear();
                    cx.write_to_clipboard(ClipboardItem::new_image(&gpui::Image::from_bytes(
                        format,
                        bytes.to_vec(),
                    )));
                    composer.paste(&Paste, window, cx);
                    assert_eq!(composer.images[0].name, name);
                })
                .unwrap();
        }
    }

    #[gpui::test]
    fn failed_drop_preserves_the_previous_batch(cx: &mut gpui::TestAppContext) {
        let directory = std::env::temp_dir().join(format!("zeta-95-batch-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let good = directory.join("good.png");
        let bad = directory.join("bad.png");
        std::fs::write(&good, b"\x89PNG\r\n\x1a\n").unwrap();
        let paths = gpui::ExternalPaths([good.clone(), bad.clone()].into_iter().collect());
        let window = cx.add_window(|_, cx| Composer::new(cx));
        window
            .update(cx, |composer, window, cx| {
                composer.images_enabled = true;
                let previous = zeta_gui::session::ImageAttachment::from_path(&good).unwrap();
                composer.images.push(previous.clone());
                // A missing second file, then invalid bytes, must preserve the first batch.
                for contents in [None, Some(b"invalid".as_slice())] {
                    if let Some(contents) = contents {
                        std::fs::write(&bad, contents).unwrap();
                    }
                    composer.drop_images(&paths, window, cx);
                    assert_eq!(composer.images, vec![previous.clone()]);
                    assert!(composer.image_error.is_some());
                }
                std::fs::write(&bad, b"\x89PNG\r\n\x1a\n").unwrap();
                composer.drop_images(&paths, window, cx);
                assert_eq!(composer.images.len(), 3);
                assert!(composer.image_error.is_none());
                composer.drop_images(&paths, window, cx);
                assert_eq!(composer.images.len(), 3);
                assert!(composer.image_error.is_some());
            })
            .unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[gpui::test]
    fn file_drop_obeys_size_limits_and_protocol_gate(cx: &mut gpui::TestAppContext) {
        let path = std::env::temp_dir().join(format!("zeta-95-drop-{}.png", std::process::id()));
        std::fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();
        let paths = gpui::ExternalPaths([path.clone()].into_iter().collect());
        let composer = cx.new(Composer::new);
        let window = cx.add_window(|_, _| gpui::Empty);
        window
            .update(cx, |_, window, cx| {
                composer.update(cx, |composer, cx| {
                    composer.drop_images(&paths, window, cx);
                    assert!(composer.images.is_empty(), "old servers hide file input");
                    composer.images_enabled = true;
                    composer.drop_images(&paths, window, cx);
                    assert_eq!(composer.images.len(), 1);
                    assert_eq!(
                        composer.images[0].name,
                        path.file_name().unwrap().to_string_lossy()
                    );
                    std::fs::write(&path, vec![0; zeta_gui::session::MAX_IMAGE_BYTES + 1]).unwrap();
                    composer.drop_images(&paths, window, cx);
                    assert_eq!(composer.images.len(), 1);
                    assert!(composer.image_error.as_ref().unwrap().contains("512 KiB"));
                });
            })
            .unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
