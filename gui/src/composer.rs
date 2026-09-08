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
    marked_range: Option<Range<usize>>,
    layout: Option<TextLayout>,
    scroll: ScrollHandle,
    reveal_cursor: bool,
    pub enabled: bool,
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

        if event.modifiers.shift {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        } else {
            self.move_to(self.index_for_mouse_position(event.position), cx)
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
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

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
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

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        if self.content.is_empty() {
            return 0;
        }

        let Some(layout) = &self.layout else {
            return 0;
        };
        layout
            .index_for_position(position)
            .unwrap_or_else(|index| index)
            .min(self.content.len())
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
        self.content = "".into();
        self.selected_range = 0..0;
        self.selection_reversed = false;
        self.marked_range = None;
        self.layout = None;
        self.scroll.set_offset(point(px(0.), px(0.)));
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
        let start = layout.position_for_index(range.start)?;
        let end = layout.position_for_index(range.end)?;
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
        Some(self.offset_to_utf16(self.index_for_mouse_position(point)))
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
            marked_range: None,
            layout: None,
            scroll: ScrollHandle::new(),
            reveal_cursor: false,
            is_selecting: false,
            enabled: true,
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
            .and_then(|layout| layout.position_for_index(self.cursor_offset()))
        else {
            return;
        };
        let offset = self.index_for_mouse_position(
            position + point(px(0.), if down { px(33.) } else { px(-11.) }),
        );
        if select {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx);
        }
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
                    0x8a9287
                } else {
                    0xd8ddd5
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
            .key_context("Composer")
            .track_focus(&self.focus_handle)
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
            .border_color(rgb(0x343832))
            .text_color(rgb(0xd8ddd5))
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
                                let cursor = layout.position_for_index(input.cursor_offset());
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
                                            rgb(0xe1a84b),
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
                assert_eq!(view.index_for_mouse_position(position), suffix);
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
