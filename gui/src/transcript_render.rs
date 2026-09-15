//! Single home for every transcript-row renderer.
//!
//! The AST-based `renderer_literal_fence` guard test parses THIS whole
//! source (not a fixed function-name list) and rejects every string,
//! byte-string, or C-string literal in expression position — no ambient
//! method-name allowance. The only literals that pass ride an allowlisted
//! macro payload (`unreachable!` and pattern-only `matches!` — the only
//! two the module actually uses); every other macro (`panic!`, `todo!`,
//! `unimplemented!`, `assert*!`, `debug_assert*!`, `stringify!`, `concat!`,
//! `write!`, unknown imports) is rejected outright. `format!` is scanned
//! at ambient depth, so any literal fragment in its payload trips too.
//! Every user-visible string a row paints has to come from the typed
//! `row_text::RowText` / `LoginRowText` model — that is what makes a new
//! stray literal impossible to add to any renderer WITHOUT failing a test,
//! and it fixes the r1 review's "new helper fn / render_login_row bypass"
//! by scanning the whole module rather than a fixed six-fn allowlist.
//!
//! Widget-ID composition (`sel::tool_verb(index)`, `Button::new((sel::
//! FORK_BUTTON_TAG, i))`, …) lives OUTSIDE this module in `row_text::sel`
//! so the module's bodies stay literal-free apart from diagnostic macros.
//! Every selector-method call (`.debug_selector(...)`, `.id(...)`) is
//! fed by a `sel::*` const or helper's `String`, so removing the r2
//! method-name allowance did not require any renderer edits.

use gpui::{div, prelude::*, px, AnyElement, App, WeakEntity};
use gpui_kit::component::{
    alert::Alert,
    button::{Button, ButtonVariants},
    text::TextView,
    ActiveTheme, Disableable, Icon, IconName, StyledExt,
};

use super::{record_state, state_text, tool_state_color, ZetaView};
use crate::{polish, theme};
use zeta_gui::row_text::{
    self, sel, AssistantRowText, ErrorRowText, LoginActionText, LoginErrorText, LoginRowText,
    RowText, ThinkingRowText, ToolGroupRowText, ToolRowText, UserRowText,
};
use zeta_gui::state::TranscriptEntry;

/// The assistant row's markdown style, extracted so the ZETA-110 clipping
/// guards (scroll layout, per-cell nowrap, flat cell border, subtle inline
/// code) can be pinned by a test that fails if any of them regress.
///
/// Every field is sourced from the ambient `cx.theme()` so the values move
/// with the theme instead of the palette accessors — the gpui-kit
/// rich-text default is `HighlightStyle { background_color: theme.accent }`,
/// so a caller that skipped this override would paint the inline chip on
/// solid accent. Routing through `theme.secondary_hover` (~6% text-normal
/// wash) and `theme.foreground` mirrors the wiki `.markdown-preview-view
/// code` rule and keeps the whole app on one inline-code shape.
///
/// The `table` and `table_cell` refinements carry the load-bearing
/// clipping fix from round 1: `overflow.x = Scroll` grows every column to
/// its measured glyph width instead of the wrap layout's character-count
/// heuristic, and `white_space = Nowrap` on `table_cell` raises the
/// per-column floors so an inline-code chip's trailing glyph never lands
/// on the wrong side of the cell's `overflow_hidden()`. The transparent
/// `border_color` on the cell keeps rows reading flat; the row-bottom
/// rules ride on the row `div`, not the cell.
pub(crate) fn assistant_markdown_style(cx: &App) -> gpui_kit::component::text::TextViewStyle {
    let theme = cx.theme();
    let code_block = gpui::StyleRefinement::default()
        .py(px(12.))
        .px(px(16.))
        .border_1()
        .border_color(theme.border)
        .whitespace_normal();
    // Inline `code` sits on a subtle text-normal wash with normal-tier
    // text, matching the wiki's `.markdown-preview-view code` rule.
    // Routing through the theme means a future TextView caller that
    // reuses this helper — or that inherits the gpui-kit rich-text
    // default we override at render time — never paints the chip on
    // the raw accent slab the component library ships as its default.
    let inline_code = gpui::HighlightStyle {
        background_color: Some(theme.secondary_hover),
        color: Some(theme.foreground),
        ..Default::default()
    };
    // Tables opt into gpui-base's SCROLL layout so column widths come
    // from the shaped text of each cell instead of the wrap layout's
    // character-count heuristic. Wrap layout budgets columns by
    // character count and clamps the cell to `overflow_hidden`; on a
    // proportional glyph run — inline code chips scaled to 0.875 plus
    // 4px padding — that budget starves narrow columns and the trailing
    // glyph disappears (`bas`, `tod`, `rea`, `edi` in the smoke shot
    // instead of `bash`, `todo`, `read`, `edit`). Scroll mode grows
    // every column to its measured content and only scrolls when the
    // total content exceeds the transcript column.
    let table = gpui::StyleRefinement {
        overflow: gpui::PointRefinement {
            x: Some(gpui::Overflow::Scroll),
            y: None,
        },
        ..Default::default()
    };
    // Cell refinement:
    //   * transparent border — kills the per-cell vertical grid so the
    //     table reads as flat rows (row bottom rules ride on the row
    //     div, not the cell, and survive this override), matching the
    //     wiki's "header rule at most" look.
    //   * white-space nowrap — in the scroll layout, per gpui-base's
    //     own docs, nowrap on `style.table_cell` "keeps the cell text
    //     on a single line, and the floors are raised to the full
    //     content widths so the single-line columns never shrink."
    //     That is the load-bearing fix for the inline-code chip
    //     clipping: with nowrap, the Tool column's floor becomes the
    //     shaped width of the widest chip (with its 4px padding) plus
    //     the cell's own padding, so the chip's trailing glyph always
    //     lands inside the column instead of getting sliced off by
    //     the cell's `overflow_hidden()`.
    let mut table_cell = gpui::StyleRefinement {
        border_color: Some(gpui::transparent_black()),
        ..Default::default()
    };
    table_cell.text.white_space = Some(gpui::WhiteSpace::Nowrap);
    gpui_kit::component::text::TextViewStyle {
        code_block,
        table,
        table_cell,
        inline_code,
        ..Default::default()
    }
}

impl ZetaView {
    pub(crate) fn render_row(&self, index: usize, view: WeakEntity<Self>, cx: &App) -> AnyElement {
        let is_first = index == 0;
        let is_last = index + 1 == self.state.transcript.len();
        let this_is_tool = matches!(self.state.transcript[index], TranscriptEntry::Tool { .. });
        let next_is_tool = self
            .state
            .transcript
            .get(index + 1)
            .is_some_and(|entry| matches!(entry, TranscriptEntry::Tool { .. }));
        // Adjacent tool rows collapse the row gap so a run of receipts reads
        // as one column — matches the wiki agent-run rhythm exactly. The last
        // row also carries no gap so column bottom padding lands cleanly.
        let row_gap = if is_last || (this_is_tool && next_is_tool) {
            px(0.)
        } else {
            theme::TRANSCRIPT_ROW_GAP
        };

        let inner = self.render_row_inner(index, view, cx);
        // Prose rows (user, assistant, thinking) cap at the narrower reading
        // measure so long assistant lines wrap at a comfortable ~88ch. Tool
        // receipts, error blocks, and any other row keep the wider
        // `TRANSCRIPT_MAX_WIDTH` so a long tool command line or an error
        // stack has room.
        //
        // r2 clarification (ZETA-124 finding 4): a fenced code block INSIDE
        // an assistant markdown row rides the SAME prose cap as the prose
        // around it — the cap sits on the row wrapper, not on the child
        // markdown segments, so a wider fenced block would need a per-block
        // split renderer we deliberately do not add here. Split rendering
        // would give code fences a second column boundary of their own and
        // fight the reading rhythm the prose cap is here to establish;
        // tool receipts / error blocks already carry the wide cap for the
        // shell / stack output that actually benefits from horizontal
        // room. See `assistant_code_fence_rides_the_prose_cap` for the
        // test that pins this shape so a peer refactor that quietly
        // reintroduces block-aware sizing lands next to the review note
        // rather than as a surprise.
        //
        // The measure scales with the appearance picker's base font so an
        // 18px reader keeps the same character budget on screen.
        let prose_row = matches!(
            self.state.transcript[index],
            TranscriptEntry::User(_) | TranscriptEntry::Assistant(_) | TranscriptEntry::Thinking
        );
        let max_width = if prose_row {
            theme::prose_max_width(cx.theme().font_size)
        } else {
            theme::TRANSCRIPT_MAX_WIDTH
        };
        div()
            .debug_selector(|| sel::TRANSCRIPT_ROW.into())
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .items_center()
            // Column top/bottom padding lives on the first/last row so it
            // travels with the virtual scroller — a wrapper around the
            // scroller would leave the padding fixed while rows scroll under.
            .when(is_first, |row| row.pt_4())
            .when(is_last, |row| row.pb_3())
            .pb(row_gap)
            .child(
                div()
                    .debug_selector(|| sel::TRANSCRIPT_COLUMN.into())
                    .w_full()
                    .min_w_0()
                    .max_w(max_width)
                    .px_4()
                    .child(inner),
            )
            .into_any_element()
    }

    // Every renderer BELOW paints only through:
    //   - fields destructured from `row_text::RowText` / `LoginRowText`
    //   - references to `row_text::chrome` constants (via the model)
    //   - selectors produced by `row_text::sel::*` (widget IDs are not
    //     user-visible; they live outside this module so the AST fence
    //     stays strict on "no bare literal anywhere in expression
    //     position outside a diagnostic / matches! macro payload")
    //
    // A dropped field trips clippy's `unused_variables` under the crate's
    // `deny(warnings)`; a new bare literal trips the AST fence.
    pub(crate) fn render_row_inner(
        &self,
        index: usize,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let entry = &self.state.transcript[index];
        // Tool receipts that sit inside a run of 3+ collapse into one
        // group summary row on the FIRST index; the interior indices paint
        // an empty spacer so the virtual-list index math stays 1:1 with
        // `TranscriptEntry` indices. See `AppState::tool_group_position`
        // and `row_text::build_tool_group`.
        if let Some(group) = self.state.tool_group_position(index) {
            let expanded = self.state.is_tool_group_expanded(&group);
            if !expanded && !group.is_start(index) {
                return self.render_tool_group_hidden(index, cx);
            }
            if !expanded && group.is_start(index) {
                let excerpts: Vec<&str> = (group.first_index..=group.last_index)
                    .filter_map(|i| match self.state.transcript.get(i) {
                        Some(TranscriptEntry::Tool { excerpt, .. }) => Some(excerpt.as_str()),
                        _ => None,
                    })
                    .collect();
                let total_bytes = self.state.tool_group_output_bytes(&group);
                let text = row_text::build_tool_group(
                    &excerpts,
                    total_bytes,
                    zeta_gui::state::TOOL_GROUP_PREVIEW_MAX,
                    false,
                );
                return self.render_tool_group_row(index, group, text, view, cx);
            }
        }
        let text = row_text::build(
            entry,
            index,
            &self.state.session_view,
            self.state.session_view.available,
        );
        match text {
            RowText::User(text) => self.render_user_row(index, text, view, cx),
            RowText::Assistant(text) => self.render_assistant_row(index, text, cx),
            RowText::Tool(text) => self.render_tool_row(index, text, entry, view, cx),
            RowText::Thinking(text) => self.render_thinking_row(index, text, cx),
            RowText::Error(text) => {
                let TranscriptEntry::Error { login_provider, .. } = entry else {
                    unreachable!("row-text Error variant maps to TranscriptEntry::Error")
                };
                self.render_error_row(index, text, login_provider.as_deref(), view, cx)
            }
            RowText::ToolGroup(_) | RowText::ToolGroupHidden => {
                unreachable!("group variants are dispatched by render_row_inner directly")
            }
        }
    }

    fn render_thinking_row(&self, index: usize, text: ThinkingRowText, cx: &App) -> AnyElement {
        // Header-only marker at muted-foreground. The header text comes
        // from the typed model; a sentinel-carrying reasoning payload
        // cannot land here because `Thinking` carries no body.
        let ThinkingRowText { header } = text;
        let color = cx.theme().muted_foreground;
        // state_text records (row_id, color) into the render_log at the
        // exact moment the color is applied — a mutation that swaps the
        // color argument at this call site is caught by the sample check.
        state_text(|| sel::thinking_header(index), color)
            .debug_selector(move || sel::thinking_header(index))
            .w_full()
            .min_w_0()
            .py(px(2.))
            .child(header)
            .into_any_element()
    }

    fn render_user_row(
        &self,
        index: usize,
        text: UserRowText<'_>,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let UserRowText {
            content,
            attachments,
            fork_label,
        } = text;
        let content = content.to_owned();
        let fork_id =
            fork_label.and_then(|_| self.state.session_view.message_ids.get(&index).cloned());
        let group = sel::user_row_group(index);
        div()
            .group(group.clone())
            .v_flex()
            .child(
                // Rectangle + hover-reveal fork button live in the SAME layout
                // cell (relative parent, absolute button). The invisible button
                // no longer reserves a phantom row that breaks the 14px rhythm.
                div()
                    .relative()
                    .w_full()
                    .min_w_0()
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .py_2()
                            .px_3()
                            .bg(cx.theme().muted)
                            .border_l(theme::RAIL_WIDTH_THICK)
                            .border_color(cx.theme().primary)
                            .whitespace_normal()
                            .child(content),
                    )
                    .when_some(fork_id.zip(fork_label), |row, (id, label)| {
                        let click_id = id.clone();
                        let click_view = view.clone();
                        row.child(
                            div()
                                .absolute()
                                .top_1()
                                .right_1()
                                .opacity(0.)
                                .group_hover(group.clone(), |style| style.opacity(1.))
                                .child(
                                    Button::new((sel::FORK_BUTTON_TAG, index))
                                        .debug_selector(move || sel::fork_button(index))
                                        .ghost()
                                        .compact()
                                        .label(label)
                                        .on_click(move |_, _, cx| {
                                            let id = click_id.clone();
                                            let _ = click_view
                                                .update(cx, |view, cx| view.fork_message(id, cx));
                                        }),
                                ),
                        )
                    }),
            )
            // Attachment tri-state: `Some(list)` — even `Some(vec![])` —
            // paints the mt_1 container so the row height matches text-only
            // history rows before the typed-seam refactor. `None` skips the
            // container entirely. See `UserRowText::attachments` docs.
            .when_some(attachments, |row, list| {
                row.child(
                    div().mt_1().h_flex().flex_wrap().gap_2().children(
                        list.into_iter()
                            .enumerate()
                            .map(|(attachment_index, label)| {
                                div()
                                    .debug_selector(|| sel::ATTACHMENT_CHIP.into())
                                    .px_2()
                                    .py_1()
                                    .text_size(theme::label_small(cx.theme().font_size))
                                    .bg(cx.theme().muted)
                                    .h_flex()
                                    .items_center()
                                    .gap_2()
                                    .when_some(
                                        self.sent_images.get(&(index, attachment_index)).cloned(),
                                        |chip, image| chip.child(polish::thumbnail(image, cx)),
                                    )
                                    .child(label)
                            }),
                    ),
                )
            })
            .into_any_element()
    }

    fn render_assistant_row(
        &self,
        index: usize,
        text: AssistantRowText<'_>,
        cx: &App,
    ) -> AnyElement {
        // Naked assistant turn: 2px vertical breath, no bg, no border, no rail.
        // Hierarchy is carried by weight + color tier + rails on OTHER row types,
        // not by framing the assistant. Prose sits at 1.65 line-height for the
        // wiki reading rhythm; code fences carry 12x16 padding, a 1px border,
        // and soft-wrap so long lines never introduce a horizontal scroll.
        let AssistantRowText {
            source,
            truncated_hint,
        } = text;
        // Text-run recorder: shape the raw source at the same wrap width
        // the prose row hands to the text system, and record the widest
        // wrap-line. See `crate::record_text_geometry` for why this seam
        // is needed — `painted_quads()` cannot observe glyph sprites, so
        // a shape that produces a line wider than the column's content
        // box is invisible to every earlier fix pass. Gated behind test /
        // smoke-test so release builds pay nothing.
        #[cfg(any(test, feature = "smoke-test"))]
        {
            let font_size = cx.theme().font_size;
            let wrap_width = theme::prose_text_measure(font_size);
            let font = gpui::font(theme::current_font_family());
            crate::record_text_geometry(
                cx,
                || sel::message(index),
                source,
                font,
                font_size,
                wrap_width,
            );
        }
        let source = source.to_owned();
        let text_view = TextView::markdown(sel::message(index), source)
            .selectable(true)
            .style(assistant_markdown_style(cx));
        #[cfg(feature = "smoke-test")]
        let text_view =
            if std::env::var_os(row_text::sel::NATIVE_GUARD_FORCE_TEXT_WIDTH_ENV).is_some() {
                // Recreate the round-3 evasion: give the live TextView a wider
                // available width while its prose column remains narrow.
                text_view.w(px(1200.))
            } else {
                // Leave a small layout margin for fractional glyph advances. This
                // changes the wrap budget; it does not clip painted pixels.
                text_view.max_w(px(f32::from(theme::prose_max_width(cx.theme().font_size))
                    - 2. * theme::PROSE_ROW_PADDING_X
                    - 2.))
            };
        #[cfg(not(feature = "smoke-test"))]
        let text_view = text_view
            .max_w(px(f32::from(theme::prose_max_width(cx.theme().font_size))
                - 2. * theme::PROSE_ROW_PADDING_X
                - 2.));
        div()
            .py(px(2.))
            .w_full()
            .min_w_0()
            .line_height(gpui::rems(1.65))
            .when_some(truncated_hint, |row, hint| row.child(hint))
            .child(text_view)
            .into_any_element()
    }

    fn render_tool_row(
        &self,
        index: usize,
        text: ToolRowText<'_>,
        entry: &TranscriptEntry,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let ToolRowText {
            tool_label,
            excerpt,
            metadata_label,
            hover_hint,
            tail_omitted_hint,
            body,
        } = text;
        let tool_label = tool_label.to_owned();
        let excerpt = excerpt.to_owned();
        let body = body.map(str::to_owned);
        let is_error = entry.unsuccessful();
        // State is signalled by COLOR ONLY. Running sits at normal text tier;
        // done fades to muted; failed/canceled land on danger. The wiki
        // contract forbids any textual state marker — the color helper hands
        // us the token, and the render below applies it through `state_text`
        // / `record_state` on the chevron + excerpt call sites. A mutation
        // that swaps the color argument at any call site records the wrong
        // color and fails the sample check. The tool_label and metadata
        // paint at the muted-foreground tier regardless of state — they
        // are chrome, not payload.
        let state_color = tool_state_color(entry.tool_state(), cx);
        let group = sel::tool_row_group(index);
        let expanded = body.is_some();

        div()
            .group(group.clone())
            .id((sel::TOOL_RECEIPT_TAG, index))
            .debug_selector(move || sel::tool_receipt(index))
            .relative()
            .w_full()
            .min_w_0()
            .cursor_pointer()
            .hover(|style| style.bg(cx.theme().list_hover))
            .on_click(move |_, _, cx| {
                let _ = view.update(cx, |view, cx| {
                    view.state.toggle_card(index);
                    view.transcript.update(cx, |scroll, cx| {
                        scroll.remeasure_items(index..index + 1, cx);
                    });
                    cx.notify();
                });
            })
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .items_center()
                    .min_h(px(20.))
                    .child(
                        Icon::new(if expanded {
                            IconName::ChevronDown
                        } else {
                            IconName::ChevronRight
                        })
                        .size(theme::label_small(cx.theme().font_size))
                        .text_color(record_state(|| sel::tool_chevron(index), state_color)),
                    )
                    .child(
                        // Tool name label — SMALL and DIM. Sits to the LEFT
                        // of the excerpt so the reader answers "which tool"
                        // before "what it ran" without the label stealing
                        // weight from the primary text.
                        div()
                            .debug_selector(move || sel::tool_label(index))
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_size(theme::label_small(cx.theme().font_size))
                            .text_color(cx.theme().muted_foreground)
                            .child(tool_label),
                    )
                    .child(
                        // Excerpt — the row's PRIMARY text. Routes its state
                        // color through the recorder so a swap on this call
                        // site is caught by the render_log sample check.
                        state_text(|| sel::tool_excerpt(index), state_color)
                            .debug_selector(move || sel::tool_excerpt(index))
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .child(excerpt),
                    )
                    // Metadata (size/duration) sits DIRECTLY after the
                    // excerpt — laws-of-ux proximity. Painted at the muted
                    // tier so it reads as chrome, not payload, and small
                    // enough that it never fights the primary text.
                    .when_some(metadata_label, |row, label| {
                        row.child(
                            div()
                                .debug_selector(move || sel::tool_metadata(index))
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.78)
                                .text_size(theme::label_small(cx.theme().font_size))
                                .child(label),
                        )
                    })
                    .when_some(hover_hint, |row, hint| {
                        row.child(
                            div()
                                .debug_selector(move || sel::tool_hover_hint(index))
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.)
                                .group_hover(group.clone(), |style| style.opacity(0.78))
                                .text_size(theme::label_small(cx.theme().font_size))
                                .child(hint),
                        )
                    }),
            )
            .when_some(body, |row, body| {
                row.child(
                    div()
                        .debug_selector(move || sel::tool_output(index))
                        // Indent rail: margin 3/0/5, padding-left 8, 1px rail,
                        // panel fill — reads as a subordinate body without
                        // fighting the row's leading verb. Vertical padding sits
                        // at 2px per the wiki contract, not the 4px `.py_1()`.
                        .mt(px(3.))
                        .mb(px(5.))
                        .pl_2()
                        .py(px(2.))
                        .border_l(theme::RAIL_WIDTH_THIN)
                        .border_color(if is_error {
                            cx.theme().danger
                        } else {
                            cx.theme().border
                        })
                        .bg(cx.theme().sidebar)
                        .text_color(cx.theme().muted_foreground)
                        .when_some(tail_omitted_hint, |output, hint| {
                            output.child(div().opacity(0.7).child(hint))
                        })
                        .child(div().whitespace_normal().child(body)),
                )
            })
            .into_any_element()
    }

    /// Collapsed tool-group summary row (ZETA-125). Reads as one row
    /// "`N` tool calls · `<total>`" with the first 1-2 excerpts previewed
    /// so a run of receipts does not eat the transcript. The row is a real
    /// tab stop (ZETA-108 a11y precedent): Enter/Space toggles the group's
    /// expanded state and remeasures the run so the virtual list picks up
    /// the new heights. The visible state is carried in the accessible
    /// label (`collapsed`/`expanded`), so keyboard-only users hear the
    /// state that changes on activation.
    fn render_tool_group_row(
        &self,
        index: usize,
        group: zeta_gui::state::ToolGroupPosition,
        text: ToolGroupRowText<'_>,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let ToolGroupRowText {
            count_label,
            total_label,
            preview_excerpts,
            aria_label,
        } = text;
        let preview_excerpts: Vec<String> =
            preview_excerpts.iter().map(|s| (*s).to_owned()).collect();
        let first_id_click = group.first_id.clone();
        let first_id_key = group.first_id.clone();
        let range = group.first_index..=group.last_index;
        let range_click = range.clone();
        let range_key = range;
        let key_view = view.clone();
        let focus_handle = tool_group_focus_handle(self, cx, &group.first_id);
        div()
            .id((sel::TOOL_GROUP_TAG, index))
            .debug_selector(move || sel::tool_group_row(index))
            .track_focus(&focus_handle)
            .tab_index(0)
            .aria_label(aria_label)
            .role(gpui::accesskit::Role::Button)
            .relative()
            .w_full()
            .min_w_0()
            .cursor_pointer()
            .hover(|style| style.bg(cx.theme().list_hover))
            .on_click(move |_, _, cx| {
                let id = first_id_click.clone();
                let range = range_click.clone();
                let _ = view.update(cx, move |view, cx| {
                    view.state.toggle_tool_group(&id);
                    view.transcript.update(cx, |scroll, cx| {
                        scroll.remeasure_items(*range.start()..*range.end() + 1, cx);
                    });
                    cx.notify();
                });
            })
            .on_key_down(move |event: &gpui::KeyDownEvent, _, cx| {
                if !matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    return;
                }
                let id = first_id_key.clone();
                let range = range_key.clone();
                let _ = key_view.update(cx, move |view, cx| {
                    view.state.toggle_tool_group(&id);
                    view.transcript.update(cx, |scroll, cx| {
                        scroll.remeasure_items(*range.start()..*range.end() + 1, cx);
                    });
                    cx.notify();
                });
                cx.stop_propagation();
            })
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .items_center()
                    .min_h(px(20.))
                    .child(
                        Icon::new(IconName::ChevronRight)
                            .size(theme::label_small(cx.theme().font_size))
                            .text_color(cx.theme().muted_foreground),
                    )
                    .child(
                        div()
                            .debug_selector(move || sel::tool_group_count(index))
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_size(theme::label_small(cx.theme().font_size))
                            .text_color(cx.theme().foreground)
                            .child(count_label),
                    )
                    .child(
                        div()
                            .debug_selector(move || sel::tool_group_preview(index))
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .text_size(theme::label_small(cx.theme().font_size))
                            .text_color(cx.theme().muted_foreground)
                            .children(
                                preview_excerpts
                                    .into_iter()
                                    .map(|excerpt| div().flex_shrink_0().pl_2().child(excerpt)),
                            ),
                    )
                    .when_some(total_label, |row, label| {
                        row.child(
                            div()
                                .debug_selector(move || sel::tool_group_metadata(index))
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.78)
                                .text_size(theme::label_small(cx.theme().font_size))
                                .child(label),
                        )
                    }),
            )
            .into_any_element()
    }

    /// Interior row of a COLLAPSED tool group — paints nothing so the
    /// virtual-list index math stays 1:1 with `TranscriptEntry` indices.
    /// The row still occupies a slot in the list so downstream inserts and
    /// removes land at the right index; it just measures to zero height.
    fn render_tool_group_hidden(&self, index: usize, _cx: &App) -> AnyElement {
        div()
            .id((sel::TOOL_GROUP_HIDDEN_TAG, index))
            .debug_selector(move || sel::tool_group_row(index))
            .w_full()
            .min_w_0()
            .into_any_element()
    }

    fn render_error_row(
        &self,
        index: usize,
        text: ErrorRowText<'_>,
        login_provider: Option<&str>,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let ErrorRowText {
            header,
            message,
            settings_action_label,
        } = text;
        let message = message.to_owned();
        div()
            .debug_selector(move || sel::error_block(index))
            .v_flex()
            .gap_2()
            .pl_3()
            .py_1()
            .border_l(theme::RAIL_WIDTH_THICK)
            .border_color(cx.theme().danger)
            .child(
                div()
                    .text_color(cx.theme().danger)
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(header),
            )
            .child(
                div()
                    .debug_selector(move || sel::error_message(index))
                    .whitespace_normal()
                    .child(message),
            )
            .when_some(
                login_provider.and_then(|provider| {
                    self.login_providers
                        .iter()
                        .find(|row| row.provider == provider)
                }),
                |block, provider| {
                    let prefix = sel::error_login_prefix(index);
                    let login = row_text::build_login(provider, &prefix, &self.state.connection);
                    block.child(self.render_login_row(login, view.clone(), cx))
                },
            )
            .when_some(settings_action_label, |block, label| {
                block.child(
                    Button::new((sel::ERROR_SETTINGS_TAG, index))
                        .debug_selector(move || sel::error_settings(index))
                        .label(label)
                        .on_click(move |_, _, cx| {
                            let _ = view.update(cx, |view, cx| view.open_settings(cx));
                        }),
                )
            })
            .into_any_element()
    }

    /// Render one login-provider row from a fully-resolved `LoginRowText`.
    /// The caller composes the model — `provider.label()` and the composed
    /// button labels are computed by `row_text::build_login` — so this
    /// renderer never touches raw provider text. Fixes r1 finding 1: the
    /// pre-fix `render_login_row` read `provider.label()` at paint time
    /// and passed the raw label to `.child(...)`.
    ///
    /// r3 finding: `LoginRowText` is consumed BY VALUE via an exhaustive
    /// destructure with no `..`. Adding a field without extending this
    /// pattern fails to compile — that is the compiler-backed guarantee
    /// that every login-row field lands somewhere on the render path.
    pub(crate) fn render_login_row(
        &self,
        text: LoginRowText,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let LoginRowText {
            outer_id,
            header_label,
            provider_slug,
            start,
            cancel,
            status_text,
            error,
        } = text;
        let start_slug = provider_slug.clone();
        let cancel_slug = provider_slug;
        let start_view = view.clone();
        let cancel_view = view;
        div()
            .id(outer_id)
            .v_flex()
            .gap_2()
            .p_2()
            .child(
                div()
                    .h_flex()
                    .gap_3()
                    .items_center()
                    .justify_between()
                    .child(header_label)
                    .child(login_action_button(start, start_slug, start_view, true))
                    .when_some(cancel, |row, cancel| {
                        row.child(login_action_button(cancel, cancel_slug, cancel_view, false))
                    }),
            )
            .child(
                div()
                    .text_size(theme::label_small(cx.theme().font_size))
                    .whitespace_normal()
                    .text_color(cx.theme().muted_foreground)
                    .child(status_text),
            )
            .when_some(error, |row, error| {
                let LoginErrorText { id, message } = error;
                row.child(Alert::error(id, message))
            })
            .into_any_element()
    }
}

/// Fetch or create a persistent focus handle for a tool-group summary row.
/// Stored on `ZetaView::tool_group_focus` keyed by the first tool receipt's
/// `tool_call_id` so the handle survives redraws AND tests can look it up
/// by the same key the renderer uses. Matches the sidebar's row-focus
/// pattern (ZETA-108). `tab_stop(true).tab_index(0)` is required because
/// `cx.focus_handle()` defaults `tab_stop=false` and `.tab_index(0)` on
/// the div alone does not push through to a tracked focus handle — see the
/// same comment in `sidebar::row_focus_handle`.
fn tool_group_focus_handle(view: &ZetaView, cx: &App, first_id: &str) -> gpui::FocusHandle {
    let key = row_text::sel::tool_group_focus_key(first_id);
    let mut map = view.tool_group_focus.borrow_mut();
    if let Some(handle) = map.get(&key) {
        return handle.clone();
    }
    let handle = cx.focus_handle().tab_stop(true).tab_index(0);
    map.insert(key, handle.clone());
    handle
}

/// Compose one login action button. Kept out of `render_login_row` so the
/// method stays readable and the shared start/cancel machinery lives in
/// one spot. `is_start` picks the click callback — `start_login` vs
/// `cancel_login` — since the two share every visible field.
///
/// r3 finding: `LoginActionText` is consumed BY VALUE via an exhaustive
/// destructure with no `..` — a new field on the struct fails to compile
/// here until every field lands on the button.
fn login_action_button(
    action: LoginActionText,
    provider_slug: String,
    view: WeakEntity<ZetaView>,
    is_start: bool,
) -> Button {
    let LoginActionText {
        id,
        label,
        disabled,
    } = action;
    let selector_id = id.clone();
    Button::new(id)
        .debug_selector(move || selector_id.clone())
        .h(px(40.))
        .label(label)
        .disabled(disabled)
        .on_click(move |_, _, cx| {
            let slug = provider_slug.clone();
            let _ = view.update(cx, move |view, cx| {
                if is_start {
                    view.start_login(&slug, cx);
                } else {
                    view.cancel_login(&slug, cx);
                }
            });
        })
}
