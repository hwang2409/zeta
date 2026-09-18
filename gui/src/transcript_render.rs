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
use gpui_kit::TestSupportExt as _;

use super::{state_text, ZetaView};
use crate::{polish, theme};
use zeta_gui::row_text::{
    self, sel, AssistantRowText, ErrorRowText, LoginActionText, LoginErrorText, LoginRowText,
    RowText, ThinkingRowText, TurnFooterText, UserRowText,
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
        // ZETA-133: every transcript row now shares ONE unified column at
        // `TRANSCRIPT_MAX_WIDTH`. Prose keeps its ~88ch reading measure and
        // tool receipts / error blocks keep the wide cap, but the split
        // now lives INSIDE the row (gutter + body) rather than on the
        // column's `max_w`, so all row kinds share the same LEFT edge
        // regardless of kind. The chevron + kind glyph on tool rows hangs
        // in the fixed `LEADING_GUTTER_WIDTH` gutter; prose / thinking /
        // error / footer rows leave that gutter empty so their content
        // starts at the same body left edge as tool receipts' TEXT (tool
        // name onward). See `render_row_inner` for the per-kind body /
        // gutter dispatch and `theme::prose_body_max_width` /
        // `theme::wide_body_max_width` for the two body caps.
        //
        // r2 clarification (ZETA-124 finding 4, preserved): a fenced code
        // block INSIDE an assistant markdown row rides the same prose cap
        // as its surrounding prose — the cap sits on the row body, not on
        // the child markdown segments. See
        // `assistant_code_fence_rides_the_prose_cap`.
        //
        // ZETA-135 (Trait 3 — turn footer): the LAST transcript row hosts
        // a quiet "provider · model · duration" strip below its content
        // so the completed conversation reads with a Peak-End cue (laws-
        // of-ux Peak-End). Data is sourced from the active session's
        // wire metadata + status metrics — every token is `Option` and
        // the footer suppresses itself when nothing survives. Under
        // ZETA-133 it hangs at the shared body left edge via
        // `render_turn_footer_row` (gutter + body pair) so the footer
        // reads under the prose column instead of at a receipt-column
        // left edge.
        let turn_footer = is_last.then(|| self.build_turn_footer()).flatten();
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
                    .max_w(theme::TRANSCRIPT_MAX_WIDTH)
                    .px_4()
                    .child(inner)
                    .when_some(turn_footer, |column, footer| {
                        column.child(self.render_turn_footer_row(footer, cx))
                    }),
            )
            .into_any_element()
    }

    fn build_turn_footer(&self) -> Option<TurnFooterText> {
        let session = self
            .state
            .active_session
            .as_deref()
            .and_then(|id| self.state.sessions.iter().find(|s| s.session_id == id));
        let provider_from_session = session.map(|s| s.provider.as_str());
        let model = self
            .state
            .metrics
            .model
            .as_deref()
            .or_else(|| session.map(|s| s.model.as_str()));
        let created = session.map(|s| s.created_at.as_str()).unwrap_or_default();
        let updated = session.map(|s| s.updated_at.as_str()).unwrap_or_default();
        row_text::build_turn_footer(provider_from_session, model, created, updated)
    }

    fn render_turn_footer_row(&self, footer: TurnFooterText, cx: &App) -> AnyElement {
        let TurnFooterText { display, .. } = footer;
        // ZETA-133: the footer strip hangs at the shared BODY left edge so
        // the "provider · model · duration" text lines up under the prose
        // column instead of a receipt-column left edge. The row carries an
        // empty leading gutter (chevron/glyph gutter reserved for tool
        // rows only) and a body cap that matches the prose measure so the
        // footer text wraps at ~88ch on a very small viewport instead of
        // spilling into the wide receipt column.
        //
        // The composed `display` string routes through the paint-text
        // recorder under `sel::TURN_FOOTER` so tests assert on WHAT
        // reached `.child(...)` — a renderer that stops painting the
        // display drops the recorder call and the sample disappears,
        // where the pre-fix model-rebuild test still passed.
        let body_cap = theme::prose_body_max_width(cx.theme().font_size);
        transcript_body_pair(
            /* gutter */ div().into_any_element(),
            /* body   */
            div()
                .debug_selector(|| sel::TURN_FOOTER.into())
                .w_full()
                .min_w_0()
                .pt_2()
                .text_size(theme::label_small(cx.theme().font_size))
                .text_color(cx.theme().muted_foreground)
                .child(crate::record_text_child(
                    || sel::TURN_FOOTER.into(),
                    display,
                ))
                .into_any_element(),
            body_cap,
            usize::MAX,
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
        // Grouping dispatch (ZETA-125 r2 fix):
        //  * collapsed + first index -> paint the group summary row.
        //  * collapsed + interior    -> paint nothing (zero-height spacer).
        //  * expanded + first index  -> paint the group header ABOVE the
        //    first receipt, stacked in the same virtual-list slot. This
        //    keeps the header (its tab stop, its Enter/Space toggle, and
        //    its accessible label) on screen while the group is expanded
        //    — the r2 review flagged the pre-fix behaviour where the
        //    header vanished on expansion and keyboard-only users lost
        //    the ability to collapse it.
        //  * expanded + interior     -> paint the receipt normally.
        if let Some(group) = self.state.tool_group_position(index) {
            let expanded = self.state.is_tool_group_expanded(&group);
            if !expanded && !group.is_start(index) {
                return self.render_tool_group_hidden(index, cx);
            }
            if group.is_start(index) {
                let excerpts: Vec<&str> = (group.first_index..=group.last_index)
                    .filter_map(|i| match self.state.transcript.get(i) {
                        Some(TranscriptEntry::Tool { excerpt, .. }) => excerpt.as_deref(),
                        _ => None,
                    })
                    .collect();
                let total_bytes = self.state.tool_group_output_bytes(&group);
                let text = row_text::build_tool_group(
                    &excerpts,
                    total_bytes,
                    zeta_gui::state::TOOL_GROUP_PREVIEW_MAX,
                    expanded,
                );
                let header =
                    self.render_tool_group_row(index, group, text, expanded, view.clone(), cx);
                if !expanded {
                    return header;
                }
                let receipt = self.render_tool_row_from_entry(index, entry, view, cx);
                // Wrapper stacks header + first receipt in ONE virtual-list
                // slot. It carries no debug selector — the header keeps
                // `sel::tool_group_row(index)` and the receipt keeps
                // `sel::tool_receipt(index)` so tests locate each element
                // by its own selector.
                return gpui::div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .child(header)
                    .child(receipt)
                    .into_any_element();
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
        //
        // ZETA-137 D1: the `+` affordance rides in the LEADING gutter (LEFT
        // of the shared body edge), aligned with tool rows' kind-glyph
        // column — `+` and `$` land in the same x. The `Thought` header
        // text starts at the shared body edge alongside `bash` / prose /
        // expanded panels. The gutter shape mirrors the tool row's
        // (h_flex, gap_2, items_center, min_h(TOOL_ROW_MIN_HEIGHT)) with
        // an empty leading spacer standing in for the chevron so the
        // second child — the marker — lands under the kind_glyph column.
        let ThinkingRowText { marker, header } = text;
        let color = cx.theme().muted_foreground;
        let font_size = cx.theme().font_size;
        // Gutter: empty chevron-slot spacer + `+` marker at the same
        // (semibold, label_small, muted) tier as the tool row's kind
        // glyph, so a vertical scan across the transcript reads `+` and
        // `$` at the same x.
        let gutter = div()
            .h_flex()
            .gap_2()
            .items_center()
            .min_h(theme::TOOL_ROW_MIN_HEIGHT)
            // Chevron-slot placeholder: a transparent chevron laid out with
            // the SAME gpui-kit Icon shape the tool row uses (see
            // `tool_receipts.rs`), so the second child (`+`) lands under
            // the tool row's kind-glyph column pixel-for-pixel across the
            // 11px→18px picker range. Painting a real Icon (with
            // transparent color) rather than a naked div avoids relying
            // on internal Icon padding math staying in sync.
            .child(
                Icon::new(IconName::ChevronRight)
                    .size(theme::label_small(font_size))
                    .text_color(gpui::transparent_black()),
            )
            .child(
                div()
                    .debug_selector(move || sel::thinking_marker(index))
                    .flex_shrink_0()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_size(theme::label_small(font_size))
                    .text_color(color)
                    .child(marker),
            )
            .into_any_element();
        // state_text records (row_id, color) into the render_log at the
        // exact moment the color is applied — a mutation that swaps the
        // color argument at this call site is caught by the sample check.
        let body = state_text(|| sel::thinking_header(index), color)
            .debug_selector(move || sel::thinking_header(index))
            .w_full()
            .min_w_0()
            .py(px(2.))
            .child(header)
            .into_any_element();
        let body_cap = theme::prose_body_max_width(font_size);
        transcript_body_pair(gutter, body, body_cap, index).into_any_element()
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
        let body_cap = theme::prose_body_max_width(cx.theme().font_size);
        let body = div()
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
            .into_any_element();
        transcript_body_pair(div().into_any_element(), body, body_cap, index).into_any_element()
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
            // ZETA-133: record the ACTUAL wrap constraint the TextView is
            // fed (`prose_wrap_budget`), not the pre-ZETA-133 shape's
            // `prose_text_measure` — under the D1 body-pair layout the
            // body has no interior padding, so the TextView's `.max_w`
            // IS the shaped wrap width, and pinning the recorder to that
            // budget keeps the ZETA-124 recorder assertions matching the
            // body's actual painted width.
            let wrap_width = theme::prose_wrap_budget(font_size);
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
        // The wrap budget the prose row hands to the TextView is
        // FLOORED (see `theme::prose_wrap_budget`). Fractional widths
        // would let the painter's rounding push one glyph's advance
        // one pixel past the column content edge — the r3 pixel-gutter
        // guard flagged that pattern at 11px on the 922×610 viewport.
        let text_wrap_budget = theme::prose_wrap_budget(cx.theme().font_size);
        #[cfg(feature = "smoke-test")]
        let text_view =
            if std::env::var_os(row_text::sel::NATIVE_GUARD_FORCE_TEXT_WIDTH_ENV).is_some() {
                // Mutation: shift a full-width probe line 8px past the live
                // rendered body edge. The body bounds stay unchanged, so
                // the native scan must observe the injected overflow.
                text_view.max_w(text_wrap_budget).relative().left(px(8.))
            } else {
                text_view.max_w(text_wrap_budget)
            };
        #[cfg(not(feature = "smoke-test"))]
        let text_view = text_view.max_w(text_wrap_budget);
        let body = div()
            .py(px(2.))
            .w_full()
            .min_w_0()
            .line_height(gpui::rems(1.65))
            .when_some(truncated_hint, |row, hint| row.child(hint))
            .child(text_view)
            .into_any_element();
        let body_cap = theme::prose_body_max_width(cx.theme().font_size);
        transcript_body_pair(div().into_any_element(), body, body_cap, index).into_any_element()
    }

    // Tool-receipt and tool-group renderers moved to
    // `crate::tool_receipts` in r4 (finding 6). See that module for
    // `render_tool_row_from_entry`, `render_tool_row`,
    // `render_tool_group_row`, and `render_tool_group_hidden`. The
    // fence's scanned set (see `renderer_literal_fence_*` tests) was
    // extended to include the extracted module so the split cannot
    // smuggle a literal past the ZETA-109 guard.

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
        let body = div()
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
            .into_any_element();
        // ZETA-133: error rows hang at the shared body left edge (empty
        // gutter). The danger rail sits inside the body's `border_l`, so
        // the rail lands at the body's left edge — the same x as every
        // other row's content, not at a receipt-column left edge. Error
        // blocks keep the wide cap so long stack output stays on one line
        // wherever it fits.
        transcript_body_pair(
            div().into_any_element(),
            body,
            theme::wide_body_max_width(),
            index,
        )
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
        // The modal shows both providers at once; keep those rows compact
        // while preserving the roomier shape on banners and prompts.
        let compact = outer_id.starts_with("settings-login-");
        let mut row = div().id(outer_id).v_flex();
        if compact {
            row = row.gap_1().px_2().py_1();
        } else {
            row = row.gap_2().p_2();
        }
        row.child(
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

/// ZETA-133: compose the gutter + body pair every transcript row now
/// wraps its content in. The outer `flex-row` places a fixed
/// `LEADING_GUTTER_WIDTH` gutter (chevron + kind glyph for tool rows,
/// empty for prose / thinking / error / footer / login) to the LEFT of a
/// `TRANSCRIPT_BODY` body div whose `max_w` sets the row kind's right cap
/// (prose measure OR wide `TRANSCRIPT_MAX_WIDTH` residue). Every row kind
/// pipes its content through this helper so the LEFT edge of the body is
/// the same for every kind — the D1 fix.
///
/// Callers wrap this Div with their own interactivity (`.id()`, `.group()`,
/// `.on_click(...)`, `.hover(...)`, `.track_focus(...)`, ...) so the
/// clickable / focusable region spans both gutter and body — a click on
/// the tool receipt's chevron and a click on its label both fire the
/// receipt's expand toggle.
pub(crate) fn transcript_body_pair(
    gutter: AnyElement,
    body: AnyElement,
    body_max_width: gpui::Pixels,
    body_index: usize,
) -> gpui::Div {
    div()
        .flex()
        .items_start()
        .w_full()
        .min_w_0()
        .child(
            div()
                .debug_selector(|| sel::TRANSCRIPT_GUTTER.into())
                .w(theme::LEADING_GUTTER_WIDTH)
                .flex_shrink_0()
                .child(gutter),
        )
        .child(
            div()
                .debug_selector(|| sel::TRANSCRIPT_BODY.into())
                .min_w_0()
                .flex_1()
                .max_w(body_max_width)
                .child(body)
                .id((sel::TRANSCRIPT_BODY, body_index))
                .test_support(),
        )
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
