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
    RowText, ThinkingRowText, ToolRowText, UserRowText,
};
use zeta_gui::state::TranscriptEntry;

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
                    .w_full()
                    .min_w_0()
                    .max_w(theme::TRANSCRIPT_MAX_WIDTH)
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
                                    .text_size(px(12.))
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
        let source = source.to_owned();
        let code_block = gpui::StyleRefinement::default()
            .py(px(12.))
            .px(px(16.))
            .border_1()
            .border_color(cx.theme().border)
            .whitespace_normal();
        // Inline `code` sits on a subtle text-normal wash with normal-tier
        // text, matching the wiki's `.markdown-preview-view code` rule. The
        // gpui-component default paints the chip on the full accent, which
        // reads as a solid violet slab in dark mode and steals attention
        // from real state chrome (pills, current-item text).
        let inline_code = gpui::HighlightStyle {
            background_color: Some(theme::palette::inline_code_bg()),
            color: Some(theme::palette::inline_code_fg()),
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
        let text_style = gpui_kit::component::text::TextViewStyle {
            code_block,
            table,
            table_cell,
            inline_code,
            ..Default::default()
        };
        div()
            .py(px(2.))
            .w_full()
            .min_w_0()
            .line_height(gpui::rems(1.65))
            .when_some(truncated_hint, |row, hint| row.child(hint))
            .child(
                TextView::markdown(sel::message(index), source)
                    .selectable(true)
                    .style(text_style),
            )
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
            verb,
            detail,
            output_size_label,
            hover_hint,
            tail_omitted_hint,
            body,
        } = text;
        let verb = verb.to_owned();
        let detail = detail.to_owned();
        let body = body.map(str::to_owned);
        let is_error = entry.unsuccessful();
        // State is signalled by COLOR ONLY. Running sits at normal text tier;
        // done fades to muted; failed/canceled land on danger. The wiki
        // contract forbids any textual state marker — the color helper hands
        // us the token, and the render below applies it through `state_text`
        // / `record_state` on the verb, detail, and chevron elements. A
        // mutation that swaps the color argument at any call site records
        // the wrong color and fails the sample check.
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
                        .size(px(12.))
                        .text_color(record_state(|| sel::tool_chevron(index), state_color)),
                    )
                    .child(
                        // Verb + detail route their state color through the
                        // recorder so a swap on this single call is caught
                        // by the render_log sample check.
                        state_text(|| sel::tool_verb(index), state_color)
                            .debug_selector(move || sel::tool_verb(index))
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(verb),
                    )
                    .child(
                        state_text(|| sel::tool_detail(index), state_color)
                            .debug_selector(move || sel::tool_detail(index))
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .opacity(0.78)
                            .child(detail),
                    )
                    // Collapsed rows carry the output size at faint tier and a
                    // hover-fade hint at hover — the visible affordance for
                    // the click-to-expand behaviour.
                    .when_some(output_size_label, |row, label| {
                        row.child(
                            div()
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.78)
                                .text_size(px(12.))
                                .child(label),
                        )
                    })
                    .when_some(hover_hint, |row, hint| {
                        row.child(
                            div()
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.)
                                .group_hover(group.clone(), |style| style.opacity(0.78))
                                .text_size(px(12.))
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
                    .text_size(px(12.))
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
