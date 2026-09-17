//! Tool-receipt and tool-group renderers extracted from `transcript_render`
//! (ZETA-125 r4 finding 6). The parent module grew past ~1k lines while
//! the tool-shape work landed; this split keeps each renderer group in a
//! file that reads without scrolling and pins the fence discipline for
//! the extracted code EXACTLY like the parent.
//!
//! The AST-based `renderer_literal_fence` guard test parses this WHOLE
//! source in addition to `transcript_render.rs`, so a stray literal here
//! trips the same fence — `.child("x")` in this module fails a dedicated
//! mutation battery entry (r4 review's fence-scope requirement). Every
//! user-visible string still comes from the typed `row_text::RowText`
//! model; widget IDs still come from `row_text::sel::*`; state color
//! still routes through `record_state` / `state_text` in the crate root.

use gpui::{div, prelude::*, px, AnyElement, App, WeakEntity};
use gpui_kit::component::{ActiveTheme, Icon, IconName, StyledExt};

use super::{record_state, state_text, theme, tool_state_color, ZetaView};
use zeta_gui::row_text::{
    self, sel, DiffPaneText, EditDiffText, RowText, ToolGroupRowText, ToolRowText,
};
use zeta_gui::state::TranscriptEntry;

impl ZetaView {
    /// Build the row-text model for a Tool entry and dispatch through
    /// `render_tool_row`. Called from the group-expanded path so the
    /// header + first receipt stack in the same virtual-list slot
    /// without re-implementing the tool row's paint contract inline.
    pub(crate) fn render_tool_row_from_entry(
        &self,
        index: usize,
        entry: &TranscriptEntry,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let text = row_text::build(
            entry,
            index,
            &self.state.session_view,
            self.state.session_view.available,
        );
        match text {
            RowText::Tool(text) => self.render_tool_row(index, text, entry, view, cx),
            _ => unreachable!("caller filtered on TranscriptEntry::Tool"),
        }
    }

    pub(crate) fn render_tool_row(
        &self,
        index: usize,
        text: ToolRowText<'_>,
        entry: &TranscriptEntry,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let ToolRowText {
            kind_glyph,
            tool_label,
            excerpt,
            metadata_label,
            hover_hint,
            tail_omitted_hint,
            body,
            panel_header,
            edit_diff,
        } = text;
        let tool_label = tool_label.to_owned();
        let excerpt = excerpt.map(str::to_owned);
        let body = body.map(str::to_owned);
        let panel_header = panel_header.map(str::to_owned);
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
            // Hover → list_hover rest wash, pressed → list_active one
            // tint step stronger so a click on a receipt reads as
            // tactile (ZETA-126). Toggle fires on click; the pressed
            // style paints for the frame(s) the mouse is held down so
            // the receipt does not feel dead on activation.
            .hover(|style| style.bg(cx.theme().list_hover))
            .active(|style| style.bg(cx.theme().list_active))
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
                    .min_h(theme::TOOL_ROW_MIN_HEIGHT)
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
                        // ZETA-135 (Trait 1 — kind glyph). Painted BEFORE
                        // the tool label so scanning the transcript reads
                        // the row's KIND before its identity — shell/edit/
                        // fetch/gear (laws-of-ux Selective Attention +
                        // Chunking). Same tier as the tool label so the
                        // pair reads as one leading cluster.
                        div()
                            .debug_selector(move || sel::tool_kind_glyph(index))
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_size(theme::label_small(cx.theme().font_size))
                            .text_color(cx.theme().muted_foreground)
                            .child(kind_glyph),
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
                        // Excerpt + metadata + hover-hint sit in ONE inner
                        // cluster so metadata paints DIRECTLY after the
                        // excerpt's painted glyph end (laws-of-ux
                        // proximity). The cluster gets `flex_1 min_w_0` to
                        // consume the leftover row width; the excerpt
                        // inside is `flex_shrink min_w_0 truncate` (NO
                        // flex_1) so it sizes to its content and metadata
                        // sits immediately after it — the pre-r2 fix
                        // routed `flex_1` onto the excerpt itself, which
                        // pushed the metadata to the row's right edge
                        // ~1409px away.
                        div()
                            .h_flex()
                            .gap_2()
                            .items_center()
                            .min_w_0()
                            .flex_1()
                            .child(
                                // Excerpt — the row's PRIMARY text. State
                                // color routes through the recorder so a
                                // swap at this call site is caught by the
                                // render_log sample check. The receipt
                                // paints an empty string for the
                                // missing-argument state (`excerpt =
                                // None`), which visually drops the
                                // redundant primary text without changing
                                // the row's layout or the debug/record
                                // selector — the r2 review's typed state
                                // lives on the model (`ToolRowText::excerpt
                                // = Option`), not on whether this
                                // element paints.
                                state_text(|| sel::tool_excerpt(index), state_color)
                                    .debug_selector(move || sel::tool_excerpt(index))
                                    .min_w_0()
                                    .flex_shrink(1.0)
                                    .truncate()
                                    .child(excerpt.unwrap_or_default()),
                            )
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
                    ),
            )
            .when_some(body, |row, body| {
                // ZETA-135: the expanded body reads as an inset panel with a
                // file-path / command header bar at the top, matching the
                // wiki session-view look. The outer container carries a full
                // 1px border (all sides — thin_rail on `.left` keeps the
                // pre-ZETA-135 error-vs-neutral rail paint test happy) and
                // a subtle panel fill; the header row shows what ran (file
                // path for edit/read/write, command for bash, tool name for
                // MCP) at the foreground tier with a dim border-bottom
                // separator; the body sits below at the muted tier. Error
                // state still tints the whole panel's border in danger.
                let panel_border = if is_error {
                    cx.theme().danger
                } else {
                    cx.theme().border
                };
                row.child(
                    div()
                        .debug_selector(move || sel::tool_output(index))
                        .w_full()
                        .min_w_0()
                        .mt(px(4.))
                        .mb(px(6.))
                        .border_1()
                        .border_color(panel_border)
                        .bg(cx.theme().sidebar)
                        .when_some(panel_header, |panel, header| {
                            // ZETA-135 review r1 finding 4: a long
                            // command/path had wrapped across multiple
                            // lines here (`whitespace_normal()`) and blew
                            // the panel's top edge out. Constrain to ONE
                            // truncated row so the header always reads as
                            // a fixed chrome bar regardless of the
                            // command's width.
                            panel.child(
                                div()
                                    .debug_selector(move || sel::tool_panel_header(index))
                                    .w_full()
                                    .min_w_0()
                                    .px_2()
                                    .py(px(4.))
                                    .border_b_1()
                                    .border_color(cx.theme().border)
                                    .text_color(cx.theme().foreground)
                                    .truncate()
                                    .child(header),
                            )
                        })
                        .when_some(edit_diff, |panel, diff| {
                            panel.child(self.render_diff_card(index, diff, cx))
                        })
                        .child(
                            div()
                                .px_2()
                                .py(px(4.))
                                .text_color(cx.theme().muted_foreground)
                                .when_some(tail_omitted_hint, |output, hint| {
                                    output.child(div().opacity(0.7).child(hint))
                                })
                                .child(div().whitespace_normal().child(body)),
                        ),
                )
            })
            .into_any_element()
    }

    /// ZETA-135 (Trait 2 — diff card). Render the two typed diff panes
    /// under the panel header for an edit receipt. At wide viewports the
    /// panes sit side-by-side (`flex_1` split); below
    /// `theme::NARROW_DIFF_STACK_WIDTH` they stack full-width, remove
    /// above add, so the shrunk half-width layout does not collapse each
    /// pane's content to one glyph. Every visible string comes from the
    /// typed `EditDiffText` model built in `row_text::build`; the render
    /// body itself carries NO literals so the fence stays strict.
    fn render_diff_card(&self, index: usize, diff: EditDiffText, cx: &App) -> gpui::AnyElement {
        let roles = theme::diff_roles(cx);
        let text_size = theme::label_small(cx.theme().font_size);
        let EditDiffText {
            remove_pane,
            add_pane,
        } = diff;
        let stacked = theme::viewport_width() < theme::NARROW_DIFF_STACK_WIDTH;
        let style = DiffPaneStyle {
            gutter_bg: roles.gutter_bg,
            gutter_fg: roles.gutter_fg,
            text_color: roles.text,
            text_size,
            stacked,
        };
        let remove = diff_pane(
            remove_pane,
            sel::tool_diff_remove_pane(index),
            roles.remove_bg,
            style,
        );
        let add = diff_pane(
            add_pane,
            sel::tool_diff_add_pane(index),
            roles.add_bg,
            style,
        );
        let container = div()
            .debug_selector(move || sel::tool_diff_card(index))
            .w_full()
            .min_w_0()
            .items_stretch()
            .border_b_1()
            .border_color(cx.theme().border);
        if stacked {
            container
                .v_flex()
                .child(remove)
                .child(add)
                .into_any_element()
        } else {
            container
                .h_flex()
                .child(remove)
                .child(add)
                .into_any_element()
        }
    }

    /// Collapsed tool-group summary row (ZETA-125). Reads as one row
    /// "`N` tool calls · `<total>`" with the first 1-2 excerpts previewed
    /// so a run of receipts does not eat the transcript. The row is a real
    /// tab stop (ZETA-108 a11y precedent): Enter/Space toggles the group's
    /// expanded state and remeasures the run so the virtual list picks up
    /// the new heights. The visible state is carried in the accessible
    /// label (`collapsed`/`expanded`), so keyboard-only users hear the
    /// state that changes on activation.
    pub(crate) fn render_tool_group_row(
        &self,
        index: usize,
        group: zeta_gui::state::ToolGroupPosition,
        text: ToolGroupRowText<'_>,
        expanded: bool,
        view: WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let ToolGroupRowText {
            count_label,
            total_label,
            preview_excerpts,
            separator,
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
            // Same hover → pressed staircase as an individual tool
            // receipt (ZETA-126): group headers are the same kind of
            // list-row control and should read as a single family.
            .hover(|style| style.bg(cx.theme().list_hover))
            .active(|style| style.bg(cx.theme().list_active))
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
                    .min_h(theme::TOOL_ROW_MIN_HEIGHT)
                    .child(
                        // Chevron flips DOWN when expanded so the header
                        // reads as an active disclosure — the r2 review
                        // called out that the header used to vanish on
                        // expansion; the fix keeps the header and swaps
                        // the chevron to signal the state change.
                        Icon::new(if expanded {
                            IconName::ChevronDown
                        } else {
                            IconName::ChevronRight
                        })
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
                            .children(preview_excerpts.into_iter().map(|excerpt| {
                                // Each preview is prefixed by the chrome
                                // separator ("N tool calls · preview ·
                                // preview"). Painting the separator here
                                // keeps the row_text model's chrome
                                // constant load-bearing.
                                div()
                                    .h_flex()
                                    .flex_shrink_0()
                                    .items_center()
                                    .child(
                                        div()
                                            .flex_shrink_0()
                                            .text_color(cx.theme().muted_foreground)
                                            .opacity(0.55)
                                            .child(separator),
                                    )
                                    .child(div().flex_shrink_0().child(excerpt))
                            })),
                    )
                    .when_some(total_label, |row, label| {
                        // Separator + total sit at the row's trailing edge
                        // ("… · <total>"). The composed aria label already
                        // includes the same separator so a screen reader
                        // hears the same rhythm sighted users see.
                        row.child(
                            div()
                                .flex_shrink_0()
                                .h_flex()
                                .items_center()
                                .child(
                                    div()
                                        .flex_shrink_0()
                                        .text_color(cx.theme().muted_foreground)
                                        .opacity(0.55)
                                        .text_size(theme::label_small(cx.theme().font_size))
                                        .child(separator),
                                )
                                .child(
                                    div()
                                        .debug_selector(move || sel::tool_group_metadata(index))
                                        .flex_shrink_0()
                                        .pl_2()
                                        .text_color(cx.theme().muted_foreground)
                                        .opacity(0.78)
                                        .text_size(theme::label_small(cx.theme().font_size))
                                        .child(label),
                                ),
                        )
                    }),
            )
            .into_any_element()
    }

    /// Interior row of a COLLAPSED tool group — paints nothing so the
    /// virtual-list index math stays 1:1 with `TranscriptEntry` indices.
    /// The row still occupies a slot in the list so downstream inserts and
    /// removes land at the right index; it just measures to zero height.
    pub(crate) fn render_tool_group_hidden(&self, index: usize, _cx: &App) -> AnyElement {
        div()
            .id((sel::TOOL_GROUP_HIDDEN_TAG, index))
            .debug_selector(move || sel::tool_group_row(index))
            .w_full()
            .min_w_0()
            .into_any_element()
    }
}

/// Paint one pane of the ZETA-135 diff card (remove or add). Free function
/// so both call sites in `render_diff_card` share ONE shape; the pane's
/// identity (remove vs. add) is carried by the caller-supplied selector +
/// bg tint. The gutter carries the pre-composed line number from the typed
/// model — the pane never composes a string here.
/// Shared style bundle for `diff_pane`. Groups the theme role tokens +
/// text size + layout mode so both call sites hand the same struct
/// without the clippy `too_many_arguments` lint firing on the seam.
#[derive(Clone, Copy)]
struct DiffPaneStyle {
    gutter_bg: gpui::Hsla,
    gutter_fg: gpui::Hsla,
    text_color: gpui::Hsla,
    text_size: gpui::Pixels,
    stacked: bool,
}

fn diff_pane(
    pane: DiffPaneText,
    selector: String,
    pane_bg: gpui::Hsla,
    style: DiffPaneStyle,
) -> gpui::AnyElement {
    let DiffPaneText { lines } = pane;
    let DiffPaneStyle {
        gutter_bg,
        gutter_fg,
        text_color,
        text_size,
        stacked,
    } = style;
    // In the wide layout, each pane takes half the container (`flex_1`)
    // and truncates row-by-row. In the stacked narrow layout, each pane
    // takes the full width so lines stay readable — the shrunk half-
    // width layout was collapsing content to one glyph.
    let base = div()
        .debug_selector(move || selector.clone())
        .min_w_0()
        .bg(pane_bg);
    let base = if stacked {
        base.w_full()
    } else {
        base.flex_1()
    };
    base.flex()
        .flex_col()
        .text_color(text_color)
        .text_size(text_size)
        .children(lines.into_iter().map(move |(number, content)| {
            div()
                .w_full()
                .min_w_0()
                .h_flex()
                .items_start()
                .child(
                    div()
                        .flex_shrink_0()
                        .min_w(px(28.))
                        .px_2()
                        .bg(gutter_bg)
                        .text_color(gutter_fg)
                        .child(number),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .px_2()
                        .whitespace_normal()
                        .child(content),
                )
        }))
        .into_any_element()
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
