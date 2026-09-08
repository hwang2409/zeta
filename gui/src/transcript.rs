//! Thin render-tree adapter. Parsing and receipt state belong to the library.
use gpui::{
    div, font, prelude::*, px, rgb, AnyElement, FontStyle, FontWeight, HighlightStyle, StyledText,
    TextRun,
};
use zeta_gui::{
    appearance::Appearance,
    markdown::{Block, BlockKind},
};

pub fn render_block(block: &Block, appearance: Appearance, id: String) -> AnyElement {
    let p = appearance.palette();
    if let BlockKind::Code(language) = &block.kind {
        let highlights = block.syntax.iter().map(|run| {
            (
                run.range.clone(),
                HighlightStyle {
                    color: Some(rgb(run.color(appearance)).into()),
                    // Never import a syntax theme's background.
                    background_color: None,
                    ..Default::default()
                },
            )
        });
        return div()
            .w_full()
            .min_w_0()
            .border_1()
            .border_color(rgb(p.border))
            .rounded(px(4.))
            .when(!language.is_empty(), |view| {
                view.child(
                    div()
                        .px_3()
                        .py_1()
                        .text_size(px(12.))
                        .text_color(rgb(p.muted))
                        .child(language.clone()),
                )
            })
            .child(
                div()
                    .id(id)
                    .debug_selector(|| "code-scroll".into())
                    .flex()
                    .w_full()
                    .overflow_x_scroll()
                    .p_3()
                    .child(
                        div()
                            .debug_selector(|| "code-content".into())
                            .flex_shrink_0()
                            .font_family("monospace")
                            .text_size(px(13.))
                            .line_height(px(20.))
                            .whitespace_nowrap()
                            .child(StyledText::new(block.text()).with_highlights(highlights)),
                    ),
            )
            .into_any_element();
    }
    if let BlockKind::List(start) = block.kind {
        return div()
            .flex()
            .flex_col()
            .gap_1()
            .children(block.children.iter().enumerate().map(|(index, item)| {
                div()
                    .flex()
                    .items_start()
                    .gap_2()
                    .child(
                        div()
                            .w(px(28.))
                            .flex_shrink_0()
                            .text_color(rgb(p.muted))
                            .child(
                                start.map_or_else(
                                    || "•".into(),
                                    |n| format!("{}.", n + index as u64),
                                ),
                            ),
                    )
                    .child(div().flex_1().min_w_0().child(render_block(
                        item,
                        appearance,
                        format!("{id}-{index}"),
                    )))
            }))
            .into_any_element();
    }
    if block.kind == BlockKind::Rule {
        return div().h(px(1.)).my_2().bg(rgb(p.border)).into_any_element();
    }
    let heading = match block.kind {
        BlockKind::Heading(level) => Some(level),
        _ => None,
    };
    let size = match heading {
        Some(1) => 28.,
        Some(2) => 24.,
        Some(3) => 20.,
        Some(4) => 18.,
        Some(_) => 16.,
        None => 14.,
    };
    let runs = block
        .spans
        .iter()
        .map(|span| {
            let mut font = font(if span.code {
                "monospace"
            } else {
                "Helvetica Neue"
            });
            font.weight = if span.bold || heading.is_some_and(|n| n <= 2) {
                FontWeight::BOLD
            } else if heading.is_some() {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            };
            if span.italic {
                font.style = FontStyle::Italic;
            }
            TextRun {
                len: span.text.len(),
                font,
                color: rgb(p.text).into(),
                background_color: span.code.then(|| rgb(p.code_chip).into()),
                underline: None,
                strikethrough: None,
            }
        })
        .collect();
    div()
        .flex()
        .flex_col()
        .gap_2()
        .min_w_0()
        .text_size(px(size))
        .line_height(px(size + 8.))
        .when(heading.is_some(), |view| view.pt_2())
        .when(block.kind == BlockKind::Quote, |view| {
            view.pl_4().border_l_1().border_color(rgb(p.border))
        })
        .when(!block.spans.is_empty(), |view| {
            view.child(StyledText::new(block.text()).with_runs(runs))
        })
        .children(
            block
                .children
                .iter()
                .enumerate()
                .map(|(index, child)| render_block(child, appearance, format!("{id}-{index}"))),
        )
        .into_any_element()
}
