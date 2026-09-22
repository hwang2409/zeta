//! Small presentation helpers for the consumer GUI.
use base64::Engine;
use gpui::{div, prelude::*, px, App, Menu, MenuItem, StyledImage};
use gpui_kit::component::ActiveTheme;
use image::ImageDecoder;
use std::io::Cursor;
use std::sync::Arc;
use zeta_gui::{client::ToolCall, session::ImageAttachment, state::StatusMetrics};

use crate::theme;

gpui::actions!(
    zeta,
    [
        NewSession,
        About,
        Quit,
        ComposerFocusNext,
        ComposerFocusPrev
    ]
);

pub const SENT_IMAGE_LIMIT: usize = 64;

pub fn init_menus(cx: &mut App) {
    cx.on_action(|_: &Quit, cx| cx.quit());
    // Override gpui-base's `Input` Tab / Shift-Tab bindings so the
    // composer participates in keyboard traversal like every other
    // primary chat composer (Slack / Discord / prompt boxes). The
    // upstream default binds Tab to `IndentInline`, which traps
    // keyboard-only users inside the composer and silently breaks
    // app-wide tab traversal (a real WCAG 2.1.2 no-keyboard-trap
    // violation). Block indent / outdent stays reachable through
    // `cmd-]` / `cmd-[` (gpui-base already binds those). Registered
    // AFTER `gpui_kit::init(cx)` so ours wins by keymap order.
    cx.bind_keys([
        gpui::KeyBinding::new("cmd-n", NewSession, None),
        gpui::KeyBinding::new("cmd-q", Quit, None),
        gpui::KeyBinding::new("tab", ComposerFocusNext, Some("Input")),
        gpui::KeyBinding::new("shift-tab", ComposerFocusPrev, Some("Input")),
    ]);
    cx.set_menus([
        Menu::new("zeta").items([
            MenuItem::action("About zeta", About),
            MenuItem::separator(),
            MenuItem::action("Quit zeta", Quit),
        ]),
        Menu::new("File").items([MenuItem::action("New Session", NewSession)]),
    ]);
}

/// Header band 2 status line — usage/cache metadata WITHOUT the model prefix.
///
/// The model name is pinned to the right side of the same band on its own,
/// so folding it into the middle slice paints the model twice on every
/// screen. Keeping it out here also frees the middle slice to shrink and
/// truncate before the pinned model clips at the window edge.
pub fn status_label(metrics: &StatusMetrics) -> String {
    format!(
        "{} tokens · {} cache",
        metrics.tokens_label(),
        metrics.cache_label()
    )
}

pub fn approval_summary(call: &ToolCall) -> Option<String> {
    let summary = match call.name.as_str() {
        "bash" => call
            .arguments
            .get("command")
            .and_then(|value| value.as_str())
            .or_else(|| call.arguments.get("cmd").and_then(|value| value.as_str())),
        "read" | "write" | "edit" => call.arguments.get("path").and_then(|value| value.as_str()),
        _ => return None,
    };
    summary.map(|text| {
        text.chars()
            .map(|ch| if ch.is_control() { ' ' } else { ch })
            .take(240)
            .collect()
    })
}

/// Short byte-size label for composer chips ("512 B", "42 KB", "0.3 MB").
/// Kept alongside the composer thumbnail helper so the chip's two visible
/// pieces travel through the same module. Never returns a negative or a
/// fractional byte count.
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn image_source(image: &ImageAttachment) -> Option<Arc<gpui::Image>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&image.data)
        .ok()?;
    let mut decoder = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    let orientation = decoder.orientation().ok()?;
    let mut decoded = image::DynamicImage::from_decoder(decoder).ok()?;
    decoded.apply_orientation(orientation);
    // Keep only a static, 2x-resolution preview, including for animated inputs.
    let thumbnail = decoded.thumbnail(128, 96).into_rgba8();
    let mut output = Cursor::new(Vec::new());
    thumbnail
        .write_to(&mut output, image::ImageFormat::Png)
        .ok()?;
    Some(Arc::new(gpui::Image::from_bytes(
        gpui::ImageFormat::Png,
        output.into_inner(),
    )))
}

pub fn thumbnail(image: Arc<gpui::Image>, cx: &App) -> impl IntoElement {
    // Capture the picker's base font at call time; the closure paints later
    // and does not carry a `cx`. Routing through `theme::label_micro` at the
    // paint site lands the fallback caption on the same role every other
    // micro-tier text site uses so the type-role guard blesses it.
    let base = cx.theme().font_size;
    gpui::img(image)
        .debug_selector(|| "attachment-thumbnail".into())
        .w(px(64.))
        .h(px(48.))
        .object_fit(gpui::ObjectFit::Contain)
        .border_1()
        .border_color(
            if cx.theme().is_dark() {
                gpui::white()
            } else {
                gpui::black()
            }
            .opacity(0.1),
        )
        // The adjacent name and size remain available when decoding fails.
        .with_fallback(move || {
            div()
                .text_size(theme::label_micro(base))
                .child("No preview")
                .into_any_element()
        })
}
