//! Small presentation helpers for the consumer GUI.
use base64::Engine;
use gpui::{div, prelude::*, px, App, Menu, MenuItem, StyledImage};
use gpui_kit::component::ActiveTheme;
use image::ImageDecoder;
use std::io::Cursor;
use std::sync::Arc;
use zeta_gui::{client::ToolCall, session::ImageAttachment, state::StatusMetrics};

gpui::actions!(zeta, [NewSession, About, Quit]);

pub const SENT_IMAGE_LIMIT: usize = 64;

pub fn init_menus(cx: &mut App) {
    cx.on_action(|_: &Quit, cx| cx.quit());
    cx.bind_keys([
        gpui::KeyBinding::new("cmd-n", NewSession, None),
        gpui::KeyBinding::new("cmd-q", Quit, None),
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
    if metrics.tokens.is_none() && metrics.cache_hit_rate.is_none() {
        return "Usage appears after the first turn".into();
    }
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
        .with_fallback(|| {
            div()
                .text_size(px(10.))
                .child("No preview")
                .into_any_element()
        })
}
