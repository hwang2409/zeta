//! Small presentation helpers for the consumer GUI.
use base64::Engine;
use gpui::{div, prelude::*, px, App, Menu, MenuItem, StyledImage};
use gpui_kit::component::ActiveTheme;
use std::sync::Arc;
use zeta_gui::{client::ToolCall, session::ImageAttachment, state::StatusMetrics};

gpui::actions!(zeta, [NewSession, About, Quit]);

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

pub fn status_label(metrics: &StatusMetrics) -> String {
    if metrics.model.is_none() && metrics.tokens.is_none() {
        return "Usage appears after the first turn".into();
    }
    format!(
        "{} · {} tokens · {} cache",
        metrics.model_label(),
        metrics.tokens_label(),
        metrics.cache_label()
    )
}

pub fn approval_summary(call: &ToolCall) -> Option<String> {
    let key = match call.name.as_str() {
        "bash" => "command",
        "read" | "write" | "edit" => "path",
        _ => return None,
    };
    call.arguments[key].as_str().map(|text| {
        text.chars()
            .map(|ch| if ch.is_control() { ' ' } else { ch })
            .take(240)
            .collect()
    })
}

pub fn image_source(image: &ImageAttachment) -> Option<Arc<gpui::Image>> {
    let format = gpui::ImageFormat::from_mime_type(&image.mime_type)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&image.data)
        .ok()?;
    Some(Arc::new(gpui::Image::from_bytes(format, bytes)))
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
