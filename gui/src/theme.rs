//! Zeta run-UI theme foundation.
//!
//! Maps the wiki app's shipped `opencode` palette onto the gpui-kit theme
//! surface so every zeta view inherits the same warm-dark, terminal-first
//! look as the wiki agent-run view. Colors, radii, typography, and focus
//! ring live here so per-surface code stays free of hardcoded hex.

use std::sync::Arc;

use gpui::{px, App, Hsla, Pixels};
use gpui_kit::component::{highlighter::HighlightTheme, Theme, ThemeMode};

/// Base UI type size — the "one size drives everything" pin from the wiki
/// run-UI extraction.
pub const FONT_SIZE: Pixels = px(15.);

/// Universal radius for zeta surfaces. Two pixels reads as "flat rectangles"
/// while still softening the corner just enough to avoid the raw-terminal
/// look.
pub const RADIUS: Pixels = px(2.);

/// Convert a 24-bit `0xRRGGBB` literal to Hsla.
fn hex(rgb: u32) -> Hsla {
    Hsla::from(gpui::rgb(rgb))
}

/// Convert a 32-bit `0xRRGGBBAA` literal to Hsla.
fn hex_a(rgba: u32) -> Hsla {
    Hsla::from(gpui::rgba(rgba))
}

/// The opencode palette values, one accessor per named token. Keeping these
/// as functions instead of consts sidesteps the const-eval limits on the
/// hex conversion while still letting tests import them by name.
pub mod palette {
    use super::{hex, hex_a};
    use gpui::Hsla;

    pub fn canvas() -> Hsla {
        hex(0x1e1e17)
    }
    pub fn panel() -> Hsla {
        hex(0x24241b)
    }
    pub fn element() -> Hsla {
        hex(0x2c2c21)
    }
    pub fn border() -> Hsla {
        hex(0x35352a)
    }
    pub fn border_active() -> Hsla {
        hex(0x706f62)
    }
    pub fn text() -> Hsla {
        hex(0xece9d8)
    }
    pub fn text_muted() -> Hsla {
        hex(0xa19e88)
    }
    pub fn text_faint() -> Hsla {
        hex(0x716f5e)
    }
    pub fn accent() -> Hsla {
        hex(0xb18bf4)
    }
    pub fn accent_hover() -> Hsla {
        hex(0xc6a9f7)
    }
    pub fn success() -> Hsla {
        hex(0xa9c957)
    }
    pub fn warning() -> Hsla {
        hex(0xd5d878)
    }
    pub fn danger() -> Hsla {
        hex(0xe2685c)
    }
    pub fn syntax_number() -> Hsla {
        hex(0xe29a5c)
    }
    pub fn syntax_type() -> Hsla {
        hex(0x7fc9b8)
    }
    pub fn hover() -> Hsla {
        hex_a(0xece9_d80f)
    }
    pub fn active() -> Hsla {
        hex_a(0xb18b_f424)
    }
    pub fn selection() -> Hsla {
        hex_a(0xece9_d829)
    }
    pub fn overlay_strong() -> Hsla {
        hex_a(0x0000_0073)
    }
    /// Solid scrollbar thumb. Wiki's opencode ships opaque olive-warm greys
    /// here so the thumb reads the same regardless of what surface it sits
    /// over.
    pub fn scrollbar_thumb() -> Hsla {
        hex(0x3a382c)
    }
    pub fn scrollbar_thumb_hover() -> Hsla {
        hex(0x4c4a3a)
    }
}

/// Opencode syntax palette, ported from `wiki/frontend/src/themes.css`.
///
/// gpui-kit's `ThemeStyle` keeps its `color`/`font_style`/`font_weight`
/// fields private and only exposes construction through Serde, so the
/// syntax map lives here as a JSON literal that parses into
/// `HighlightThemeStyle` at startup.
const HIGHLIGHT_STYLE_JSON: &str = r##"{
  "editor.background": "#1e1e17",
  "editor.foreground": "#ece9d8",
  "editor.active_line.background": "#24241b",
  "editor.line_number": "#716f5e",
  "editor.active_line_number": "#ece9d8",
  "editor.invisible": "#716f5e66",
  "editor.gutter.background": "#1e1e17",
  "syntax": {
    "attribute": { "color": "#7fc9b8" },
    "boolean":   { "color": "#e29a5c" },
    "comment":   { "color": "#716f5e", "font_style": "italic" },
    "comment.doc": { "color": "#716f5e", "font_style": "italic" },
    "constant":  { "color": "#e29a5c" },
    "constructor": { "color": "#d5d878" },
    "embedded":  { "color": "#ece9d8" },
    "emphasis":  { "color": "#ece9d8", "font_style": "italic" },
    "emphasis.strong": { "color": "#ece9d8", "font_weight": 700 },
    "enum":      { "color": "#7fc9b8" },
    "function":  { "color": "#d5d878" },
    "hint":      { "color": "#a19e88" },
    "keyword":   { "color": "#b18bf4" },
    "label":     { "color": "#d5d878" },
    "link_text": { "color": "#b18bf4" },
    "link_uri":  { "color": "#a19e88", "font_style": "italic" },
    "number":    { "color": "#e29a5c" },
    "operator":  { "color": "#b18bf4" },
    "preproc":   { "color": "#b18bf4" },
    "property":  { "color": "#ece9d8" },
    "punctuation": { "color": "#a19e88" },
    "punctuation.bracket": { "color": "#a19e88" },
    "punctuation.delimiter": { "color": "#a19e88" },
    "punctuation.list_marker": { "color": "#b18bf4" },
    "punctuation.special": { "color": "#b18bf4" },
    "string":    { "color": "#a9c957" },
    "string.escape": { "color": "#e29a5c" },
    "string.regex": { "color": "#a9c957" },
    "string.special": { "color": "#e29a5c" },
    "string.special.symbol": { "color": "#e29a5c" },
    "tag":       { "color": "#b18bf4" },
    "tag.doctype": { "color": "#716f5e" },
    "text.code.span": { "color": "#a9c957" },
    "text.literal": { "color": "#ece9d8" },
    "title":     { "color": "#ece9d8", "font_weight": 700 },
    "type":      { "color": "#7fc9b8" },
    "variable":  { "color": "#ece9d8" },
    "variable.special": { "color": "#e29a5c" },
    "variant":   { "color": "#7fc9b8" }
  }
}"##;

fn opencode_highlight_theme() -> Arc<HighlightTheme> {
    Arc::new(HighlightTheme {
        name: "Opencode Zeta".to_string(),
        appearance: ThemeMode::Dark,
        style: serde_json::from_str(HIGHLIGHT_STYLE_JSON)
            .expect("opencode highlight style JSON is well-formed"),
    })
}

/// Apply the opencode palette + shape + typography rhythm onto the global
/// theme, then push the update down to the Base layer so scrollbars and
/// resize handles paint with the same tokens.
///
/// The gpui-kit `Theme` ships every button, table, chart, and diagnostic
/// field pre-populated with the shadcn light defaults. Rebuilding
/// `theme.tokens` from `theme.colors` at the end only refreshes the fields
/// listed below, so any field left untouched keeps painting light —
/// hence the exhaustive assignment.
pub fn apply(cx: &mut App) {
    let theme = Theme::global_mut(cx);
    theme.mode = ThemeMode::Dark;

    theme.font_family = "JetBrains Mono".into();
    theme.mono_font_family = "JetBrains Mono".into();
    theme.font_size = FONT_SIZE;
    theme.mono_font_size = FONT_SIZE;

    theme.radius = RADIUS;
    theme.radius_lg = RADIUS;
    theme.tile_radius = px(0.);
    theme.shadow = false;
    theme.tile_shadow = false;
    theme.focus_ring = true;

    // Install the opencode syntax palette BEFORE `sync_base`, so
    // `install_text_view_defaults` snapshots it into the code-block
    // highlighter used by markdown TextView renders.
    theme.highlight_theme = opencode_highlight_theme();

    let colors = &mut theme.colors;

    colors.background = palette::canvas();
    colors.foreground = palette::text();
    colors.border = palette::border();
    colors.drag_border = palette::border_active();
    colors.input = palette::border();

    colors.muted = palette::element();
    colors.muted_foreground = palette::text_muted();

    colors.accent = palette::accent();
    // Solid canvas on solid violet — matches wiki's `--text-on-accent`
    // and keeps menu items / accent chips at readable contrast.
    colors.accent_foreground = palette::canvas();

    colors.primary = palette::accent();
    colors.primary_foreground = palette::canvas();
    colors.primary_active = palette::accent();
    colors.primary_hover = palette::accent_hover();

    colors.secondary = palette::element();
    colors.secondary_foreground = palette::text();
    colors.secondary_active = palette::panel();
    colors.secondary_hover = palette::hover();

    // Default (unstyled) button follows the secondary look — a warm
    // element sitting on the panel with subtle hover / darker active.
    colors.button = palette::element();
    colors.button_foreground = palette::text();
    colors.button_hover = palette::hover();
    colors.button_active = palette::panel();

    colors.button_primary = palette::accent();
    colors.button_primary_foreground = palette::canvas();
    colors.button_primary_hover = palette::accent_hover();
    colors.button_primary_active = palette::accent();

    colors.button_secondary = palette::element();
    colors.button_secondary_foreground = palette::text();
    colors.button_secondary_hover = palette::hover();
    colors.button_secondary_active = palette::panel();

    colors.button_danger = palette::danger();
    colors.button_danger_foreground = palette::canvas();
    colors.button_danger_hover = palette::danger();
    colors.button_danger_active = palette::danger();

    colors.button_warning = palette::warning();
    colors.button_warning_foreground = palette::canvas();
    colors.button_warning_hover = palette::warning();
    colors.button_warning_active = palette::warning();

    colors.button_success = palette::success();
    colors.button_success_foreground = palette::canvas();
    colors.button_success_hover = palette::success();
    colors.button_success_active = palette::success();

    colors.button_info = palette::accent();
    colors.button_info_foreground = palette::canvas();
    colors.button_info_hover = palette::accent_hover();
    colors.button_info_active = palette::accent();

    colors.popover = palette::panel();
    colors.popover_foreground = palette::text();

    colors.sidebar = palette::panel();
    colors.sidebar_foreground = palette::text();
    colors.sidebar_border = palette::border();
    colors.sidebar_accent = palette::active();
    colors.sidebar_accent_foreground = palette::text();
    colors.sidebar_primary = palette::accent();
    colors.sidebar_primary_foreground = palette::canvas();

    colors.list = palette::panel();
    colors.list_hover = palette::hover();
    colors.list_active = palette::active();
    colors.list_active_border = palette::accent();
    colors.list_even = palette::panel();
    colors.list_head = palette::panel();

    colors.tab = palette::panel();
    colors.tab_active = palette::element();
    colors.tab_active_foreground = palette::text();
    colors.tab_bar = palette::panel();
    colors.tab_bar_segmented = palette::element();
    colors.tab_foreground = palette::text_muted();

    colors.description_list_label = palette::panel();
    colors.description_list_label_foreground = palette::text_faint();

    // Table surfaces. Wiki markdown tables render body rows on the panel
    // surface with a slightly darker head; hover and active states borrow
    // the sidebar/list vocabulary so a table row in a list feels like the
    // list rows around it.
    colors.table = palette::panel();
    colors.table_head = palette::element();
    colors.table_head_foreground = palette::text_faint();
    colors.table_foot = palette::panel();
    colors.table_foot_foreground = palette::text_faint();
    colors.table_even = palette::panel();
    colors.table_hover = palette::hover();
    colors.table_active = palette::active();
    colors.table_active_border = palette::accent();
    colors.table_row_border = palette::border();

    colors.link = palette::accent();
    colors.link_active = palette::accent();
    colors.link_hover = palette::accent_hover();

    colors.title_bar = palette::panel();
    colors.title_bar_border = palette::border();
    colors.status_bar = palette::panel();
    colors.status_bar_border = palette::border();

    colors.danger = palette::danger();
    colors.danger_foreground = palette::canvas();
    colors.danger_active = palette::danger();
    colors.danger_hover = palette::danger();
    colors.warning = palette::warning();
    colors.warning_foreground = palette::canvas();
    colors.warning_hover = palette::warning();
    colors.warning_active = palette::warning();
    colors.success = palette::success();
    colors.success_foreground = palette::canvas();
    colors.success_hover = palette::success();
    colors.success_active = palette::success();
    colors.info = palette::accent();
    colors.info_foreground = palette::canvas();
    colors.info_hover = palette::accent_hover();
    colors.info_active = palette::accent();

    colors.ring = palette::accent();
    colors.caret = palette::accent();
    colors.selection = palette::selection();

    colors.scrollbar = gpui::transparent_black();
    colors.scrollbar_thumb = palette::scrollbar_thumb();
    colors.scrollbar_thumb_hover = palette::scrollbar_thumb_hover();

    colors.overlay = palette::overlay_strong();
    colors.drop_target = palette::active();

    // Remaining component surfaces — kept in-palette so any styled
    // component that lands in a future view does not fall back to a light
    // shadcn default and stick out.
    colors.accordion = palette::panel();
    colors.group_box = palette::panel();
    colors.group_box_foreground = palette::text();
    colors.progress_bar = palette::accent();
    colors.skeleton = palette::element();
    colors.slider_bar = palette::element();
    colors.slider_thumb = palette::accent();
    colors.switch = palette::element();
    colors.switch_thumb = palette::text();
    colors.tiles = palette::panel();
    colors.window_border = palette::border();

    // Diagnostic base colors — used by `StatusColors` fallbacks (error,
    // warning, info, success, hint) when the highlight theme leaves a
    // status color unset.
    colors.red = palette::danger();
    colors.red_light = palette::danger();
    colors.green = palette::success();
    colors.green_light = palette::success();
    colors.blue = palette::accent();
    colors.blue_light = palette::accent_hover();
    colors.yellow = palette::warning();
    colors.yellow_light = palette::warning();
    colors.magenta = palette::accent();
    colors.magenta_light = palette::accent_hover();
    colors.cyan = palette::syntax_type();
    colors.cyan_light = palette::syntax_type();

    // Charts — zeta does not render any today, but a warm palette keeps
    // future analytics surfaces on-theme rather than the shadcn blues.
    colors.chart_1 = palette::accent();
    colors.chart_2 = palette::success();
    colors.chart_3 = palette::warning();
    colors.chart_4 = palette::syntax_number();
    colors.chart_5 = palette::syntax_type();
    colors.chart_bullish = palette::success();
    colors.chart_bearish = palette::danger();

    theme.tokens = (&theme.colors).into();

    Theme::sync_base(cx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use gpui_kit::component::ActiveTheme;

    /// A distinct sentinel color the light defaults never use, so an
    /// assertion that fails to overwrite a field is caught even if the
    /// default happens to share hue with the opencode palette.
    fn poison() -> Hsla {
        hex(0xff00ff)
    }

    #[gpui::test]
    fn palette_lands_on_the_expected_opencode_tokens(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(|cx| {
            let theme = Theme::global_mut(cx);
            let colors = &mut theme.colors;
            colors.background = poison();
            colors.foreground = poison();
            colors.muted_foreground = poison();
            colors.border = poison();
            colors.accent = poison();
            colors.accent_foreground = poison();
            colors.primary = poison();
            colors.ring = poison();
            colors.danger = poison();
            colors.success = poison();
            colors.warning = poison();
            colors.sidebar = poison();
            colors.popover = poison();
            colors.overlay = poison();
            colors.selection = poison();
            colors.scrollbar_thumb = poison();
            colors.scrollbar_thumb_hover = poison();
        });
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.background, palette::canvas());
            assert_eq!(theme.foreground, palette::text());
            assert_eq!(theme.muted_foreground, palette::text_muted());
            assert_eq!(theme.border, palette::border());
            assert_eq!(theme.accent, palette::accent());
            // The accent-on-canvas mapping — regressing this back to
            // near-white text on violet drops menu contrast to ~2:1.
            assert_eq!(theme.accent_foreground, palette::canvas());
            assert_eq!(theme.primary, palette::accent());
            assert_eq!(theme.ring, palette::accent());
            assert_eq!(theme.danger, palette::danger());
            assert_eq!(theme.success, palette::success());
            assert_eq!(theme.warning, palette::warning());
            assert_eq!(theme.sidebar, palette::panel());
            assert_eq!(theme.popover, palette::panel());
            assert_eq!(theme.overlay, palette::overlay_strong());
            assert_eq!(theme.selection, palette::selection());
            // Solid, not alpha — a translucent thumb shifts hue with the
            // surface underneath and stops reading as a scrollbar.
            assert_eq!(theme.scrollbar_thumb, palette::scrollbar_thumb());
            assert_eq!(
                theme.scrollbar_thumb_hover,
                palette::scrollbar_thumb_hover()
            );
        });
    }

    #[gpui::test]
    fn every_rendered_component_field_is_repainted(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(|cx| {
            let colors = &mut Theme::global_mut(cx).colors;
            // Buttons — every state on every variant. Any field left at
            // the light default paints a stray Kit button in the run UI.
            colors.button = poison();
            colors.button_active = poison();
            colors.button_foreground = poison();
            colors.button_hover = poison();
            colors.button_primary = poison();
            colors.button_primary_active = poison();
            colors.button_primary_foreground = poison();
            colors.button_primary_hover = poison();
            colors.button_secondary = poison();
            colors.button_secondary_active = poison();
            colors.button_secondary_foreground = poison();
            colors.button_secondary_hover = poison();
            colors.button_danger = poison();
            colors.button_danger_active = poison();
            colors.button_danger_foreground = poison();
            colors.button_danger_hover = poison();
            colors.button_warning = poison();
            colors.button_warning_active = poison();
            colors.button_warning_foreground = poison();
            colors.button_warning_hover = poison();
            colors.button_success = poison();
            colors.button_success_active = poison();
            colors.button_success_foreground = poison();
            colors.button_success_hover = poison();
            colors.button_info = poison();
            colors.button_info_active = poison();
            colors.button_info_foreground = poison();
            colors.button_info_hover = poison();
            // Tables — markdown code fences and settings surfaces both
            // render tables; a light head row leaks straight through.
            colors.table = poison();
            colors.table_head = poison();
            colors.table_head_foreground = poison();
            colors.table_foot = poison();
            colors.table_foot_foreground = poison();
            colors.table_even = poison();
            colors.table_hover = poison();
            colors.table_active = poison();
            colors.table_active_border = poison();
            colors.table_row_border = poison();
        });
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            for value in [
                theme.button,
                theme.button_active,
                theme.button_foreground,
                theme.button_hover,
                theme.button_primary,
                theme.button_primary_active,
                theme.button_primary_foreground,
                theme.button_primary_hover,
                theme.button_secondary,
                theme.button_secondary_active,
                theme.button_secondary_foreground,
                theme.button_secondary_hover,
                theme.button_danger,
                theme.button_danger_active,
                theme.button_danger_foreground,
                theme.button_danger_hover,
                theme.button_warning,
                theme.button_warning_active,
                theme.button_warning_foreground,
                theme.button_warning_hover,
                theme.button_success,
                theme.button_success_active,
                theme.button_success_foreground,
                theme.button_success_hover,
                theme.button_info,
                theme.button_info_active,
                theme.button_info_foreground,
                theme.button_info_hover,
                theme.table,
                theme.table_head,
                theme.table_head_foreground,
                theme.table_foot,
                theme.table_foot_foreground,
                theme.table_even,
                theme.table_hover,
                theme.table_active,
                theme.table_active_border,
                theme.table_row_border,
            ] {
                assert_ne!(value, poison(), "field left at poison after apply");
            }

            // A handful of specific pins for the fields the review flagged
            // by name, so a future regression names them instead of just
            // "some field is still poison."
            assert_eq!(theme.button, palette::element());
            assert_eq!(theme.button_foreground, palette::text());
            assert_eq!(theme.button_primary, palette::accent());
            assert_eq!(theme.button_primary_foreground, palette::canvas());
            assert_eq!(theme.table, palette::panel());
            assert_eq!(theme.table_head, palette::element());
            assert_eq!(theme.table_head_foreground, palette::text_faint());
        });
    }

    #[gpui::test]
    fn highlight_theme_paints_code_fences_in_palette(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let highlight = &cx.theme().highlight_theme;
            assert_eq!(highlight.appearance, ThemeMode::Dark);

            // Keyword must land on the violet accent, not the light
            // default's #0433ff blue — that was the visible regression.
            let keyword = highlight
                .style
                .syntax
                .style("keyword")
                .expect("keyword style is set");
            assert_eq!(keyword.color, Some(palette::accent()));

            let string = highlight
                .style
                .syntax
                .style("string")
                .expect("string style is set");
            assert_eq!(string.color, Some(palette::success()));

            let comment = highlight
                .style
                .syntax
                .style("comment")
                .expect("comment style is set");
            assert_eq!(comment.color, Some(palette::text_faint()));

            let editor_bg = highlight
                .style
                .editor_background
                .expect("editor background is set");
            assert_eq!(editor_bg, palette::canvas());
        });
    }

    #[gpui::test]
    fn shape_and_typography_pin_flat_terminal_defaults(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.radius, RADIUS);
            assert_eq!(theme.radius_lg, RADIUS);
            assert_eq!(theme.font_size, FONT_SIZE);
            assert_eq!(theme.mono_font_size, FONT_SIZE);
            assert_eq!(theme.font_family.as_ref(), "JetBrains Mono");
            assert_eq!(theme.mono_font_family.as_ref(), "JetBrains Mono");
            assert!(!theme.shadow);
            assert!(!theme.tile_shadow);
            assert!(theme.focus_ring);
            assert!(theme.is_dark());
        });
    }

    #[gpui::test]
    fn semantic_tokens_project_the_flat_radius_and_no_elevation(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let semantic = cx.theme().semantic_tokens();
            assert_eq!(semantic.radius.md, RADIUS);
            assert!(semantic.shadow.md.is_empty());
            assert!(semantic.shadow.lg.is_empty());
            // The semantic accent-on-accent pair also has to land on the
            // canvas — anything else would surface as low-contrast text
            // on any Base-consumer that reads accent_foreground.
            assert_eq!(semantic.colors.accent_foreground, palette::canvas());
        });
    }
}
