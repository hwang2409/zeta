//! Zeta run-UI theme foundation.
//!
//! Maps the wiki app's shipped `opencode` palette onto the gpui-kit theme
//! surface so every zeta view inherits the same warm-dark, terminal-first
//! look as the wiki agent-run view. Colors, radii, typography, and focus
//! ring live here so per-surface code stays free of hardcoded hex.

use std::sync::Arc;

use gpui::{px, App, Hsla, Pixels};
use gpui_kit::component::{highlighter::HighlightTheme, ActiveTheme, Theme, ThemeMode};

/// Base UI type size — the "one size drives everything" pin from the wiki
/// run-UI extraction.
pub const FONT_SIZE: Pixels = px(15.);

/// Universal radius for zeta surfaces. Two pixels reads as "flat rectangles"
/// while still softening the corner just enough to avoid the raw-terminal
/// look.
pub const RADIUS: Pixels = px(2.);

/// Readable-column ceiling for the transcript. Wiki caps its agent-run column
/// at 1024px so long assistant lines break at a scannable measure.
pub const TRANSCRIPT_MAX_WIDTH: Pixels = px(1024.);

/// Vertical rhythm between transcript rows. Consecutive tool rows collapse
/// this gap to zero so a run of receipts reads as one column.
pub const TRANSCRIPT_ROW_GAP: Pixels = px(14.);

/// Baseline padding for the composer strip (padding 8 x 10 from the contract).
pub const COMPOSER_PADDING_Y: Pixels = px(8.);
pub const COMPOSER_PADDING_X: Pixels = px(10.);

/// Composer minimum height (contract: min-height 64px).
pub const COMPOSER_MIN_HEIGHT: Pixels = px(64.);

/// Send button minimum width. Kept square, mono 600, and just wide enough for
/// the word "Send" plus breathing room per the contract.
pub const SEND_BUTTON_MIN_WIDTH: Pixels = px(82.);
pub const SEND_BUTTON_HEIGHT: Pixels = px(40.);

/// Width of the coloured left rail used on user turns and the composer.
pub const RAIL_WIDTH_THICK: Pixels = px(3.);

/// Width of the thin rail used on expanded tool bodies.
pub const RAIL_WIDTH_THIN: Pixels = px(1.);

/// Small streaming indicator dot size — wiki uses 7px.
pub const STREAM_DOT_SIZE: Pixels = px(7.);

/// Composer target-line row height — the muted "→ model" label above the
/// textarea. Kept tight so the 64px composer floor stays honest.
pub const COMPOSER_TARGET_HEIGHT: Pixels = px(16.);

/// Sidebar container width — the wiki agent-run column pins this at 216px so
/// the panel reads as a fixed column rather than a fluid drawer.
pub const SIDEBAR_WIDTH: Pixels = px(216.);

/// Session row min-height and padding. Contract line 81 pins 40px rows with
/// 5px vertical / 8px horizontal padding — a compact-but-breathing rhythm
/// that carries a ticket label + age on one line.
pub const SIDEBAR_ROW_HEIGHT: Pixels = px(40.);
pub const SIDEBAR_ROW_PADDING_X: Pixels = px(8.);
pub const SIDEBAR_ROW_PADDING_Y: Pixels = px(5.);

/// Nested branch row height — one tint step shorter than session rows so a
/// run of siblings under a session reads as sub-items.
pub const SIDEBAR_NESTED_ROW_HEIGHT: Pixels = px(32.);

/// Left-gutter width for the accent dot the current session paints. Sized so
/// the dot sits centred in a mono ticket's leading margin without pushing
/// the label rightward.
pub const SIDEBAR_GUTTER_WIDTH: Pixels = px(10.);

/// Accent dot for the current sidebar row. Reads as a mono bullet at 15px
/// text without borrowing hover fill.
pub const SIDEBAR_CURRENT_DOT_SIZE: Pixels = px(5.);

/// Attention rail that pins the leftmost 2px of a sidebar row when the row is
/// signalling a failure or the connection is lost.
pub const ATTENTION_RAIL_WIDTH: Pixels = px(2.);

/// Two-band header/status strip heights. Band 1 (title + state pill) sits at
/// 44px; band 2 (metadata) sits at 40px so the strip is a compact 84px
/// column rather than a fluid banner.
pub const HEADER_BAND1_MIN_HEIGHT: Pixels = px(44.);
pub const HEADER_BAND2_MIN_HEIGHT: Pixels = px(40.);

/// State pill padding — near-square shape the wiki agent-run uses to carry a
/// one-word state label ("ready", "streaming", "offline").
pub const STATE_PILL_PADDING_X: Pixels = px(9.);
pub const STATE_PILL_PADDING_Y: Pixels = px(3.);

/// Vertical separator ruled between header/status metadata items. The rule
/// is a 1x14px line, drawn as a thin div with a border color.
pub const STATUS_RULE_HEIGHT: Pixels = px(14.);

/// Flat-panel modal shape. Width caps at 480px, padding is 12px on top / 16px
/// horizontally / 14px on bottom, and the panel sits below a scrim at 25% of
/// the viewport height.
pub const MODAL_WIDTH: Pixels = px(480.);
pub const MODAL_PADDING_TOP: Pixels = px(12.);
pub const MODAL_PADDING_X: Pixels = px(16.);
pub const MODAL_PADDING_BOTTOM: Pixels = px(14.);
pub const MODAL_BUTTON_HEIGHT: Pixels = px(30.);
pub const MODAL_TOP_FRACTION: f32 = 0.25;

/// Semantic composer color roles. The composer paints its rail, fill, and
/// target-line label from these — never from `palette::*` directly — so the
/// call sites read as "composer at rest / composer focused" rather than
/// "some palette function looks composer-shaped."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComposerRoles {
    pub rail_rest: Hsla,
    pub rail_focus: Hsla,
    pub fill_rest: Hsla,
    pub fill_focus: Hsla,
    pub target_label: Hsla,
    pub target_value: Hsla,
    pub send_disabled_outline: Hsla,
}

/// Semantic composer tokens routed through `cx.theme()`. Consumers read here
/// instead of touching `palette::*` — a future theme refactor changes tokens
/// in one place, and every composer state moves with it.
pub fn composer_roles(cx: &App) -> ComposerRoles {
    let theme = cx.theme();
    ComposerRoles {
        rail_rest: palette::accent_rail_dim(),
        rail_focus: theme.primary,
        fill_rest: theme.muted,
        fill_focus: palette::composer_focus_fill(),
        target_label: theme.muted_foreground,
        target_value: theme.primary,
        send_disabled_outline: palette::border_active(),
    }
}

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
    /// Composer rail at rest — accent at ~62% opacity. Focus promotes it back
    /// to the full accent so the rail is the composer's focus signal.
    pub fn accent_rail_dim() -> Hsla {
        hex_a(0xb18b_f49e)
    }
    /// Fill the composer takes when the input receives focus — one tint step
    /// lighter than the element surface so focus stays visible without a ring.
    pub fn composer_focus_fill() -> Hsla {
        hex(0x333326)
    }
    /// Danger 10% tint on canvas — the wiki blocker row's bg. Paired with the
    /// 2px danger left rail so a failure surface reads as an alarm strip
    /// without a full solid-red panel.
    pub fn danger_tint() -> Hsla {
        hex_a(0xe268_5c1a)
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
    // Head and foot foregrounds carry table labels and summary cells; the
    // wiki markdown table paints both with normal body text so they land on
    // WCAG AA (>= 4.5:1) against panel/element. `text_faint` here drops the
    // pair below 3.1:1 and the head becomes hard to read.
    colors.table = palette::panel();
    colors.table_head = palette::element();
    colors.table_head_foreground = palette::text();
    colors.table_foot = palette::panel();
    colors.table_foot_foreground = palette::text();
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

    /// WCAG 2.x relative luminance of an sRGB color.
    fn relative_luminance(color: Hsla) -> f32 {
        let rgba = color.to_rgb();
        let channel = |c: f32| {
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(rgba.r) + 0.7152 * channel(rgba.g) + 0.0722 * channel(rgba.b)
    }

    /// WCAG 2.x contrast ratio between two sRGB colors, in [1, 21].
    fn contrast_ratio(a: Hsla, b: Hsla) -> f32 {
        let la = relative_luminance(a);
        let lb = relative_luminance(b);
        let (lmax, lmin) = if la >= lb { (la, lb) } else { (lb, la) };
        (lmax + 0.05) / (lmin + 0.05)
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
            assert_eq!(theme.table_head_foreground, palette::text());
            assert_eq!(theme.table_foot_foreground, palette::text());
        });
    }

    #[gpui::test]
    fn table_label_pairs_clear_wcag_aa_contrast(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            // Body-text minimum, per WCAG 2.1 AA. Table head/foot cells carry
            // labels and summary rows, so they read as normal text and cannot
            // fall back onto the 3:1 large-text tier.
            let head = contrast_ratio(theme.table_head_foreground, theme.table_head);
            let foot = contrast_ratio(theme.table_foot_foreground, theme.table_foot);
            assert!(
                head >= 4.5,
                "table head foreground/background contrast {head:.2}:1 fails WCAG AA (4.5:1)"
            );
            assert!(
                foot >= 4.5,
                "table foot foreground/background contrast {foot:.2}:1 fails WCAG AA (4.5:1)"
            );
        });
    }

    #[gpui::test]
    fn highlight_theme_paints_code_fences_in_palette(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        // Poison the highlight theme BEFORE apply so a future refactor that
        // drops the `theme.highlight_theme = opencode_highlight_theme()` line
        // is caught: without the write, the sentinel would still be here
        // after apply and the assertions below would fail.
        cx.update(|cx| {
            Theme::global_mut(cx).highlight_theme = Arc::new(HighlightTheme {
                name: "Zeta Poison".to_string(),
                appearance: ThemeMode::Light,
                style: Default::default(),
            });
        });
        cx.update(apply);

        cx.update(|cx| {
            let highlight = &cx.theme().highlight_theme;
            assert_eq!(highlight.name, "Opencode Zeta");
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

    #[test]
    fn sidebar_and_chrome_tokens_land_on_the_wiki_contract() {
        // Guards the lane-3 numbers so a later ticket that widens the sidebar
        // or grows the pill padding trips a named assert rather than only the
        // paint-probes downstream.
        assert_eq!(SIDEBAR_WIDTH, px(216.));
        assert_eq!(SIDEBAR_ROW_HEIGHT, px(40.));
        assert_eq!(SIDEBAR_NESTED_ROW_HEIGHT, px(32.));
        assert_eq!(SIDEBAR_ROW_PADDING_X, px(8.));
        assert_eq!(SIDEBAR_ROW_PADDING_Y, px(5.));
        assert_eq!(SIDEBAR_GUTTER_WIDTH, px(10.));
        assert_eq!(SIDEBAR_CURRENT_DOT_SIZE, px(5.));
        assert_eq!(ATTENTION_RAIL_WIDTH, px(2.));
        assert_eq!(HEADER_BAND1_MIN_HEIGHT, px(44.));
        assert_eq!(HEADER_BAND2_MIN_HEIGHT, px(40.));
        assert_eq!(STATE_PILL_PADDING_X, px(9.));
        assert_eq!(STATE_PILL_PADDING_Y, px(3.));
        assert_eq!(STATUS_RULE_HEIGHT, px(14.));
        assert_eq!(MODAL_WIDTH, px(480.));
        assert_eq!(MODAL_PADDING_TOP, px(12.));
        assert_eq!(MODAL_PADDING_X, px(16.));
        assert_eq!(MODAL_PADDING_BOTTOM, px(14.));
        assert_eq!(MODAL_BUTTON_HEIGHT, px(30.));
        assert!((MODAL_TOP_FRACTION - 0.25).abs() < f32::EPSILON);

        // The blocker-row tint borrows the danger hue but stays transparent
        // enough to read as ambient alarm chrome, not a solid red panel.
        let tint = palette::danger_tint();
        let danger = palette::danger();
        assert_eq!(tint.h, danger.h);
        assert_eq!(tint.s, danger.s);
        assert_eq!(tint.l, danger.l);
        assert!(tint.a > 0.05 && tint.a < 0.20);
    }

    #[test]
    fn transcript_and_composer_tokens_land_on_the_wiki_contract() {
        // Guards against a silent number drift when a later ticket bumps
        // spacing or rethinks the readable-column measure.
        assert_eq!(TRANSCRIPT_MAX_WIDTH, px(1024.));
        assert_eq!(TRANSCRIPT_ROW_GAP, px(14.));
        assert_eq!(COMPOSER_MIN_HEIGHT, px(64.));
        assert_eq!(COMPOSER_PADDING_Y, px(8.));
        assert_eq!(COMPOSER_PADDING_X, px(10.));
        assert_eq!(SEND_BUTTON_MIN_WIDTH, px(82.));
        assert_eq!(RAIL_WIDTH_THICK, px(3.));
        assert_eq!(RAIL_WIDTH_THIN, px(1.));
        assert_eq!(STREAM_DOT_SIZE, px(7.));

        // Composer rail at rest sits between muted and full accent so a
        // future palette shuffle keeps the focus contrast well-defined.
        let rail = palette::accent_rail_dim();
        let accent = palette::accent();
        assert!(
            rail.a < accent.a,
            "rail must dim the accent it borrows from"
        );
        assert!(
            rail.a > 0.4,
            "rail must stay visible on the element surface"
        );
        assert_eq!(rail.h, accent.h);
        assert_eq!(rail.s, accent.s);
        assert_eq!(rail.l, accent.l);

        // Focus fill is one tint step lighter than the element surface — a
        // regression that dropped it back onto the element loses the focus
        // signal entirely (the rail alone reads as ambient chrome).
        assert!(palette::composer_focus_fill().l > palette::element().l);
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
        // Poison the Base global's tokens BEFORE apply. `Theme::sync_base`
        // is what pushes the styled tokens down to the Base layer where the
        // scrollbar and resize-handle paint from; without that call this
        // radius stays at the light default and scrollbars keep their pill.
        cx.update(|cx| {
            let base = gpui_kit::base::Theme::global_mut(cx);
            base.tokens.radius.md = px(999.);
            base.tokens.colors.accent_foreground = poison();
        });
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

            // And the Base global itself has to carry those same tokens —
            // otherwise sync_base was skipped and Base-layer consumers keep
            // painting with the shadcn defaults.
            let base = gpui_kit::base::Theme::global(cx);
            assert_eq!(base.tokens.radius.md, RADIUS);
            assert!(base.tokens.shadow.md.is_empty());
            assert!(base.tokens.shadow.lg.is_empty());
            assert_eq!(base.tokens.colors.accent_foreground, palette::canvas());
        });
    }
}
