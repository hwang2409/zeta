//! Zeta run-UI theme foundation.
//!
//! Maps the wiki app's shipped `opencode` palette onto the gpui-kit theme
//! surface so every zeta view inherits the same warm-dark, terminal-first
//! look as the wiki agent-run view. Colors, radii, typography, and focus
//! ring live here so per-surface code stays free of hardcoded hex.

use gpui::{px, App, Hsla, Pixels};
use gpui_kit::component::{Theme, ThemeMode};

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
    pub fn success() -> Hsla {
        hex(0xa9c957)
    }
    pub fn warning() -> Hsla {
        hex(0xd5d878)
    }
    pub fn danger() -> Hsla {
        hex(0xe2685c)
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
    pub fn scrollbar_thumb() -> Hsla {
        hex_a(0xece9_d833)
    }
    pub fn scrollbar_thumb_hover() -> Hsla {
        hex_a(0xece9_d866)
    }
}

/// Apply the opencode palette + shape + typography rhythm onto the global
/// theme, then push the update down to the Base layer so scrollbars and
/// resize handles paint with the same tokens.
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

    let colors = &mut theme.colors;
    colors.background = palette::canvas();
    colors.foreground = palette::text();
    colors.border = palette::border();
    colors.drag_border = palette::border_active();
    colors.input = palette::border();

    colors.muted = palette::element();
    colors.muted_foreground = palette::text_muted();

    colors.accent = palette::accent();
    colors.accent_foreground = palette::text();
    colors.primary = palette::accent();
    colors.primary_foreground = palette::canvas();
    colors.primary_active = palette::accent();
    colors.primary_hover = palette::accent();
    colors.secondary = palette::element();
    colors.secondary_foreground = palette::text();
    colors.secondary_active = palette::panel();
    colors.secondary_hover = palette::hover();

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
    colors.table_head_foreground = palette::text_faint();
    colors.table_foot_foreground = palette::text_faint();
    colors.link = palette::accent();
    colors.link_active = palette::accent();
    colors.link_hover = palette::accent();

    colors.title_bar = palette::panel();
    colors.title_bar_border = palette::border();
    colors.status_bar = palette::panel();
    colors.status_bar_border = palette::border();

    colors.danger = palette::danger();
    colors.danger_foreground = palette::text();
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
    colors.info_hover = palette::accent();
    colors.info_active = palette::accent();

    colors.ring = palette::accent();
    colors.caret = palette::accent();
    colors.selection = palette::selection();

    colors.scrollbar = gpui::transparent_black();
    colors.scrollbar_thumb = palette::scrollbar_thumb();
    colors.scrollbar_thumb_hover = palette::scrollbar_thumb_hover();

    colors.overlay = palette::overlay_strong();
    colors.drop_target = palette::active();

    theme.tokens = (&theme.colors).into();

    Theme::sync_base(cx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use gpui_kit::component::ActiveTheme;

    #[gpui::test]
    fn palette_lands_on_the_expected_opencode_tokens(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.background, palette::canvas());
            assert_eq!(theme.foreground, palette::text());
            assert_eq!(theme.muted_foreground, palette::text_muted());
            assert_eq!(theme.border, palette::border());
            assert_eq!(theme.accent, palette::accent());
            assert_eq!(theme.primary, palette::accent());
            assert_eq!(theme.ring, palette::accent());
            assert_eq!(theme.danger, palette::danger());
            assert_eq!(theme.success, palette::success());
            assert_eq!(theme.warning, palette::warning());
            assert_eq!(theme.sidebar, palette::panel());
            assert_eq!(theme.popover, palette::panel());
            assert_eq!(theme.overlay, palette::overlay_strong());
            assert_eq!(theme.selection, palette::selection());
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
        });
    }
}
