//! Zeta run-UI theme foundation.
//!
//! Wraps a small registry of named palettes onto the gpui-kit theme surface.
//! Opencode (dark) ships as the default and every view still inherits the
//! same warm-dark terminal look, but the appearance is now user-picked at
//! runtime through `apply_with` — font size, font family, and theme id are
//! read from a client-side prefs file and re-applied live from the
//! Settings > Appearance panel. The `palette::*` accessors keep their names
//! so per-surface code stays free of hardcoded hex; they read the currently
//! active palette out of a lazily initialised global.

use std::{
    cell::RefCell,
    sync::{Arc, LazyLock},
};

use gpui::{px, App, Hsla, Pixels, SharedString, StyleRefinement, Styled as _};
use gpui_kit::component::{highlighter::HighlightTheme, ActiveTheme, Theme, ThemeMode};

/// Default UI type size — the "one size drives everything" pin from the wiki
/// run-UI extraction, adjustable at runtime through Settings > Appearance.
pub const DEFAULT_FONT_SIZE: Pixels = px(13.);

/// Minimum / maximum font size the appearance picker exposes. Whole px only.
pub const MIN_FONT_SIZE_PX: f32 = 11.0;
pub const MAX_FONT_SIZE_PX: f32 = 18.0;

/// Default font family. Every family in `FONT_FAMILIES` must resolve on the
/// user's platform; the picker filters unavailable families before display.
pub const DEFAULT_FONT_FAMILY: &str = "JetBrains Mono";

/// Curated monospace families offered by the appearance picker. Order is the
/// order shown in Settings; the first is the shipped default. The picker
/// filters this list against the platform's font system before display so a
/// family the OS cannot resolve never lands in the theme.
pub const FONT_FAMILIES: &[&str] = &["JetBrains Mono", "Fira Code", "SF Mono", "Menlo", "Monaco"];

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

/// Left-gutter width reserved for the current-item dot. Wide enough to seat
/// a 9px dot with a 4px left inset per contract line 81 (`left 4px`).
pub const SIDEBAR_GUTTER_WIDTH: Pixels = px(13.);

/// Distance from the row's left edge to the dot. Contract line 81 pins this
/// at 4px so the dot sits inside the row's leading margin, not centred.
pub const SIDEBAR_CURRENT_DOT_INSET: Pixels = px(4.);

/// Accent dot for the current sidebar row. Wiki contract calls for ~0.58em
/// at a 15px base — 9px rounded — so the dot reads as a mono bullet without
/// borrowing hover fill.
pub const SIDEBAR_CURRENT_DOT_SIZE: Pixels = px(9.);

/// Attention rail that pins the leftmost 2px of a sidebar row when the row is
/// signalling a failure or the connection is lost.
pub const ATTENTION_RAIL_WIDTH: Pixels = px(2.);

/// Width of every scrollbar thumb. Contract line 93 pins the strip to a thin
/// 8px lane so the transcript column keeps its readable measure.
pub const SCROLLBAR_THUMB_WIDTH: Pixels = px(8.);

/// Single-row run header height. Session title on the left, quiet metadata
/// cluster (tokens/cache, dot + state word, model) pinned right. Contract
/// line 83 (ZETA-123): one 44px row, not two stacked bands.
pub const HEADER_BAND1_MIN_HEIGHT: Pixels = px(44.);

/// Vertical separator ruled between header/status metadata items. The rule
/// is a 1x14px line, drawn as a thin div with a border color.
pub const STATUS_RULE_HEIGHT: Pixels = px(14.);

/// Ceiling for the tokens/cache slot in the run header. Truncates a very
/// long metrics string before it steals room from the title. Named so a
/// palette / density refactor can widen or shrink it in one place.
pub const HEADER_STATUS_METRICS_MAX_WIDTH: Pixels = px(240.);

/// Floor for the run-header title. Below this the identity of the run
/// becomes unscannable and the cluster wins the row. Wide enough to seat
/// roughly six mono characters of a session name; narrow enough that
/// the metadata cluster still gets meaningful room at 600–760px.
pub const HEADER_TITLE_MIN_WIDTH: Pixels = px(80.);

/// Ceiling for the model-name slot in the run header. Long model ids
/// (`claude-opus-4-7-us-east`) truncate before the chip pushes the
/// cluster past the header's right edge.
pub const HEADER_MODEL_MAX_WIDTH: Pixels = px(200.);

/// Gap between the status dot and its state word in the header's mode
/// cluster. Tight enough that the dot reads as a leading glyph on the
/// word rather than two independent chips.
pub const HEADER_MODE_GAP: Pixels = px(6.);

/// Ceiling for the composer target line ("→ model") above the input.
/// Long model ids truncate before the target row overruns the composer
/// column.
pub const COMPOSER_TARGET_MAX_WIDTH: Pixels = px(260.);

/// Flat-panel modal shape. Width caps at 480px, padding is 12px on top / 16px
/// horizontally / 14px on bottom, and the panel sits below a scrim at 25% of
/// the viewport height.
pub const MODAL_WIDTH: Pixels = px(480.);
pub const MODAL_PADDING_TOP: Pixels = px(12.);
pub const MODAL_PADDING_X: Pixels = px(16.);
pub const MODAL_PADDING_BOTTOM: Pixels = px(14.);
pub const MODAL_BUTTON_HEIGHT: Pixels = px(30.);
pub const MODAL_TOP_FRACTION: f32 = 0.25;

/// Clamp a candidate font size to the appearance picker's whole-px window.
pub fn clamp_font_size(px_value: f32) -> Pixels {
    let clamped = px_value.round().clamp(MIN_FONT_SIZE_PX, MAX_FONT_SIZE_PX);
    px(clamped)
}

/// Floor for size roles derived from the base font size. Kept above browsers'
/// unreadable-tier so shrinking the base to `MIN_FONT_SIZE_PX` still leaves
/// chip / hint / preview labels legible.
pub const MIN_LABEL_PX: f32 = 9.0;

/// Title-tier size derived from the current base font size. Reserved for the
/// header session label and modal titles — the ONE strongly-promoted role on
/// screen so a run-header title reads as the top of the hierarchy without
/// borrowing an oversized weight. Sits one step above body at every base.
///
/// At the shipped default (13px) this lands at 15px (~1.15x body), matching
/// the ratio the type-scale contract asks for. The `+2` grows linearly with
/// the picker so 11px→13px and 18px→20px keep the same visual step.
pub fn title(base: Pixels) -> Pixels {
    px(f32::from(base) + 2.)
}

/// Body-tier size. The transcript prose and every unadorned block of user
/// text ride here — the pin the appearance picker moves. Kept as an alias
/// for the base so a text site that means "normal reading text" reads that
/// way at the call site rather than passing `cx.theme().font_size` bare.
pub fn body(base: Pixels) -> Pixels {
    base
}

/// Label-tier size derived from the current base font size. One step below
/// body — sidebar rows, branch rows, and any secondary label that must sit
/// tighter than prose without falling into hint territory. Floored at
/// `MIN_LABEL_PX` so the picker's `MIN_FONT_SIZE_PX` still lands legibly.
pub fn label(base: Pixels) -> Pixels {
    px((f32::from(base) - 1.).max(MIN_LABEL_PX))
}

/// Small-tier label size derived from the current base font size. Attachment
/// chips, composer target lines, tool hints, login status, status pills and
/// runtime metadata paint with this — two steps below body at every base,
/// floored at `MIN_LABEL_PX`.
pub fn label_small(base: Pixels) -> Pixels {
    px((f32::from(base) - 2.).max(MIN_LABEL_PX))
}

/// Micro-tier label size derived from the current base font size. Reserved
/// for the smallest secondary text (thumbnail fallback captions). Sits one
/// step below `label_small`, floored at `MIN_LABEL_PX`.
pub fn label_micro(base: Pixels) -> Pixels {
    px((f32::from(base) - 3.).max(MIN_LABEL_PX))
}

/// Reading-measure target for transcript prose, in characters of the base
/// mono font. Sits inside the "comfortable measure" window (~66-90ch for
/// readability). Chosen at 88 (not 90) so `prose_max_width` — which now
/// includes the row's 32px horizontal padding — still lands strictly
/// under `TRANSCRIPT_MAX_WIDTH` at the picker's MAX 18px base
/// (18 * 0.62 * 88 + 32 ≈ 1014 < 1024). Gives ~88ch of shaped mono text
/// inside the padding at every picker step.
pub const PROSE_MEASURE_CH: f32 = 88.0;

/// Monospace glyph advance as a fraction of the font size. JetBrains Mono
/// (and every family the appearance picker filters to) advances ~0.6em per
/// glyph; the extra 0.02 gives a small margin so wrapping never pushes the
/// last glyph past the column edge on subpixel rounding.
pub const MONO_CH_ADVANCE: f32 = 0.62;

/// Horizontal padding on each side of a transcript prose row (from the row
/// wrapper's `.px_4()`). The prose cap includes this so the effective TEXT
/// measure inside the padding is `PROSE_MEASURE_CH` chars, not that minus
/// the ~4 chars 32px would otherwise steal at the shipped base.
pub const PROSE_ROW_PADDING_X: f32 = 16.0;

/// Reading-measure cap for transcript PROSE rows (user, assistant,
/// thinking) — the row-kind narrower column that keeps assistant lines
/// scannable. Tool receipts and framed error blocks keep
/// `TRANSCRIPT_MAX_WIDTH` so a wide command line or code block does not
/// re-wrap at the prose measure. Scales with the appearance picker's base
/// so an 18px reader keeps their character measure.
///
/// The cap is `PROSE_MEASURE_CH` characters of shaped mono text PLUS the
/// row's horizontal padding on each side, so a caller that pipes this
/// through `.max_w(...).px_4()` lands the TEXT area at exactly
/// `PROSE_MEASURE_CH` glyph advances — the value the picker's base font
/// promises. Without the padding term the effective measure at 13px base
/// would be ~86ch (32 / (0.62 * 13) ≈ 4ch shorter than advertised).
pub fn prose_max_width(base: Pixels) -> Pixels {
    px(f32::from(base) * MONO_CH_ADVANCE * PROSE_MEASURE_CH + 2.0 * PROSE_ROW_PADDING_X)
}

/// Effective text measure INSIDE the prose row's horizontal padding —
/// `prose_max_width(base)` minus 2× `PROSE_ROW_PADDING_X`. Tests and the
/// render-time text-run recorder both route through this so a padding
/// change lands in ONE place. Gated on `test` + `smoke-test` because
/// both callers are cfg-gated; a release build never needs the measure
/// separately from `prose_max_width`.
#[cfg(any(test, feature = "smoke-test"))]
pub fn prose_text_measure(base: Pixels) -> Pixels {
    px(f32::from(base) * MONO_CH_ADVANCE * PROSE_MEASURE_CH)
}

/// Pending-attachment chip thumbnail size. Base 13px keeps parity with the
/// original 32x24 rectangle; other sizes scale proportionally so the chip
/// row rhythm matches the appearance picker's base font.
pub fn chip_thumbnail_size(base: Pixels) -> (Pixels, Pixels) {
    let scale = f32::from(base) / 13.0;
    let width = (32. * scale).round().max(24.);
    let height = (24. * scale).round().max(18.);
    (px(width), px(height))
}

/// Pending-attachment chip remove-button hit target. Same 24px baseline
/// as the thumbnail height at 13px; scales with the base font.
pub fn chip_control_size(base: Pixels) -> Pixels {
    let scale = f32::from(base) / 13.0;
    px((24. * scale).round().max(20.))
}

/// Vertical padding around the chip row's content. Small enough to keep the
/// chip visually flat; scales gently with the appearance base.
pub fn chip_padding_y(base: Pixels) -> Pixels {
    let scale = f32::from(base) / 13.0;
    px((3. * scale).round().max(2.))
}

/// Chip label-column maximum width. Truncates a long filename before the
/// chip stretches past a scannable measure; scales with the base font so
/// bigger picks keep roughly the same character budget on-screen.
pub fn chip_label_max_width(base: Pixels) -> Pixels {
    let scale = f32::from(base) / 13.0;
    px((180. * scale).round().max(140.))
}

/// Named palettes the appearance picker exposes. Opencode ships as the
/// default and mirrors the wiki agent-run look; the other four give the
/// user a spread of dark and light options without leaving the flat,
/// terminal-first shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThemeId {
    #[default]
    Opencode,
    GruvboxDark,
    VscodeDarkPlus,
    Nord,
    GruvboxLight,
}

impl ThemeId {
    /// Every theme the picker offers, in display order.
    pub const ALL: &'static [ThemeId] = &[
        Self::Opencode,
        Self::GruvboxDark,
        Self::VscodeDarkPlus,
        Self::Nord,
        Self::GruvboxLight,
    ];

    /// Slug used in prefs.json and status-selector strings. Never changes.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Opencode => "opencode",
            Self::GruvboxDark => "gruvbox-dark",
            Self::VscodeDarkPlus => "vscode-dark-plus",
            Self::Nord => "nord",
            Self::GruvboxLight => "gruvbox-light",
        }
    }

    /// Human-readable label the picker renders in Settings.
    pub fn label(self) -> &'static str {
        match self {
            Self::Opencode => "Opencode",
            Self::GruvboxDark => "Gruvbox Dark",
            Self::VscodeDarkPlus => "VSCode Dark+",
            Self::Nord => "Nord",
            Self::GruvboxLight => "Gruvbox Light",
        }
    }

    /// Parse a persisted slug back to a `ThemeId`. Returns `None` on any
    /// unknown slug so callers can fall back to `ThemeId::default()`.
    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|id| id.slug() == slug)
    }

    pub fn palette(self) -> &'static Palette {
        match self {
            Self::Opencode => &PALETTE_OPENCODE,
            Self::GruvboxDark => &PALETTE_GRUVBOX_DARK,
            Self::VscodeDarkPlus => &PALETTE_VSCODE_DARK_PLUS,
            Self::Nord => &PALETTE_NORD,
            Self::GruvboxLight => &PALETTE_GRUVBOX_LIGHT,
        }
    }
}

/// User appearance preferences — theme id, font family, base font size.
/// The apply layer consumes this whole struct so a single call reconfigures
/// every surface in one pass, matching the "font size / family / theme all
/// live-update from Settings" contract.
#[derive(Debug, Clone)]
pub struct Appearance {
    pub theme: ThemeId,
    pub font_family: SharedString,
    pub font_size: Pixels,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: ThemeId::default(),
            font_family: SharedString::new_static(DEFAULT_FONT_FAMILY),
            font_size: DEFAULT_FONT_SIZE,
        }
    }
}

/// Base color values for one theme. Alpha-blended tokens (hover, scrollbar,
/// composer rail dim, danger tint) derive from these at read time so a
/// palette author only fills in the base — the derivation stays consistent.
#[derive(Debug, Clone)]
pub struct Palette {
    pub id: ThemeId,
    pub mode: ThemeMode,
    pub canvas: Hsla,
    pub panel: Hsla,
    pub element: Hsla,
    pub border: Hsla,
    pub border_subtle: Hsla,
    pub border_active: Hsla,
    pub text: Hsla,
    pub text_muted: Hsla,
    pub text_faint: Hsla,
    pub accent: Hsla,
    pub accent_hover: Hsla,
    pub success: Hsla,
    pub warning: Hsla,
    pub danger: Hsla,
    pub syntax_number: Hsla,
    pub syntax_type: Hsla,
    pub composer_focus_fill: Hsla,
    /// Syntax color hex literals used to build the highlight theme JSON.
    pub syntax: SyntaxHex,
    /// Text painted ON each solid semantic fill (state pill, primary/danger
    /// button, inverted badge). One foreground per semantic because a single
    /// fg cannot clear WCAG AA against four different fills — a light warning
    /// wants a dark label, a medium red danger may want a near-black label
    /// on some themes, and the same white that works over a dark accent
    /// fails over a light beige warning. Every pair is verified by
    /// `every_solid_semantic_pair_clears_wcag_aa`.
    pub accent_fg: Hsla,
    pub success_fg: Hsla,
    pub warning_fg: Hsla,
    pub danger_fg: Hsla,
}

/// Syntax palette in raw hex-string form. The highlight-theme JSON is built
/// per palette by templating these into the same style map opencode used.
#[derive(Debug, Clone, Copy)]
pub struct SyntaxHex {
    pub background: &'static str,
    pub foreground: &'static str,
    pub gutter_background: &'static str,
    pub active_line_background: &'static str,
    pub line_number: &'static str,
    pub active_line_number: &'static str,
    pub invisible: &'static str,
    pub attribute: &'static str,
    pub boolean: &'static str,
    pub comment: &'static str,
    pub constant: &'static str,
    pub constructor: &'static str,
    pub embedded: &'static str,
    pub emphasis: &'static str,
    pub enum_: &'static str,
    pub function: &'static str,
    pub hint: &'static str,
    pub keyword: &'static str,
    pub label: &'static str,
    pub link_text: &'static str,
    pub link_uri: &'static str,
    pub number: &'static str,
    pub operator: &'static str,
    pub preproc: &'static str,
    pub property: &'static str,
    pub punctuation: &'static str,
    pub string: &'static str,
    pub string_escape: &'static str,
    pub tag: &'static str,
    pub tag_doctype: &'static str,
    pub text_code_span: &'static str,
    pub text_literal: &'static str,
    pub title: &'static str,
    pub type_: &'static str,
    pub variable: &'static str,
    pub variable_special: &'static str,
    pub variant: &'static str,
}

impl Palette {
    /// Row-hover fill — text color at ~6% opacity so hover feels like a
    /// warm-off-white wash rather than a solid stripe.
    pub fn hover(&self) -> Hsla {
        with_alpha(self.text, 0.06)
    }
    /// Row-pressed fill — accent at ~14% opacity, kept consistent across
    /// palettes so the "just clicked" state reads on every canvas.
    pub fn active(&self) -> Hsla {
        with_alpha(self.accent, 0.14)
    }
    /// Text selection wash — text color at 16% opacity.
    pub fn selection(&self) -> Hsla {
        with_alpha(self.text, 0.16)
    }
    /// Modal backdrop — black at 45% opacity across every theme so the scrim
    /// reads the same whether the canvas is dark or light.
    pub fn overlay_strong(&self) -> Hsla {
        gpui::hsla(0.0, 0.0, 0.0, 0.45)
    }
    /// Scrollbar thumb at rest — text color at 20% opacity.
    pub fn scrollbar_thumb(&self) -> Hsla {
        with_alpha(self.text, 0.20)
    }
    /// Scrollbar thumb on hover — text color at 40% opacity.
    pub fn scrollbar_thumb_hover(&self) -> Hsla {
        with_alpha(self.text, 0.40)
    }
    /// Composer rail at rest — accent at ~62% opacity so focus can promote
    /// it to the full accent hue.
    pub fn accent_rail_dim(&self) -> Hsla {
        with_alpha(self.accent, 0.62)
    }
    /// Blocker-row bg — danger color at ~10% opacity.
    pub fn danger_tint(&self) -> Hsla {
        with_alpha(self.danger, 0.10)
    }
}

/// Copy `base` with a new alpha channel. Keeps h/s/l untouched so an
/// alpha-blended token still borrows its hue from the base palette.
fn with_alpha(base: Hsla, alpha: f32) -> Hsla {
    gpui::hsla(base.h, base.s, base.l, alpha)
}

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

// The currently applied appearance. Held as thread-local state so parallel
// gpui-test workers each observe their own value — production has one main
// thread + one App, so the practical semantics stay identical there while
// the test suite gets deterministic per-worker isolation. `apply_with`
// updates this snapshot BEFORE mutating the gpui-kit theme so a palette
// accessor invoked during `sync_base` sees the new palette.
thread_local! {
    static ACTIVE: RefCell<Appearance> = RefCell::new(Appearance::default());
}

fn active_palette() -> &'static Palette {
    ACTIVE.with(|slot| slot.borrow().theme.palette())
}

fn set_active(appearance: Appearance) {
    ACTIVE.with(|slot| *slot.borrow_mut() = appearance);
}

/// Read the currently active appearance for THIS thread. UI paths that need
/// to reflow when the user picks a different font size / family read here.
#[allow(dead_code)]
pub fn current_appearance() -> Appearance {
    ACTIVE.with(|slot| slot.borrow().clone())
}

/// Convenience: the currently active font size in pixels.
pub fn current_font_size() -> Pixels {
    ACTIVE.with(|slot| slot.borrow().font_size)
}

/// Convenience: the currently active font family.
pub fn current_font_family() -> SharedString {
    ACTIVE.with(|slot| slot.borrow().font_family.clone())
}

/// Named accessors onto the currently active palette. Every function
/// reads through `active_palette()` so a theme switch propagates without
/// touching any call site. The full set is exposed publicly (some are
/// unused from the binary today but stay part of the palette surface for
/// future renderers and tests).
#[allow(dead_code)]
pub mod palette {
    use super::active_palette;
    use gpui::Hsla;

    pub fn canvas() -> Hsla {
        active_palette().canvas
    }
    pub fn panel() -> Hsla {
        active_palette().panel
    }
    pub fn element() -> Hsla {
        active_palette().element
    }
    pub fn border() -> Hsla {
        active_palette().border
    }
    /// Softer rail used on the sidebar edge — one tint step below `border`
    /// so the column reads as separation without a visible seam. Contract
    /// line 3 lists `border-subtle ~#2f2f25`.
    pub fn border_subtle() -> Hsla {
        active_palette().border_subtle
    }
    pub fn border_active() -> Hsla {
        active_palette().border_active
    }
    pub fn text() -> Hsla {
        active_palette().text
    }
    pub fn text_muted() -> Hsla {
        active_palette().text_muted
    }
    pub fn text_faint() -> Hsla {
        active_palette().text_faint
    }
    pub fn accent() -> Hsla {
        active_palette().accent
    }
    pub fn accent_hover() -> Hsla {
        active_palette().accent_hover
    }
    pub fn success() -> Hsla {
        active_palette().success
    }
    pub fn warning() -> Hsla {
        active_palette().warning
    }
    pub fn danger() -> Hsla {
        active_palette().danger
    }
    pub fn syntax_number() -> Hsla {
        active_palette().syntax_number
    }
    pub fn syntax_type() -> Hsla {
        active_palette().syntax_type
    }
    pub fn hover() -> Hsla {
        active_palette().hover()
    }
    pub fn active() -> Hsla {
        active_palette().active()
    }
    pub fn selection() -> Hsla {
        active_palette().selection()
    }
    pub fn overlay_strong() -> Hsla {
        active_palette().overlay_strong()
    }
    /// Scrollbar thumb painted as a text-normal mix on transparent — a
    /// warm-off-white at 20% at rest, 40% on hover. Contract line 93 pins
    /// the mix so the thumb reads on any surface without importing a fresh
    /// warm-grey token that would drift when the palette shifts.
    pub fn scrollbar_thumb() -> Hsla {
        active_palette().scrollbar_thumb()
    }
    pub fn scrollbar_thumb_hover() -> Hsla {
        active_palette().scrollbar_thumb_hover()
    }
    /// Composer rail at rest — accent at ~62% opacity. Focus promotes it back
    /// to the full accent so the rail is the composer's focus signal.
    pub fn accent_rail_dim() -> Hsla {
        active_palette().accent_rail_dim()
    }
    /// Fill the composer takes when the input receives focus — one tint step
    /// lighter than the element surface so focus stays visible without a ring.
    pub fn composer_focus_fill() -> Hsla {
        active_palette().composer_focus_fill
    }
    /// Danger 10% tint on canvas — the wiki blocker row's bg. Paired with the
    /// 2px danger left rail so a failure surface reads as an alarm strip
    /// without a full solid-red panel.
    pub fn danger_tint() -> Hsla {
        active_palette().danger_tint()
    }
    /// Text painted over the solid accent / primary surface. Opencode paints
    /// canvas here; light themes typically use a near-black so accent state
    /// pills clear AA contrast.
    pub fn accent_fg() -> Hsla {
        active_palette().accent_fg
    }
    pub fn success_fg() -> Hsla {
        active_palette().success_fg
    }
    pub fn warning_fg() -> Hsla {
        active_palette().warning_fg
    }
    pub fn danger_fg() -> Hsla {
        active_palette().danger_fg
    }
}

// ---------------------------------------------------------------------------
// Palette registry
// ---------------------------------------------------------------------------

static PALETTE_OPENCODE: LazyLock<Palette> = LazyLock::new(|| Palette {
    id: ThemeId::Opencode,
    mode: ThemeMode::Dark,
    canvas: hex(0x1e1e17),
    panel: hex(0x24241b),
    element: hex(0x2c2c21),
    border: hex(0x35352a),
    border_subtle: hex(0x2f2f25),
    border_active: hex(0x706f62),
    text: hex(0xece9d8),
    text_muted: hex(0xa19e88),
    text_faint: hex(0x716f5e),
    accent: hex(0xb18bf4),
    accent_hover: hex(0xc6a9f7),
    success: hex(0xa9c957),
    warning: hex(0xd5d878),
    danger: hex(0xe2685c),
    syntax_number: hex(0xe29a5c),
    syntax_type: hex(0x7fc9b8),
    composer_focus_fill: hex(0x333326),
    accent_fg: hex(0x1e1e17),
    success_fg: hex(0x1e1e17),
    warning_fg: hex(0x1e1e17),
    danger_fg: hex(0x1e1e17),
    syntax: SyntaxHex {
        background: "#1e1e17",
        foreground: "#ece9d8",
        gutter_background: "#1e1e17",
        active_line_background: "#24241b",
        line_number: "#716f5e",
        active_line_number: "#ece9d8",
        invisible: "#716f5e66",
        attribute: "#7fc9b8",
        boolean: "#e29a5c",
        comment: "#716f5e",
        constant: "#e29a5c",
        constructor: "#d5d878",
        embedded: "#ece9d8",
        emphasis: "#ece9d8",
        enum_: "#7fc9b8",
        function: "#d5d878",
        hint: "#a19e88",
        keyword: "#b18bf4",
        label: "#d5d878",
        link_text: "#b18bf4",
        link_uri: "#a19e88",
        number: "#e29a5c",
        operator: "#b18bf4",
        preproc: "#b18bf4",
        property: "#ece9d8",
        punctuation: "#a19e88",
        string: "#a9c957",
        string_escape: "#e29a5c",
        tag: "#b18bf4",
        tag_doctype: "#716f5e",
        text_code_span: "#a9c957",
        text_literal: "#ece9d8",
        title: "#ece9d8",
        type_: "#7fc9b8",
        variable: "#ece9d8",
        variable_special: "#e29a5c",
        variant: "#7fc9b8",
    },
});

static PALETTE_GRUVBOX_DARK: LazyLock<Palette> = LazyLock::new(|| Palette {
    id: ThemeId::GruvboxDark,
    mode: ThemeMode::Dark,
    canvas: hex(0x1d2021),
    panel: hex(0x272828),
    element: hex(0x31302f),
    border: hex(0x45413e),
    border_subtle: hex(0x3a3835),
    border_active: hex(0x7c6f64),
    text: hex(0xebdbb2),
    text_muted: hex(0xa89984),
    text_faint: hex(0x928374),
    accent: hex(0x83a598),
    accent_hover: hex(0x8ec07c),
    success: hex(0xb8bb26),
    warning: hex(0xfabd2f),
    danger: hex(0xfb4934),
    syntax_number: hex(0xd3869b),
    syntax_type: hex(0xfabd2f),
    composer_focus_fill: hex(0x3a3835),
    accent_fg: hex(0x1d2021),
    success_fg: hex(0x1d2021),
    warning_fg: hex(0x1d2021),
    danger_fg: hex(0x1d2021),
    syntax: SyntaxHex {
        background: "#1d2021",
        foreground: "#ebdbb2",
        gutter_background: "#1d2021",
        active_line_background: "#272828",
        line_number: "#928374",
        active_line_number: "#ebdbb2",
        invisible: "#92837466",
        attribute: "#fabd2f",
        boolean: "#d3869b",
        comment: "#928374",
        constant: "#d3869b",
        constructor: "#fabd2f",
        embedded: "#ebdbb2",
        emphasis: "#ebdbb2",
        enum_: "#fabd2f",
        function: "#b8bb26",
        hint: "#a89984",
        keyword: "#fb4934",
        label: "#b8bb26",
        link_text: "#83a598",
        link_uri: "#a89984",
        number: "#d3869b",
        operator: "#fe8019",
        preproc: "#fb4934",
        property: "#ebdbb2",
        punctuation: "#a89984",
        string: "#b8bb26",
        string_escape: "#fe8019",
        tag: "#fb4934",
        tag_doctype: "#928374",
        text_code_span: "#b8bb26",
        text_literal: "#ebdbb2",
        title: "#ebdbb2",
        type_: "#fabd2f",
        variable: "#ebdbb2",
        variable_special: "#fe8019",
        variant: "#fabd2f",
    },
});

static PALETTE_VSCODE_DARK_PLUS: LazyLock<Palette> = LazyLock::new(|| Palette {
    id: ThemeId::VscodeDarkPlus,
    mode: ThemeMode::Dark,
    canvas: hex(0x1f1f1f),
    panel: hex(0x181818),
    element: hex(0x252526),
    border: hex(0x2b2b2b),
    border_subtle: hex(0x252526),
    border_active: hex(0x6e7681),
    text: hex(0xcccccc),
    text_muted: hex(0x9d9d9d),
    text_faint: hex(0x6e7681),
    accent: hex(0x4daafc),
    accent_hover: hex(0x85b6ff),
    success: hex(0x2ea043),
    warning: hex(0xe2c08d),
    danger: hex(0xf85149),
    syntax_number: hex(0xb5cea8),
    syntax_type: hex(0x4ec9b0),
    composer_focus_fill: hex(0x2a2a2b),
    accent_fg: hex(0x1f1f1f),
    success_fg: hex(0x1f1f1f),
    warning_fg: hex(0x1f1f1f),
    danger_fg: hex(0x1f1f1f),
    syntax: SyntaxHex {
        background: "#1f1f1f",
        foreground: "#d4d4d4",
        gutter_background: "#1f1f1f",
        active_line_background: "#252526",
        line_number: "#6e7681",
        active_line_number: "#cccccc",
        invisible: "#6e768166",
        attribute: "#9cdcfe",
        boolean: "#569cd6",
        comment: "#6a9955",
        constant: "#4fc1ff",
        constructor: "#dcdcaa",
        embedded: "#d4d4d4",
        emphasis: "#d4d4d4",
        enum_: "#4ec9b0",
        function: "#dcdcaa",
        hint: "#9d9d9d",
        keyword: "#c586c0",
        label: "#dcdcaa",
        link_text: "#4daafc",
        link_uri: "#9d9d9d",
        number: "#b5cea8",
        operator: "#d4d4d4",
        preproc: "#c586c0",
        property: "#9cdcfe",
        punctuation: "#d4d4d4",
        string: "#ce9178",
        string_escape: "#d7ba7d",
        tag: "#569cd6",
        tag_doctype: "#6a9955",
        text_code_span: "#ce9178",
        text_literal: "#d4d4d4",
        title: "#d4d4d4",
        type_: "#4ec9b0",
        variable: "#9cdcfe",
        variable_special: "#4fc1ff",
        variant: "#4ec9b0",
    },
});

static PALETTE_NORD: LazyLock<Palette> = LazyLock::new(|| Palette {
    id: ThemeId::Nord,
    mode: ThemeMode::Dark,
    canvas: hex(0x2e3440),
    panel: hex(0x3b4252),
    element: hex(0x434c5e),
    border: hex(0x4c566a),
    border_subtle: hex(0x3f4757),
    border_active: hex(0x81a1c1),
    text: hex(0xeceff4),
    text_muted: hex(0xc4cad4),
    text_faint: hex(0x8695a8),
    accent: hex(0x88c0d0),
    accent_hover: hex(0x8fbcbb),
    success: hex(0xa3be8c),
    warning: hex(0xebcb8b),
    danger: hex(0xbf616a),
    syntax_number: hex(0xd08770),
    syntax_type: hex(0x8fbcbb),
    composer_focus_fill: hex(0x4a5364),
    accent_fg: hex(0x2e3440),
    success_fg: hex(0x2e3440),
    warning_fg: hex(0x2e3440),
    // Nord's danger #bf616a sits at mid-luminance where neither `canvas`
    // (2.98:1) nor `text` (3.54:1) clears AA. A near-black label reaches
    // ~4.6:1 against the same red.
    danger_fg: hex(0x101010),
    syntax: SyntaxHex {
        background: "#2e3440",
        foreground: "#eceff4",
        gutter_background: "#2e3440",
        active_line_background: "#3b4252",
        line_number: "#4c566a",
        active_line_number: "#eceff4",
        invisible: "#4c566a66",
        attribute: "#8fbcbb",
        boolean: "#d08770",
        comment: "#81a1c1",
        constant: "#d08770",
        constructor: "#88c0d0",
        embedded: "#eceff4",
        emphasis: "#eceff4",
        enum_: "#8fbcbb",
        function: "#88c0d0",
        hint: "#c4cad4",
        keyword: "#b48ead",
        label: "#88c0d0",
        link_text: "#88c0d0",
        link_uri: "#c4cad4",
        number: "#d08770",
        operator: "#81a1c1",
        preproc: "#b48ead",
        property: "#eceff4",
        punctuation: "#c4cad4",
        string: "#a3be8c",
        string_escape: "#d08770",
        tag: "#b48ead",
        tag_doctype: "#81a1c1",
        text_code_span: "#a3be8c",
        text_literal: "#eceff4",
        title: "#eceff4",
        type_: "#8fbcbb",
        variable: "#e5e9f0",
        variable_special: "#d08770",
        variant: "#8fbcbb",
    },
});

static PALETTE_GRUVBOX_LIGHT: LazyLock<Palette> = LazyLock::new(|| Palette {
    id: ThemeId::GruvboxLight,
    mode: ThemeMode::Light,
    canvas: hex(0xfbf1c7),
    panel: hex(0xf2e5bc),
    element: hex(0xebdbb2),
    border: hex(0xd5c4a1),
    border_subtle: hex(0xd5c4a1),
    border_active: hex(0xa89984),
    text: hex(0x3c3836),
    text_muted: hex(0x665c54),
    text_faint: hex(0x928374),
    accent: hex(0x458588),
    // Hover LIGHTENS the neutral-blue rest fill to Gruvbox's bright_blue
    // #83a598. The obvious "darker teal" direction (faded_blue #076678)
    // drops the near-black `accent_fg` label to 3.00:1 on the primary /
    // info button hover — accent rest is already at the darkness floor
    // #458588 clears 4.68:1, so hover has to go the other way.
    accent_hover: hex(0x83a598),
    success: hex(0x98971a),
    warning: hex(0xd79921),
    danger: hex(0xcc241d),
    syntax_number: hex(0xaf3a03),
    syntax_type: hex(0x076678),
    composer_focus_fill: hex(0xf4e8bd),
    // Gruvbox Light's three brighter solids (teal accent, olive success,
    // amber warning) sit at mid-luminance where the canvas #fbf1c7 falls
    // between 2.19:1 and 3.78:1 against them. A near-black label clears
    // AA against all three; only danger #cc241d — the darkest solid —
    // pairs with the light canvas.
    accent_fg: hex(0x0a0a0a),
    success_fg: hex(0x0a0a0a),
    warning_fg: hex(0x0a0a0a),
    danger_fg: hex(0xfbf1c7),
    syntax: SyntaxHex {
        background: "#fbf1c7",
        foreground: "#3c3836",
        gutter_background: "#fbf1c7",
        active_line_background: "#f2e5bc",
        line_number: "#928374",
        active_line_number: "#3c3836",
        invisible: "#92837466",
        attribute: "#076678",
        boolean: "#af3a03",
        comment: "#928374",
        constant: "#af3a03",
        constructor: "#79740e",
        embedded: "#3c3836",
        emphasis: "#3c3836",
        enum_: "#076678",
        function: "#79740e",
        hint: "#665c54",
        keyword: "#8f3f71",
        label: "#79740e",
        link_text: "#458588",
        link_uri: "#665c54",
        number: "#af3a03",
        operator: "#8f3f71",
        preproc: "#8f3f71",
        property: "#3c3836",
        punctuation: "#665c54",
        string: "#427b58",
        string_escape: "#af3a03",
        tag: "#8f3f71",
        tag_doctype: "#928374",
        text_code_span: "#427b58",
        text_literal: "#3c3836",
        title: "#3c3836",
        type_: "#076678",
        variable: "#b57614",
        variable_special: "#af3a03",
        variant: "#076678",
    },
});

fn highlight_theme_for(p: &Palette) -> Arc<HighlightTheme> {
    // Built programmatically (not via `serde_json::json!`) because the macro
    // hits the crate-wide recursion limit on maps this wide.
    use serde_json::{Map, Value};
    fn styled(color: &str) -> Value {
        let mut m = Map::new();
        m.insert("color".to_string(), Value::String(color.to_string()));
        Value::Object(m)
    }
    fn styled_italic(color: &str) -> Value {
        let mut m = Map::new();
        m.insert("color".to_string(), Value::String(color.to_string()));
        m.insert(
            "font_style".to_string(),
            Value::String("italic".to_string()),
        );
        Value::Object(m)
    }
    fn styled_bold(color: &str) -> Value {
        let mut m = Map::new();
        m.insert("color".to_string(), Value::String(color.to_string()));
        m.insert("font_weight".to_string(), Value::from(700));
        Value::Object(m)
    }
    let s = p.syntax;
    let mut syntax = Map::new();
    syntax.insert("attribute".into(), styled(s.attribute));
    syntax.insert("boolean".into(), styled(s.boolean));
    syntax.insert("comment".into(), styled_italic(s.comment));
    syntax.insert("comment.doc".into(), styled_italic(s.comment));
    syntax.insert("constant".into(), styled(s.constant));
    syntax.insert("constructor".into(), styled(s.constructor));
    syntax.insert("embedded".into(), styled(s.embedded));
    syntax.insert("emphasis".into(), styled_italic(s.emphasis));
    syntax.insert("emphasis.strong".into(), styled_bold(s.emphasis));
    syntax.insert("enum".into(), styled(s.enum_));
    syntax.insert("function".into(), styled(s.function));
    syntax.insert("hint".into(), styled(s.hint));
    syntax.insert("keyword".into(), styled(s.keyword));
    syntax.insert("label".into(), styled(s.label));
    syntax.insert("link_text".into(), styled(s.link_text));
    syntax.insert("link_uri".into(), styled_italic(s.link_uri));
    syntax.insert("number".into(), styled(s.number));
    syntax.insert("operator".into(), styled(s.operator));
    syntax.insert("preproc".into(), styled(s.preproc));
    syntax.insert("property".into(), styled(s.property));
    syntax.insert("punctuation".into(), styled(s.punctuation));
    syntax.insert("punctuation.bracket".into(), styled(s.punctuation));
    syntax.insert("punctuation.delimiter".into(), styled(s.punctuation));
    syntax.insert("punctuation.list_marker".into(), styled(s.keyword));
    syntax.insert("punctuation.special".into(), styled(s.keyword));
    syntax.insert("string".into(), styled(s.string));
    syntax.insert("string.escape".into(), styled(s.string_escape));
    syntax.insert("string.regex".into(), styled(s.string));
    syntax.insert("string.special".into(), styled(s.string_escape));
    syntax.insert("string.special.symbol".into(), styled(s.string_escape));
    syntax.insert("tag".into(), styled(s.tag));
    syntax.insert("tag.doctype".into(), styled(s.tag_doctype));
    syntax.insert("text.code.span".into(), styled(s.text_code_span));
    syntax.insert("text.literal".into(), styled(s.text_literal));
    syntax.insert("title".into(), styled_bold(s.title));
    syntax.insert("type".into(), styled(s.type_));
    syntax.insert("variable".into(), styled(s.variable));
    syntax.insert("variable.special".into(), styled(s.variable_special));
    syntax.insert("variant".into(), styled(s.variant));

    let mut root = Map::new();
    root.insert(
        "editor.background".into(),
        Value::String(s.background.to_string()),
    );
    root.insert(
        "editor.foreground".into(),
        Value::String(s.foreground.to_string()),
    );
    root.insert(
        "editor.active_line.background".into(),
        Value::String(s.active_line_background.to_string()),
    );
    root.insert(
        "editor.line_number".into(),
        Value::String(s.line_number.to_string()),
    );
    root.insert(
        "editor.active_line_number".into(),
        Value::String(s.active_line_number.to_string()),
    );
    root.insert(
        "editor.invisible".into(),
        Value::String(s.invisible.to_string()),
    );
    root.insert(
        "editor.gutter.background".into(),
        Value::String(s.gutter_background.to_string()),
    );
    root.insert("syntax".into(), Value::Object(syntax));

    Arc::new(HighlightTheme {
        name: format!("Zeta {}", p.id.label()),
        appearance: p.mode,
        style: serde_json::from_value(Value::Object(root))
            .expect("highlight theme JSON is well-formed"),
    })
}

/// Base rich-text style for zeta's app-wide `TextViewDefaults`.
///
/// This mirrors gpui-component's crate-private `base_text_view_style`
/// field-for-field from the same `cx.theme()` tokens, then swaps only
/// `inline_code` so any UNSTYLED `TextView::markdown` in the app (input
/// popovers, error surfaces, previews, gpui-component-internal renders)
/// paints inline `code` on the wiki `.markdown-preview-view code` wash
/// instead of gpui-component's shipped `theme.accent` slab.
///
/// The rest of the fields track the component defaults exactly — link,
/// selection, code-block corner radii, table corner radii, table-head
/// bg/fg refinement — so consumers keep the themed layout and only the
/// inline-chip color moves. The assistant path routes through
/// `resolve_component_style` and folds its own `HighlightStyle` on top,
/// so this default only matters when a caller does NOT provide a local
/// `.style(...)`; every unstyled TextView in zeta or in the component
/// library it re-exports picks up the subtle wash here.
pub(crate) fn zeta_text_view_style(theme: &Theme) -> gpui_kit::base::TextViewStyle {
    let radius = theme.semantic_tokens().radius.md;
    let corner_radii = gpui::CornersRefinement {
        top_left: Some(radius.into()),
        top_right: Some(radius.into()),
        bottom_left: Some(radius.into()),
        bottom_right: Some(radius.into()),
    };
    let table = StyleRefinement {
        corner_radii: corner_radii.clone(),
        ..Default::default()
    };
    let code_block = StyleRefinement {
        corner_radii,
        ..Default::default()
    };
    let table_head = StyleRefinement::default()
        .bg(theme.table_head)
        .text_color(theme.table_head_foreground);
    // Inline `code` sits on a subtle text-normal wash at normal-tier
    // glyph color — the exact pair the assistant renderer uses through
    // `assistant_markdown_style`. Routing the DEFAULT through the same
    // pair keeps every unstyled TextView on one inline-code shape.
    let inline_code = gpui::HighlightStyle {
        background_color: Some(theme.secondary_hover),
        color: Some(theme.foreground),
        ..Default::default()
    };
    gpui_kit::base::TextViewStyle::default()
        .with_foreground(theme.foreground)
        .with_muted_foreground(theme.muted_foreground)
        .with_link(theme.link)
        .with_selection(theme.selection)
        .with_code_background(theme.muted)
        .with_border(theme.border)
        .with_code_block(code_block)
        .with_table(table)
        .with_table_head(table_head)
        .with_inline_code(inline_code)
        .with_dark(theme.is_dark())
}

/// Apply the shipped default appearance — opencode, JetBrains Mono, 13px.
/// Kept as the zero-arg entry point so tests / smoke drivers that only want
/// the default look call one function. Main.rs boots through `apply_with`
/// on the loaded prefs.
#[allow(dead_code)]
pub fn apply(cx: &mut App) {
    apply_with(cx, &Appearance::default());
}

/// Apply an explicit appearance snapshot onto the global theme, refresh the
/// palette accessors, and push the update down to the Base layer so
/// scrollbars, popovers, and every kit-driven surface paint with the same
/// tokens. Callable at boot (first paint) and again at runtime whenever the
/// user picks a new theme / font family / font size from Settings.
pub fn apply_with(cx: &mut App, appearance: &Appearance) {
    // Update the palette pointer BEFORE mutating the gpui-kit theme so any
    // `palette::*` read taken during `sync_base` sees the new palette.
    set_active(appearance.clone());
    let palette = appearance.theme.palette();

    let theme = Theme::global_mut(cx);
    theme.mode = palette.mode;

    theme.font_family = appearance.font_family.clone();
    theme.mono_font_family = appearance.font_family.clone();
    theme.font_size = appearance.font_size;
    theme.mono_font_size = appearance.font_size;

    theme.radius = RADIUS;
    theme.radius_lg = RADIUS;
    theme.tile_radius = px(0.);
    theme.shadow = false;
    theme.tile_shadow = false;
    theme.focus_ring = true;

    // Install the palette's syntax colors BEFORE `sync_base` snapshots the
    // highlight theme into the code-block highlighter.
    theme.highlight_theme = highlight_theme_for(palette);

    let colors = &mut theme.colors;

    colors.background = palette.canvas;
    colors.foreground = palette.text;
    colors.border = palette.border;
    colors.drag_border = palette.border_active;
    colors.input = palette.border;

    colors.muted = palette.element;
    colors.muted_foreground = palette.text_muted;

    colors.accent = palette.accent;
    colors.accent_foreground = palette.accent_fg;

    colors.primary = palette.accent;
    colors.primary_foreground = palette.accent_fg;
    colors.primary_active = palette.accent;
    colors.primary_hover = palette.accent_hover;

    colors.secondary = palette.element;
    colors.secondary_foreground = palette.text;
    colors.secondary_active = palette.panel;
    colors.secondary_hover = palette.hover();

    colors.button = palette.element;
    colors.button_foreground = palette.text;
    colors.button_hover = palette.hover();
    colors.button_active = palette.panel;

    colors.button_primary = palette.accent;
    colors.button_primary_foreground = palette.accent_fg;
    colors.button_primary_hover = palette.accent_hover;
    colors.button_primary_active = palette.accent;

    colors.button_secondary = palette.element;
    colors.button_secondary_foreground = palette.text;
    colors.button_secondary_hover = palette.hover();
    colors.button_secondary_active = palette.panel;

    colors.button_danger = palette.danger;
    colors.button_danger_foreground = palette.danger_fg;
    colors.button_danger_hover = palette.danger;
    colors.button_danger_active = palette.danger;

    colors.button_warning = palette.warning;
    colors.button_warning_foreground = palette.warning_fg;
    colors.button_warning_hover = palette.warning;
    colors.button_warning_active = palette.warning;

    colors.button_success = palette.success;
    colors.button_success_foreground = palette.success_fg;
    colors.button_success_hover = palette.success;
    colors.button_success_active = palette.success;

    colors.button_info = palette.accent;
    colors.button_info_foreground = palette.accent_fg;
    colors.button_info_hover = palette.accent_hover;
    colors.button_info_active = palette.accent;

    colors.popover = palette.panel;
    colors.popover_foreground = palette.text;

    colors.sidebar = palette.panel;
    colors.sidebar_foreground = palette.text;
    colors.sidebar_border = palette.border_subtle;
    colors.sidebar_accent = palette.active();
    colors.sidebar_accent_foreground = palette.text;
    colors.sidebar_primary = palette.accent;
    colors.sidebar_primary_foreground = palette.accent_fg;

    colors.list = palette.panel;
    colors.list_hover = palette.hover();
    colors.list_active = palette.active();
    colors.list_active_border = palette.accent;
    colors.list_even = palette.panel;
    colors.list_head = palette.panel;

    colors.tab = palette.panel;
    colors.tab_active = palette.element;
    colors.tab_active_foreground = palette.text;
    colors.tab_bar = palette.panel;
    colors.tab_bar_segmented = palette.element;
    colors.tab_foreground = palette.text_muted;

    colors.description_list_label = palette.panel;
    colors.description_list_label_foreground = palette.text_faint;

    colors.table = palette.panel;
    colors.table_head = palette.element;
    colors.table_head_foreground = palette.text;
    colors.table_foot = palette.panel;
    colors.table_foot_foreground = palette.text;
    colors.table_even = palette.panel;
    colors.table_hover = palette.hover();
    colors.table_active = palette.active();
    colors.table_active_border = palette.accent;
    colors.table_row_border = palette.border;

    colors.link = palette.accent;
    colors.link_active = palette.accent;
    colors.link_hover = palette.accent_hover;

    colors.title_bar = palette.panel;
    colors.title_bar_border = palette.border;
    colors.status_bar = palette.panel;
    colors.status_bar_border = palette.border;

    colors.danger = palette.danger;
    colors.danger_foreground = palette.danger_fg;
    colors.danger_active = palette.danger;
    colors.danger_hover = palette.danger;
    colors.warning = palette.warning;
    colors.warning_foreground = palette.warning_fg;
    colors.warning_hover = palette.warning;
    colors.warning_active = palette.warning;
    colors.success = palette.success;
    colors.success_foreground = palette.success_fg;
    colors.success_hover = palette.success;
    colors.success_active = palette.success;
    colors.info = palette.accent;
    colors.info_foreground = palette.accent_fg;
    colors.info_hover = palette.accent_hover;
    colors.info_active = palette.accent;

    colors.ring = palette.accent;
    colors.caret = palette.accent;
    colors.selection = palette.selection();

    colors.scrollbar = gpui::transparent_black();
    colors.scrollbar_thumb = palette.scrollbar_thumb();
    colors.scrollbar_thumb_hover = palette.scrollbar_thumb_hover();

    colors.overlay = palette.overlay_strong();
    colors.drop_target = palette.active();

    colors.accordion = palette.panel;
    colors.group_box = palette.panel;
    colors.group_box_foreground = palette.text;
    colors.progress_bar = palette.accent;
    colors.skeleton = palette.element;
    colors.slider_bar = palette.element;
    colors.slider_thumb = palette.accent;
    colors.switch = palette.element;
    colors.switch_thumb = palette.text;
    colors.tiles = palette.panel;
    colors.window_border = palette.border;

    colors.red = palette.danger;
    colors.red_light = palette.danger;
    colors.green = palette.success;
    colors.green_light = palette.success;
    colors.blue = palette.accent;
    colors.blue_light = palette.accent_hover;
    colors.yellow = palette.warning;
    colors.yellow_light = palette.warning;
    colors.magenta = palette.accent;
    colors.magenta_light = palette.accent_hover;
    colors.cyan = palette.syntax_type;
    colors.cyan_light = palette.syntax_type;

    colors.chart_1 = palette.accent;
    colors.chart_2 = palette.success;
    colors.chart_3 = palette.warning;
    colors.chart_4 = palette.syntax_number;
    colors.chart_5 = palette.syntax_type;
    colors.chart_bullish = palette.success;
    colors.chart_bearish = palette.danger;

    theme.tokens = (&theme.colors).into();

    Theme::sync_base(cx);

    gpui_kit::base::TextViewDefaults::global(cx)
        .with_style(zeta_text_view_style(cx.theme()))
        .install(cx);

    let base = gpui_kit::base::Theme::global_mut(cx);
    base.scrollbar = base.scrollbar.clone().with_styles(
        gpui_kit::base::ScrollbarStyles::default()
            .track(|style| style.bg(gpui::transparent_black()))
            .track_hover(|style| style.bg(gpui::transparent_black()))
            .track_active(|style| style.bg(gpui::transparent_black()))
            .thumb(|style| {
                style
                    .bg(palette.scrollbar_thumb())
                    .width(SCROLLBAR_THUMB_WIDTH)
                    .radius(px(0.))
            })
            .thumb_hover(|style| {
                style
                    .bg(palette.scrollbar_thumb_hover())
                    .width(SCROLLBAR_THUMB_WIDTH)
                    .radius(px(0.))
            })
            .thumb_active(|style| {
                style
                    .bg(palette.scrollbar_thumb_hover())
                    .width(SCROLLBAR_THUMB_WIDTH)
                    .radius(px(0.))
            }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use gpui_kit::component::ActiveTheme;

    fn poison() -> Hsla {
        hex(0xff00ff)
    }

    /// Reset the active appearance to the shipped default. Tests that run
    /// in-process after another test flipped the theme use this to keep the
    /// palette assertions below deterministic.
    fn reset_active_default() {
        set_active(Appearance::default());
    }

    fn opencode() -> &'static Palette {
        &PALETTE_OPENCODE
    }

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
            let op = opencode();
            assert_eq!(theme.background, op.canvas);
            assert_eq!(theme.foreground, op.text);
            assert_eq!(theme.muted_foreground, op.text_muted);
            assert_eq!(theme.border, op.border);
            assert_eq!(theme.accent, op.accent);
            assert_eq!(theme.accent_foreground, op.canvas);
            assert_eq!(theme.primary, op.accent);
            assert_eq!(theme.ring, op.accent);
            assert_eq!(theme.danger, op.danger);
            assert_eq!(theme.success, op.success);
            assert_eq!(theme.warning, op.warning);
            assert_eq!(theme.sidebar, op.panel);
            assert_eq!(theme.popover, op.panel);
            assert_eq!(theme.overlay, op.overlay_strong());
            assert_eq!(theme.selection, op.selection());
            assert_eq!(theme.scrollbar_thumb, op.scrollbar_thumb());
            assert_eq!(theme.scrollbar_thumb_hover, op.scrollbar_thumb_hover());
            assert!(
                theme.scrollbar_thumb.a > 0.1 && theme.scrollbar_thumb.a < 0.35,
                "thumb rest alpha {:.3} must land near the 20% contract mix",
                theme.scrollbar_thumb.a
            );
            assert!(
                theme.scrollbar_thumb_hover.a > 0.3 && theme.scrollbar_thumb_hover.a < 0.55,
                "thumb hover alpha {:.3} must land near the 40% contract mix",
                theme.scrollbar_thumb_hover.a
            );
            assert_eq!(theme.scrollbar_thumb.h, op.text.h);
            assert_eq!(theme.scrollbar_thumb.s, op.text.s);
            assert_eq!(theme.scrollbar_thumb.l, op.text.l);
        });
        reset_active_default();
    }

    #[gpui::test]
    fn every_rendered_component_field_is_repainted(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(|cx| {
            let colors = &mut Theme::global_mut(cx).colors;
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

            let op = opencode();
            assert_eq!(theme.button, op.element);
            assert_eq!(theme.button_foreground, op.text);
            assert_eq!(theme.button_primary, op.accent);
            assert_eq!(theme.button_primary_foreground, op.canvas);
            assert_eq!(theme.table, op.panel);
            assert_eq!(theme.table_head, op.element);
            assert_eq!(theme.table_head_foreground, op.text);
            assert_eq!(theme.table_foot_foreground, op.text);
        });
        reset_active_default();
    }

    #[gpui::test]
    fn table_label_pairs_clear_wcag_aa_contrast(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
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
        reset_active_default();
    }

    #[gpui::test]
    fn highlight_theme_paints_code_fences_in_palette(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
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
            assert_eq!(highlight.name, "Zeta Opencode");
            assert_eq!(highlight.appearance, ThemeMode::Dark);

            let keyword = highlight
                .style
                .syntax
                .style("keyword")
                .expect("keyword style is set");
            assert_eq!(keyword.color, Some(opencode().accent));

            let string = highlight
                .style
                .syntax
                .style("string")
                .expect("string style is set");
            assert_eq!(string.color, Some(opencode().success));

            let comment = highlight
                .style
                .syntax
                .style("comment")
                .expect("comment style is set");
            assert_eq!(comment.color, Some(opencode().text_faint));

            let editor_bg = highlight
                .style
                .editor_background
                .expect("editor background is set");
            assert_eq!(editor_bg, opencode().canvas);
        });
        reset_active_default();
    }

    #[test]
    fn sidebar_and_chrome_tokens_land_on_the_wiki_contract() {
        assert_eq!(SIDEBAR_WIDTH, px(216.));
        assert_eq!(SIDEBAR_ROW_HEIGHT, px(40.));
        assert_eq!(SIDEBAR_NESTED_ROW_HEIGHT, px(32.));
        assert_eq!(SIDEBAR_ROW_PADDING_X, px(8.));
        assert_eq!(SIDEBAR_ROW_PADDING_Y, px(5.));
        assert_eq!(SIDEBAR_GUTTER_WIDTH, px(13.));
        assert_eq!(SIDEBAR_CURRENT_DOT_SIZE, px(9.));
        assert_eq!(SIDEBAR_CURRENT_DOT_INSET, px(4.));
        assert_eq!(SCROLLBAR_THUMB_WIDTH, px(8.));
        assert_eq!(ATTENTION_RAIL_WIDTH, px(2.));
        assert_eq!(HEADER_BAND1_MIN_HEIGHT, px(44.));
        assert_eq!(STATUS_RULE_HEIGHT, px(14.));
        assert_eq!(MODAL_WIDTH, px(480.));
        assert_eq!(MODAL_PADDING_TOP, px(12.));
        assert_eq!(MODAL_PADDING_X, px(16.));
        assert_eq!(MODAL_PADDING_BOTTOM, px(14.));
        assert_eq!(MODAL_BUTTON_HEIGHT, px(30.));
        assert!((MODAL_TOP_FRACTION - 0.25).abs() < f32::EPSILON);

        let tint = opencode().danger_tint();
        let danger = opencode().danger;
        assert_eq!(tint.h, danger.h);
        assert_eq!(tint.s, danger.s);
        assert_eq!(tint.l, danger.l);
        assert!(tint.a > 0.05 && tint.a < 0.20);
    }

    #[test]
    fn transcript_and_composer_tokens_land_on_the_wiki_contract() {
        assert_eq!(TRANSCRIPT_MAX_WIDTH, px(1024.));
        assert_eq!(TRANSCRIPT_ROW_GAP, px(14.));
        assert_eq!(COMPOSER_MIN_HEIGHT, px(64.));
        assert_eq!(COMPOSER_PADDING_Y, px(8.));
        assert_eq!(COMPOSER_PADDING_X, px(10.));
        assert_eq!(SEND_BUTTON_MIN_WIDTH, px(82.));
        assert_eq!(RAIL_WIDTH_THICK, px(3.));
        assert_eq!(RAIL_WIDTH_THIN, px(1.));
        assert_eq!(STREAM_DOT_SIZE, px(7.));

        let rail = opencode().accent_rail_dim();
        let accent = opencode().accent;
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

        assert!(opencode().composer_focus_fill.l > opencode().element.l);
    }

    #[gpui::test]
    fn shape_and_typography_pin_flat_terminal_defaults(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.radius, RADIUS);
            assert_eq!(theme.radius_lg, RADIUS);
            assert_eq!(theme.font_size, DEFAULT_FONT_SIZE);
            assert_eq!(theme.mono_font_size, DEFAULT_FONT_SIZE);
            assert_eq!(theme.font_family.as_ref(), DEFAULT_FONT_FAMILY);
            assert_eq!(theme.mono_font_family.as_ref(), DEFAULT_FONT_FAMILY);
            assert!(!theme.shadow);
            assert!(!theme.tile_shadow);
            assert!(theme.focus_ring);
            assert!(theme.is_dark());
        });
        reset_active_default();
    }

    #[gpui::test]
    fn sidebar_border_lands_on_the_subtle_tier(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(apply);
        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.sidebar_border, opencode().border_subtle);
            assert_ne!(theme.sidebar_border, opencode().border);
        });
        reset_active_default();
    }

    #[gpui::test]
    fn semantic_tokens_project_the_flat_radius_and_no_elevation(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
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
            assert_eq!(semantic.colors.accent_foreground, opencode().canvas);

            let base = gpui_kit::base::Theme::global(cx);
            assert_eq!(base.tokens.radius.md, RADIUS);
            assert!(base.tokens.shadow.md.is_empty());
            assert!(base.tokens.shadow.lg.is_empty());
            assert_eq!(base.tokens.colors.accent_foreground, opencode().canvas);
        });
        reset_active_default();
    }

    #[gpui::test]
    fn non_default_theme_shifts_canvas_and_accent_tokens(cx: &mut TestAppContext) {
        // Mutation-style: swapping the active theme must move the canvas +
        // accent + palette accessors off opencode. If a future refactor
        // dropped `apply_with`'s palette write, this test fails.
        cx.update(gpui_kit::init);
        cx.update(apply);

        cx.update(|cx| {
            assert_eq!(cx.theme().background, opencode().canvas);
            assert_eq!(palette::accent(), opencode().accent);
        });

        cx.update(|cx| {
            apply_with(
                cx,
                &Appearance {
                    theme: ThemeId::GruvboxLight,
                    ..Appearance::default()
                },
            );
        });

        // Assertions read the PER-APP theme (`cx.theme()`), not the
        // process-wide `palette::*` accessors — those share state across
        // parallel gpui-test workers, and a peer test can race the ACTIVE
        // slot between our write and the assertion.
        cx.update(|cx| {
            let light = ThemeId::GruvboxLight.palette();
            assert_eq!(cx.theme().background, light.canvas);
            assert_eq!(cx.theme().foreground, light.text);
            assert_eq!(cx.theme().accent, light.accent);
            assert!(!cx.theme().is_dark());
            assert_ne!(cx.theme().background, opencode().canvas);
            assert_ne!(cx.theme().accent, opencode().accent);
        });

        reset_active_default();
    }

    #[gpui::test]
    fn font_size_and_family_update_live_from_appearance(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        cx.update(|cx| {
            apply_with(
                cx,
                &Appearance {
                    theme: ThemeId::Opencode,
                    font_family: SharedString::new_static("Menlo"),
                    font_size: px(17.),
                },
            );
        });

        cx.update(|cx| {
            let theme = cx.theme();
            assert_eq!(theme.font_size, px(17.));
            assert_eq!(theme.mono_font_size, px(17.));
            assert_eq!(theme.font_family.as_ref(), "Menlo");
            assert_eq!(theme.mono_font_family.as_ref(), "Menlo");
        });

        reset_active_default();
    }

    #[test]
    fn clamp_font_size_snaps_to_whole_px_in_range() {
        assert_eq!(clamp_font_size(9.0), px(MIN_FONT_SIZE_PX));
        assert_eq!(clamp_font_size(11.4), px(11.));
        assert_eq!(clamp_font_size(13.6), px(14.));
        assert_eq!(clamp_font_size(18.0), px(MAX_FONT_SIZE_PX));
        assert_eq!(clamp_font_size(999.0), px(MAX_FONT_SIZE_PX));
    }

    #[test]
    fn every_solid_semantic_pair_clears_wcag_aa() {
        // Text painted on a solid semantic fill (accent state pill, primary
        // button, warning/danger button) must reach the WCAG AA normal-text
        // ratio of 4.5:1 in every state that repaints the fill. The
        // regressions this catches:
        //   * Routing one foreground onto every semantic drops several rest
        //     pairs below that bar (VSCode warning 1.73:1, Gruvbox Light
        //     warning 2.19:1, Nord danger 3.05:1 on the pre-fix single-fg
        //     palette).
        //   * Picking an `accent_hover` fill that stays paired with
        //     `accent_fg` on the primary / info button hover state but
        //     drops below 4.5:1 (Gruvbox Light 3.00:1 on the pre-fix
        //     faded-blue #076678).
        //
        // Coverage picks the (fill, fg) pair actually painted:
        //   * Rest: `<semantic>` fill + `<semantic>_fg` (theme.rs primary
        //     button assigns button_primary_foreground = accent_fg, etc.).
        //   * Hover: primary + info buttons swap to `accent_hover` while
        //     keeping `accent_fg` on top (theme.rs button_primary_hover /
        //     button_info_hover). success / warning / danger hover reuse
        //     the rest fill (theme.rs button_*_hover = palette.<semantic>),
        //     so their hover check is subsumed by the rest pair.
        //   * Active: every solid semantic button repaints the rest fill
        //     (theme.rs button_*_active = palette.<semantic>), so no extra
        //     pair is painted.
        for id in ThemeId::ALL {
            let p = id.palette();
            for (label, bg, fg) in [
                ("accent rest", p.accent, p.accent_fg),
                ("success rest", p.success, p.success_fg),
                ("warning rest", p.warning, p.warning_fg),
                ("danger rest", p.danger, p.danger_fg),
                ("accent hover", p.accent_hover, p.accent_fg),
            ] {
                let ratio = contrast_ratio(fg, bg);
                assert!(
                    ratio >= 4.5,
                    "{} {label}: {ratio:.2}:1 fails WCAG AA (need >=4.5:1)",
                    id.label(),
                );
            }
        }
    }

    #[test]
    fn label_size_roles_scale_with_the_base_font_size() {
        // Small / micro label roles derive from the base font size so
        // attachment chips, composer target, tool hints, login status,
        // and thumbnail-fallback captions reflow when the user picks a
        // new base. Guards against re-introducing a hardcoded `px(12.)`
        // that ignores the appearance picker.
        let small_low = label_small(px(MIN_FONT_SIZE_PX));
        let small_high = label_small(px(MAX_FONT_SIZE_PX));
        assert_ne!(
            small_low, small_high,
            "small role must move when the base font size moves"
        );
        assert!(f32::from(small_low) < f32::from(small_high));

        let micro_low = label_micro(px(MIN_FONT_SIZE_PX));
        let micro_high = label_micro(px(MAX_FONT_SIZE_PX));
        assert_ne!(
            micro_low, micro_high,
            "micro role must move when the base font size moves"
        );
        assert!(f32::from(micro_low) < f32::from(micro_high));

        // Roles land below the base at both bounds, and never below the
        // legibility floor.
        assert!(f32::from(small_high) < MAX_FONT_SIZE_PX);
        assert!(f32::from(micro_high) < MAX_FONT_SIZE_PX);
        assert!(f32::from(small_low) >= MIN_LABEL_PX);
        assert!(f32::from(micro_low) >= MIN_LABEL_PX);
    }

    #[test]
    fn every_theme_has_a_usable_palette_and_slug_roundtrips() {
        for id in ThemeId::ALL {
            let p = id.palette();
            assert_eq!(p.id, *id);
            let round = ThemeId::from_slug(id.slug()).expect("slug roundtrips");
            assert_eq!(round, *id);

            // Body contrast must clear WCAG AA against the canvas so every
            // shipped theme stays legible without further work.
            let ratio = contrast_ratio(p.text, p.canvas);
            assert!(
                ratio >= 4.5,
                "{}: text/canvas contrast {ratio:.2}:1 below AA",
                id.label()
            );

            // Alpha derivations must land in the documented mix ranges even
            // when the base moves — a regression that lost the derivation
            // (e.g. baked a fixed alpha) surfaces here.
            let hover = p.hover();
            assert!((0.04..=0.10).contains(&hover.a), "hover alpha drift");
            let active = p.active();
            assert!((0.10..=0.20).contains(&active.a), "active alpha drift");
        }
        assert_eq!(ThemeId::from_slug("does-not-exist"), None);
    }
}
