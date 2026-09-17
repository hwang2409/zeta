//! Typed model of every user-visible string a transcript row can paint.
//!
//! One struct per row kind whose fields NAME every visible string the row
//! renders — body text, action/button labels, attachment names and sizes,
//! tool output sizes, chevrons/hints, and fixed chrome. The renderers in
//! `transcript_render.rs` build their visible-text elements by destructuring
//! this model; no visible string is sourced outside it. Together with the
//! `renderer_literal_fence` AST guard — which parses the transcript render
//! module and rejects ANY string or byte-string literal that does not sit
//! inside the tight allowed-context allowlist — the two make a NEW stray
//! literal impossible to add without failing a test.
//!
//! The typed model REPLACES the earlier positional
//! `TranscriptEntry::visible_text -> Vec<&str>` seam. That seam covered row
//! CONTENT only; action labels, attachment names/sizes, tool output sizes,
//! AND the entire login-row surface still flowed from entry fields via
//! inline `format!` calls in the render layer, so a new stray literal on
//! any of those paths bypassed the seam. The typed model closes those
//! paths — the login row now flows through `LoginRowText` too.
//!
//! Borrowing: content-shaped fields (`content`, `source`, `verb`, `detail`,
//! `body`, `message`) hold a `&str` into the transcript entry, so a per-row
//! `build()` allocates only for the composed labels (attachment lines,
//! output-size string, login labels). A per-frame full-transcript clone is
//! not introduced — this matches the ZETA-107 r5 borrowing invariant.

use crate::cards::EditData;
use crate::login::{LoginProgress, LoginProvider};
use crate::session::SessionView;
use crate::state::{ConnectionState, TranscriptEntry, ERROR_HEADER_LABEL, THINKING_HEADER_LABEL};

/// Every fixed literal painted as row chrome (headings, hints, unit
/// suffixes, action labels). Every user-visible string that reaches a row
/// but does not come from an entry field lives here. `ALL` is the exhaustive
/// set the guard tests iterate.
pub mod chrome {
    pub const ASSISTANT_TRUNCATED: &str = "Showing the latest streamed text…";
    pub const TOOL_HOVER_HINT: &str = "show output";
    pub const TOOL_TAIL_OMITTED: &str = "Earlier output omitted";
    /// Singular unit painted after the count in a tool-group summary
    /// row ("1 tool call").
    pub const TOOL_GROUP_CALL: &str = " tool call";
    /// Plural unit painted after the count in a tool-group summary row
    /// ("N tool calls" for N != 1).
    pub const TOOL_GROUP_CALLS: &str = " tool calls";
    /// Separator that precedes the total-size metadata inside the summary
    /// row (" · 12.4KB").
    pub const TOOL_GROUP_META_SEPARATOR: &str = " · ";
    /// Suffix on the accessible label announcing that the group is
    /// currently collapsed (Enter/Space expands).
    pub const TOOL_GROUP_ARIA_COLLAPSED: &str = ", collapsed";
    /// Suffix on the accessible label announcing that the group is
    /// currently expanded (Enter/Space collapses).
    pub const TOOL_GROUP_ARIA_EXPANDED: &str = ", expanded";
    pub const OUTPUT_SIZE_UNIT_B: &str = "B";
    pub const OUTPUT_SIZE_UNIT_KB: &str = "KB";
    pub const OUTPUT_SIZE_UNIT_MB: &str = "MB";
    pub const ATTACHMENT_SIZE_SEPARATOR: &str = " · ";
    pub const ATTACHMENT_SIZE_SUFFIX: &str = " bytes";
    pub const FORK_HERE: &str = "Fork here";
    pub const OPEN_SETTINGS: &str = "Open Settings";
    pub const LOGIN_START_PREFIX: &str = "Log in with ";
    pub const LOGIN_CANCEL: &str = "Cancel";

    // ZETA-135 (Trait 1 — kind glyphs). Small, high-scannability leading
    // glyph on every tool receipt — a `$` for shell, an arrow for edit-
    // shaped tools, an up-right arrow for network fetches, a gear for
    // everything else. Reads as visual chunking without a legend (laws-of-
    // ux Selective Attention + Chunking); a scan of a receipt column
    // answers "what KIND of thing ran" before "which tool" and "with what
    // arg". Sourced once here so the renderer never composes a glyph
    // inline.
    pub const TOOL_KIND_SHELL: &str = "$";
    pub const TOOL_KIND_EDIT: &str = "←";
    pub const TOOL_KIND_FETCH: &str = "↗";
    pub const TOOL_KIND_SEARCH: &str = "⌕";
    pub const TOOL_KIND_GENERIC: &str = "⚙";

    /// Separator used between metadata cells in the transcript-end turn
    /// footer ("cc · claude-fable-5 · 6m 32s"). Same middle-dot rhythm as
    /// `TOOL_GROUP_META_SEPARATOR` but hoisted to its own name so the two
    /// call sites stay independent.
    pub const TURN_FOOTER_META_SEPARATOR: &str = " · ";

    /// Exhaustive set the guard tests iterate. Adding a new chrome literal
    /// without adding it here fails the fence's chrome-coverage check.
    pub const ALL: &[&str] = &[
        ASSISTANT_TRUNCATED,
        TOOL_HOVER_HINT,
        TOOL_TAIL_OMITTED,
        TOOL_GROUP_CALL,
        TOOL_GROUP_CALLS,
        TOOL_GROUP_META_SEPARATOR,
        TOOL_GROUP_ARIA_COLLAPSED,
        TOOL_GROUP_ARIA_EXPANDED,
        OUTPUT_SIZE_UNIT_B,
        OUTPUT_SIZE_UNIT_KB,
        OUTPUT_SIZE_UNIT_MB,
        ATTACHMENT_SIZE_SEPARATOR,
        ATTACHMENT_SIZE_SUFFIX,
        FORK_HERE,
        OPEN_SETTINGS,
        LOGIN_START_PREFIX,
        LOGIN_CANCEL,
        TOOL_KIND_SHELL,
        TOOL_KIND_EDIT,
        TOOL_KIND_FETCH,
        TOOL_KIND_SEARCH,
        TOOL_KIND_GENERIC,
        TURN_FOOTER_META_SEPARATOR,
    ];
}

/// Category a tool call belongs to. One home for the mapping so the glyph
/// classifier, the excerpt picker (`state::tool_excerpt`), and the diff-card
/// gate (`state::extract_edit_data`) never disagree about which family a
/// name belongs to. Adding a new alias here lights it up across all three
/// sites at once — the round-2 finding was that a `str_replace` classified
/// as Generic for the glyph but Edit-shaped for diff extraction, and a
/// `websearch` collapsed to Generic for the glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Shell,
    Edit,
    Fetch,
    Search,
    Generic,
}

impl ToolKind {
    pub fn classify(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "bash" | "exec" | "shell" => Self::Shell,
            "read" | "write" | "edit" | "list" | "str_replace" | "str_replace_editor"
            | "multi_edit" => Self::Edit,
            "fetch" | "webfetch" => Self::Fetch,
            "websearch" | "web_search" | "search" | "grep" | "glob" => Self::Search,
            _ => Self::Generic,
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::Shell => chrome::TOOL_KIND_SHELL,
            Self::Edit => chrome::TOOL_KIND_EDIT,
            Self::Fetch => chrome::TOOL_KIND_FETCH,
            Self::Search => chrome::TOOL_KIND_SEARCH,
            Self::Generic => chrome::TOOL_KIND_GENERIC,
        }
    }

    /// Argument keys, in priority order, that identify what THIS kind ran.
    /// `state::tool_excerpt` walks the list and takes the first that
    /// resolves to a primitive JSON value; a classified kind (non-Generic)
    /// never falls back to an arbitrary argument value, so a `str_replace`
    /// without a `path` argument does not surface its replacement text as
    /// the panel header (round-2 finding).
    pub fn excerpt_keys(self) -> &'static [&'static str] {
        match self {
            Self::Shell => &["command"],
            Self::Edit => &["path", "file_path"],
            Self::Fetch => &["url"],
            Self::Search => &["query", "pattern"],
            Self::Generic => &[],
        }
    }

    /// True when the tool NAME carries old/new text pairs that build a
    /// diff card. Narrower than `ToolKind::Edit` because `read` and `list`
    /// also classify as Edit but never carry old/new content.
    pub fn is_diff_capable(name: &str) -> bool {
        matches!(
            name.to_ascii_lowercase().as_str(),
            "edit" | "write" | "str_replace" | "str_replace_editor" | "multi_edit"
        )
    }
}

/// Legacy shorthand for `ToolKind::classify(name).glyph()` — kept as one
/// function call so the renderer stays a thin destructure over the typed
/// model.
pub fn kind_glyph_for(name: &str) -> &'static str {
    ToolKind::classify(name).glyph()
}

/// Widget-ID / debug-selector composers. Every selector the render module
/// paints is produced by one of these helpers, so the render module itself
/// has no bare `format!` fragments in its bodies — the AST fence stays
/// strict on "no literal outside debug_selector/id/aria_label/role args".
pub mod sel {
    #[cfg(feature = "smoke-test")]
    pub const NATIVE_GUARD_FORCE_TEXT_WIDTH_ENV: &str = "ZETA_GUI_NATIVE_GUARDS_FORCE_TEXT_WIDTH";
    pub const TRANSCRIPT_ROW: &str = "transcript-row";
    pub const TRANSCRIPT_COLUMN: &str = "transcript-column";
    pub const ATTACHMENT_CHIP: &str = "attachment-chip";
    pub const FORK_BUTTON_TAG: &str = "fork";
    pub const TOOL_RECEIPT_TAG: &str = "tool-receipt";
    pub const TOOL_GROUP_TAG: &str = "tool-group";
    pub const TOOL_GROUP_HIDDEN_TAG: &str = "tool-group-hidden";
    pub const ERROR_SETTINGS_TAG: &str = "error-settings";

    pub fn thinking_header(i: usize) -> String {
        format!("thinking-header-{i}")
    }
    pub fn user_row_group(i: usize) -> String {
        format!("user-row-{i}")
    }
    pub fn fork_button(i: usize) -> String {
        format!("fork-button-{i}")
    }
    pub fn message(i: usize) -> String {
        format!("message-{i}")
    }
    pub fn tool_row_group(i: usize) -> String {
        format!("tool-row-{i}")
    }
    pub fn tool_receipt(i: usize) -> String {
        format!("tool-receipt-{i}")
    }
    pub fn tool_chevron(i: usize) -> String {
        format!("tool-chevron-{i}")
    }
    /// ZETA-135 (Trait 1 — kind glyph): the small leading `$` / arrow /
    /// gear painted before the tool label. Painted on every receipt AND
    /// on the group-summary row so a scan of the transcript answers
    /// "what KIND ran" without reading the tool name.
    pub fn tool_kind_glyph(i: usize) -> String {
        format!("tool-kind-glyph-{i}")
    }
    /// ZETA-135 (Trait 2 — diff card): container that holds the two
    /// side-by-side panes below the panel header for edit receipts.
    pub fn tool_diff_card(i: usize) -> String {
        format!("tool-diff-card-{i}")
    }
    pub fn tool_diff_remove_pane(i: usize) -> String {
        format!("tool-diff-remove-{i}")
    }
    pub fn tool_diff_add_pane(i: usize) -> String {
        format!("tool-diff-add-{i}")
    }
    /// ZETA-135 (Trait 3 — turn footer): quiet strip below the LAST
    /// transcript row that names the provider · model · duration for the
    /// completed conversation. Selector is scalar because at most one
    /// footer paints per view.
    pub const TURN_FOOTER: &str = "turn-footer";
    /// Selector for the compact tool-name label. Small, dim, sits to the
    /// LEFT of the excerpt so a reader sees "which tool ran" before
    /// "what it ran" without the label stealing weight from the primary
    /// text — the ZETA-125 hierarchy inversion of the pre-fix layout.
    pub fn tool_label(i: usize) -> String {
        format!("tool-label-{i}")
    }
    pub fn tool_excerpt(i: usize) -> String {
        format!("tool-excerpt-{i}")
    }
    pub fn tool_metadata(i: usize) -> String {
        format!("tool-metadata-{i}")
    }
    pub fn tool_hover_hint(i: usize) -> String {
        format!("tool-hover-hint-{i}")
    }
    pub fn tool_output(i: usize) -> String {
        format!("tool-output-{i}")
    }
    /// Selector for the file-path / command header bar that sits at the top
    /// of the expanded receipt's inset panel (ZETA-135). Painted only when
    /// the row is expanded AND the model carries a header label.
    pub fn tool_panel_header(i: usize) -> String {
        format!("tool-panel-header-{i}")
    }
    pub fn tool_group_row(i: usize) -> String {
        format!("tool-group-{i}")
    }
    pub fn tool_group_chevron(i: usize) -> String {
        format!("tool-group-chevron-{i}")
    }
    pub fn tool_group_count(i: usize) -> String {
        format!("tool-group-count-{i}")
    }
    pub fn tool_group_preview(i: usize) -> String {
        format!("tool-group-preview-{i}")
    }
    pub fn tool_group_metadata(i: usize) -> String {
        format!("tool-group-metadata-{i}")
    }
    pub fn tool_group_focus_key(first_id: &str) -> String {
        format!("tool-group:{first_id}")
    }
    pub fn error_block(i: usize) -> String {
        format!("error-block-{i}")
    }
    pub fn error_message(i: usize) -> String {
        format!("error-message-{i}")
    }
    pub fn error_settings(i: usize) -> String {
        format!("error-settings-{i}")
    }
    pub fn error_login_prefix(i: usize) -> String {
        format!("error-login-{i}")
    }
    pub fn login_base_id(prefix: &str, provider: &str) -> String {
        format!("{prefix}-{provider}")
    }
    pub fn login_start(base: &str) -> String {
        format!("{base}-start")
    }
    pub fn login_cancel(base: &str) -> String {
        format!("{base}-cancel")
    }
    pub fn login_error(base: &str) -> String {
        format!("{base}-error")
    }
}

/// Typed model of the user text a single transcript row paints. One variant
/// per `TranscriptEntry` kind. Fields are named after the visible role each
/// string plays; the renderer destructures the struct and consumes every
/// field — an unused field trips clippy's `unused_variables` under the
/// crate's `deny(warnings)` lint, so a renderer that stops painting a field
/// fails to build.
#[derive(Debug, Clone)]
pub enum RowText<'a> {
    User(UserRowText<'a>),
    Assistant(AssistantRowText<'a>),
    Thinking(ThinkingRowText),
    Tool(ToolRowText<'a>),
    /// A collapsed run of 3+ consecutive tool receipts, painted as one row
    /// so a long run does not eat the transcript with near-identical
    /// receipts. See `ToolGroupRowText`.
    ToolGroup(ToolGroupRowText<'a>),
    /// An interior row of a collapsed tool group: paints nothing so the
    /// virtual-list index math stays 1:1 with `TranscriptEntry` indices
    /// without introducing a projection layer.
    ToolGroupHidden,
    Error(ErrorRowText<'a>),
}

/// User row: the prompt body plus optional attachment chips and the
/// hover-revealed fork action. The attachment field is tri-state on purpose:
/// text-only history rows carry `Some(vec![])` (attachment key present but
/// empty) while pre-history rows carry `None`; the old positional seam
/// collapsed both to "empty", which changed the row height when a user turn
/// restored from history — the empty container's `mt_1` margin disappeared.
#[derive(Debug, Clone)]
pub struct UserRowText<'a> {
    /// The prompt body — the row's only dynamic body string.
    pub content: &'a str,
    /// `Some(list)` when the session view has an attachment entry for this
    /// index — even if `list` is empty. The renderer paints the mt_1
    /// container whenever the value is `Some`, matching the pre-typed-seam
    /// row height for text-only history rows. `None` skips the container.
    pub attachments: Option<Vec<String>>,
    /// `Some(chrome::FORK_HERE)` when the row is fork-eligible; `None`
    /// otherwise (no button paints).
    pub fork_label: Option<&'static str>,
}

/// Assistant row: the rendered markdown plus the truncated-preview hint.
#[derive(Debug, Clone)]
pub struct AssistantRowText<'a> {
    /// The rendered markdown source — the row's only dynamic body string.
    pub source: &'a str,
    /// `Some(chrome::ASSISTANT_TRUNCATED)` when the streamed preview was
    /// truncated; `None` when the full text is on screen.
    pub truncated_hint: Option<&'static str>,
}

/// Thinking row: the generic header. The provider protocol carries no
/// display-safe summary channel, so the row never paints body text.
#[derive(Debug, Clone)]
pub struct ThinkingRowText {
    /// Exactly `THINKING_HEADER_LABEL`. A sentinel-carrying reasoning
    /// payload contributes NOTHING to this field.
    pub header: &'static str,
}

/// Tool row: the receipt line plus optional collapsed peek and expanded
/// body. Every state variant flows through the same model — running, done,
/// failed, canceled — because state is signalled by COLOR only, not text.
///
/// ZETA-125 redesign: the receipt line reads as `<tool_label> <excerpt>
/// <metadata_label>`, where the excerpt names what the tool actually ran
/// (bash command, read/edit path, fetch URL) rather than the first line of
/// its output. The label is small and dim, the excerpt is the row's primary
/// text, and the metadata sits DIRECTLY next to the excerpt end so a run of
/// receipts reads as one column instead of the pre-ZETA-125 "eight
/// identical `bash stdout:` rows with a huge right-aligned gap".
#[derive(Debug, Clone)]
pub struct ToolRowText<'a> {
    /// ZETA-135 (Trait 1 — kind glyph): leading `$` / `←` / `↗` / `⚙`
    /// painted before the tool label so a scan of the transcript column
    /// answers "what KIND ran" (shell / edit / fetch / other) before
    /// "which tool" and "with what arg". Sourced from
    /// `kind_glyph_for(tool_label)` at build time so the renderer never
    /// inspects the tool name.
    pub kind_glyph: &'static str,
    /// Small dim tool-name label (bash, read, edit, fetch, ...).
    pub tool_label: &'a str,
    /// Primary text — the excerpt of what ran (bash command's first line,
    /// read/write/edit path, fetch URL). Comes from
    /// `TranscriptEntry::Tool::excerpt` which was derived from the tool
    /// call's arguments at construction, so it is stable across streaming.
    /// `None` means the tool call had no argument to name (`tool_start` with
    /// zero primitive args) — the render layer paints the tool label alone
    /// so nothing collapses because it happens to equal the label text
    /// (`read` file, `bash` command, ZETA-134 review r2).
    pub excerpt: Option<&'a str>,
    /// `Some(...)` while the row is collapsed AND the tail carries bytes:
    /// the compact byte-count peek, painted DIRECTLY after the excerpt
    /// (laws-of-ux proximity). `None` when expanded or empty.
    pub metadata_label: Option<String>,
    /// `Some(chrome::TOOL_HOVER_HINT)` under the same condition as
    /// `metadata_label`; `None` otherwise.
    pub hover_hint: Option<&'static str>,
    /// `Some(chrome::TOOL_TAIL_OMITTED)` when the expanded body was
    /// clipped; `None` when the tail is complete or the row is collapsed.
    pub tail_omitted_hint: Option<&'static str>,
    /// The expanded output body. `Some(&tail)` when the row is expanded,
    /// `None` when collapsed.
    pub body: Option<&'a str>,
    /// Header text for the expanded inset panel (ZETA-135): the file path
    /// for read/edit/write, the command for bash, the tool name for MCP
    /// tools without a nameable argument. `Some(&excerpt)` when the row is
    /// expanded AND the excerpt is populated; `None` when the row is
    /// collapsed OR the tool call carried no nameable argument. The panel
    /// paints its header row only when this field is populated so a
    /// legitimately argument-less receipt (ZETA-134 review r2) still opens
    /// without a chromeless header bar.
    pub panel_header: Option<&'a str>,
    /// ZETA-135 (Trait 2 — diff card). `Some(...)` when the row is
    /// expanded AND `Card::edit_data` is populated: a pair of pre-
    /// numbered line lists the render layer walks to paint the two side-
    /// by-side panes. Composed here so `transcript_render` / `tool_receipts`
    /// never touch `format!` or `usize::to_string` to compose line numbers
    /// (the fence bans literals in the render module). `None` for every
    /// non-edit receipt and every collapsed row.
    pub edit_diff: Option<EditDiffText>,
}

/// Pre-composed diff pane the ZETA-135 diff card paints. Each `lines` entry
/// is a `(line_number, content)` pair; both strings are ready to hand to a
/// `.child(...)` directly.
#[derive(Debug, Clone)]
pub struct DiffPaneText {
    pub lines: Vec<(String, String)>,
}

/// The two panes of a diff card. `remove_pane` paints red-tinted on the
/// LEFT, `add_pane` paints green-tinted on the RIGHT — a stable spatial
/// mapping so the read direction is deterministic (laws-of-ux Mental
/// Model). Each pane carries its own line-numbered lines so an empty pane
/// still paints as a real column with a lone `1` gutter (the model owns
/// this shape decision, not the renderer).
#[derive(Debug, Clone)]
pub struct EditDiffText {
    pub remove_pane: DiffPaneText,
    pub add_pane: DiffPaneText,
}

/// Cap on lines the diff card renders per pane. Beyond this the pane
/// paints the leading chunk and a trailing "[N more lines]" hint so a huge
/// paste-in edit does not turn the transcript into a wall. Public so tests
/// can drive over-the-cap payloads deterministically.
pub const DIFF_PANE_LINE_CAP: usize = 40;

/// The trailing hint painted when a diff pane clipped its content. Kept as
/// a constant so tests can look it up without redefining the string.
pub const DIFF_TRUNCATED_MARKER: &str = "…";

fn build_diff_pane(text: &str) -> DiffPaneText {
    let raw: Vec<&str> = if text.is_empty() {
        Vec::new()
    } else {
        text.split('\n').collect()
    };
    let total = raw.len();
    let capped = raw.len().min(DIFF_PANE_LINE_CAP);
    let mut lines: Vec<(String, String)> = raw
        .iter()
        .take(capped)
        .enumerate()
        .map(|(index, content)| (format!("{}", index + 1), (*content).to_owned()))
        .collect();
    if total > capped {
        lines.push((
            DIFF_TRUNCATED_MARKER.to_owned(),
            format!("… {} more lines", total - capped),
        ));
    }
    if lines.is_empty() {
        // An empty side of a diff still paints one placeholder row so the
        // pane keeps a stable column shape. The content is a single space
        // so the row's height matches its counterpart.
        lines.push((format!("{}", 1), String::new()));
    }
    DiffPaneText { lines }
}

fn build_edit_diff(data: &EditData) -> EditDiffText {
    EditDiffText {
        remove_pane: build_diff_pane(&data.old_text),
        add_pane: build_diff_pane(&data.new_text),
    }
}

/// Collapsed tool-group row: a run of 3+ consecutive tool receipts that
/// paints as one summary line ("`N` tool calls · `<total>`") with the first
/// 1-2 excerpts previewed. Expanding the group reveals each row unchanged;
/// state, output, and expansion of individual receipts survive across the
/// group toggle. The typed model owns the composed count/total strings, the
/// preview excerpts, and the composed accessible label so the render layer
/// paints ONLY through this struct.
#[derive(Debug, Clone)]
pub struct ToolGroupRowText<'a> {
    /// Composed count string — e.g. "5 tool calls" (dynamic).
    pub count_label: String,
    /// Composed total-size string — e.g. "12.4KB" (dynamic); `None` when
    /// every receipt in the group finished with empty output.
    pub total_label: Option<String>,
    /// The first up-to-two excerpts, previewed on the summary row so the
    /// group's contents remain recognisable without expanding it. Each
    /// entry is a borrow from the underlying transcript entry — paired
    /// with the row-level `chrome::TOOL_GROUP_META_SEPARATOR` in the
    /// render layer so the summary reads
    /// "N tool calls · preview · preview · total".
    pub preview_excerpts: Vec<&'a str>,
    /// Chrome separator painted between the count / preview / total
    /// clusters. Sourced from `chrome::TOOL_GROUP_META_SEPARATOR` so a
    /// wording change has one home; the render layer paints it once
    /// before each preview excerpt and once before the total when either
    /// is present.
    pub separator: &'static str,
    /// Composed accessible label — the same visible text a screen reader
    /// hears when the summary row receives focus, including expansion
    /// state (`… collapsed` / `… expanded`) so keyboard-only users hear the
    /// state that changes on Enter/Space.
    pub aria_label: String,
}

/// ZETA-135 (Trait 3 — turn footer). Composed metadata strip painted
/// BELOW the last transcript row so the completed conversation reads with
/// a Peak-End cue (laws-of-ux Peak-End Rule): the final glance names the
/// provider, model, and elapsed time without a click.
///
/// Every field is `Option<String>` so a missing wire value drops the
/// token — one join with `chrome::TURN_FOOTER_META_SEPARATOR` skips
/// missing slots automatically. No field is invented on the client;
/// duration is `Some(...)` only when both `created_at` and `updated_at`
/// parsed as RFC-3339 offsets.
#[derive(Debug, Clone)]
pub struct TurnFooterText {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub duration: Option<String>,
    /// The pre-joined display string ("cc · claude-fable-5 · 6m 32s"),
    /// composed once from the fields above so the render layer paints
    /// one child.
    pub display: String,
}

impl TurnFooterText {
    pub fn is_empty(&self) -> bool {
        self.provider.is_none() && self.model.is_none() && self.duration.is_none()
    }
}

/// Build the turn-footer model from wire session metadata + status metrics.
/// `provider`/`model` are `Some(...)` when non-empty; `duration` is
/// `Some(...)` only when both timestamps parse (RFC-3339 or the wire's
/// naive-UTC seconds shape). Returns `None` when NOTHING is available so
/// the render layer never paints a footer with zero real fields.
pub fn build_turn_footer(
    provider: Option<&str>,
    model: Option<&str>,
    created_at: &str,
    updated_at: &str,
) -> Option<TurnFooterText> {
    let provider = provider.filter(|s| !s.is_empty()).map(str::to_owned);
    let model = model.filter(|s| !s.is_empty()).map(str::to_owned);
    let duration = duration_string(created_at, updated_at);
    let text = TurnFooterText {
        provider,
        model,
        duration,
        display: String::new(),
    };
    if text.is_empty() {
        return None;
    }
    let mut parts: Vec<&str> = Vec::new();
    if let Some(p) = text.provider.as_deref() {
        parts.push(p);
    }
    if let Some(m) = text.model.as_deref() {
        parts.push(m);
    }
    if let Some(d) = text.duration.as_deref() {
        parts.push(d);
    }
    let display = parts.join(chrome::TURN_FOOTER_META_SEPARATOR);
    Some(TurnFooterText { display, ..text })
}

/// Parse a wire timestamp string to unix seconds. The server emits
/// RFC-3339 with an offset; some legacy sessions omit the offset entirely
/// (naive UTC). Offset-aware branch first so timestamps recorded in
/// different zones subtract to the true elapsed span; naive branch second
/// for the legacy shape. The hand-rolled state machine this replaced
/// ignored offsets and used a year/4 leap-year approximation that also
/// dropped the Feb-29 adjustment (round-2 finding: Feb 28 -> Mar 1 2028
/// reported 24h instead of 48h).
fn parse_epoch_seconds(text: &str) -> Option<i64> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(text) {
        return Some(dt.timestamp());
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(text, fmt) {
            return Some(dt.and_utc().timestamp());
        }
    }
    None
}

fn duration_string(created_at: &str, updated_at: &str) -> Option<String> {
    let start = parse_epoch_seconds(created_at)?;
    let end = parse_epoch_seconds(updated_at)?;
    let elapsed = (end - start).max(0);
    if elapsed <= 0 {
        return None;
    }
    let minutes = elapsed / 60;
    let seconds = elapsed % 60;
    let hours = minutes / 60;
    let minutes = minutes % 60;
    if hours > 0 {
        Some(format!("{}h {}m {}s", hours, minutes, seconds))
    } else if minutes > 0 {
        Some(format!("{}m {}s", minutes, seconds))
    } else {
        Some(format!("{}s", seconds))
    }
}

/// Error row: the header, the message body, and the optional settings
/// recovery button.
#[derive(Debug, Clone)]
pub struct ErrorRowText<'a> {
    /// Exactly `ERROR_HEADER_LABEL`.
    pub header: &'static str,
    /// The provider's error text.
    pub message: &'a str,
    /// `Some(chrome::OPEN_SETTINGS)` when the recovery button should
    /// paint (settings_action AND session_view.available); `None`
    /// otherwise.
    pub settings_action_label: Option<&'static str>,
}

/// Typed model of the login-row surface. Rendered from four call sites in
/// the app — settings overlay, inline error recovery, in-progress banner,
/// first-conversation prompt — every one of which paints the same six
/// visible strings. Resolving the provider label BEFORE render kills the
/// "raw provider text handed to the renderer" bypass the r1 review found.
#[derive(Debug, Clone)]
pub struct LoginRowText {
    /// Outer widget id — used for `.id(...)` and as the base for the
    /// button selectors below. Composed once in `build_login`, never in
    /// the render body.
    pub outer_id: String,
    /// Provider header label — dynamic ("Claude", "ChatGPT", ...).
    pub header_label: String,
    /// Provider slug — carried alongside the label so the click callback
    /// can identify the provider without seeing the visible text. Not
    /// painted as visible text — it is the opaque handle passed to
    /// `start_login`/`cancel_login`.
    pub provider_slug: String,
    /// Start button model — id + composed label + disabled flag.
    pub start: LoginActionText,
    /// Cancel button model — present only while the login is busy.
    pub cancel: Option<LoginActionText>,
    /// Status text — one of a fixed set of `&'static str` values sourced
    /// from `LoginProvider::status()`.
    pub status_text: &'static str,
    /// Error alert model — present when the last attempt failed.
    pub error: Option<LoginErrorText>,
}

/// One button on the login row.
#[derive(Debug, Clone)]
pub struct LoginActionText {
    /// Composed widget id (also used for the debug selector).
    pub id: String,
    /// Composed visible label — always begins with `chrome::LOGIN_START_PREFIX`
    /// for the start action, or equals `chrome::LOGIN_CANCEL` for cancel.
    pub label: String,
    /// Whether the button paints as disabled.
    pub disabled: bool,
}

/// Error alert on the login row.
#[derive(Debug, Clone)]
pub struct LoginErrorText {
    /// Composed widget id — `format!("{outer_id}-error")`.
    pub id: String,
    /// The provider's error text.
    pub message: String,
}

impl<'a> RowText<'a> {
    /// Iterate every visible string the row paints — used by guard tests
    /// to sweep for stray markers or sentinel leaks. Chrome-only fields
    /// (headers, hints, action labels) are yielded alongside content
    /// fields so a marker anywhere in the row's paint set fails the
    /// sweep.
    pub fn visible_strings(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        match self {
            Self::User(text) => {
                let UserRowText {
                    content,
                    attachments,
                    fork_label,
                } = text;
                out.push(content);
                if let Some(list) = attachments {
                    out.extend(list.iter().map(String::as_str));
                }
                out.extend(fork_label.iter().copied());
            }
            Self::Assistant(text) => {
                let AssistantRowText {
                    source,
                    truncated_hint,
                } = text;
                out.push(source);
                out.extend(truncated_hint.iter().copied());
            }
            Self::Thinking(text) => {
                let ThinkingRowText { header } = text;
                out.push(header);
            }
            Self::Tool(text) => {
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
                out.push(kind_glyph);
                out.push(tool_label);
                out.extend(excerpt);
                out.extend(metadata_label.as_deref());
                out.extend(hover_hint.iter().copied());
                out.extend(tail_omitted_hint.iter().copied());
                out.extend(body.iter().copied());
                out.extend(panel_header.iter().copied());
                if let Some(diff) = edit_diff {
                    for pane in [&diff.remove_pane, &diff.add_pane] {
                        for (number, content) in &pane.lines {
                            out.push(number.as_str());
                            out.push(content.as_str());
                        }
                    }
                }
            }
            Self::ToolGroup(text) => {
                let ToolGroupRowText {
                    count_label,
                    total_label,
                    preview_excerpts,
                    separator,
                    aria_label,
                } = text;
                out.push(count_label.as_str());
                out.extend(total_label.as_deref());
                out.extend(preview_excerpts.iter().copied());
                out.push(separator);
                out.push(aria_label.as_str());
            }
            Self::ToolGroupHidden => {}
            Self::Error(text) => {
                let ErrorRowText {
                    header,
                    message,
                    settings_action_label,
                } = text;
                out.push(header);
                out.push(message);
                out.extend(settings_action_label.iter().copied());
            }
        }
        out
    }
}

impl LoginRowText {
    /// Sentinel/marker sweep for guard tests — yields every visible string
    /// a login row paints.
    pub fn visible_strings(&self) -> Vec<&str> {
        let mut out: Vec<&str> = vec![
            self.header_label.as_str(),
            self.start.label.as_str(),
            self.status_text,
        ];
        if let Some(cancel) = &self.cancel {
            out.push(cancel.label.as_str());
        }
        if let Some(error) = &self.error {
            out.push(error.message.as_str());
        }
        out
    }
}

/// Build the row-text model for one transcript row. Everything the
/// renderer will paint as user-visible text is computed here — never in
/// the render body.
pub fn build<'a>(
    entry: &'a TranscriptEntry,
    index: usize,
    session_view: &SessionView,
    settings_action_available: bool,
) -> RowText<'a> {
    match entry {
        TranscriptEntry::User(text) => RowText::User(UserRowText {
            content: text,
            attachments: session_view.attachments.get(&index).map(|list| {
                list.iter()
                    .map(|(name, size)| format_attachment(name, *size))
                    .collect()
            }),
            fork_label: fork_label_for(index, session_view),
        }),
        TranscriptEntry::Assistant(doc) => RowText::Assistant(AssistantRowText {
            source: doc.source.as_ref(),
            truncated_hint: doc.preview_truncated.then_some(chrome::ASSISTANT_TRUNCATED),
        }),
        TranscriptEntry::Thinking => RowText::Thinking(ThinkingRowText {
            header: THINKING_HEADER_LABEL,
        }),
        TranscriptEntry::Tool {
            name,
            excerpt,
            card,
            ..
        } => {
            // Metadata reads cumulative bytes flowed through the tail
            // (`bytes_seen`), not just the on-screen retained bytes. That
            // keeps the receipt's size label consistent with a group
            // summary that sums the same field across its members — the
            // r2 review flagged the mismatch when tails truncated.
            let output_size = card.tail.bytes_seen;
            let has_output = output_size > 0;
            let collapsed_with_output = !card.expanded && has_output;
            // `excerpt: Option<&str>` carries the missing-argument state
            // explicitly (ZETA-134 review r2): an argument-less `tool_start`
            // stores `None` at construction, so the render layer paints the
            // tool label alone — no primary text. A file literally named
            // `read` or a bash command named `bash` still stores its real
            // string in `Some(...)` and paints, because the previous
            // string-equality dedupe collapsed those legitimate values into
            // the label.
            RowText::Tool(ToolRowText {
                kind_glyph: kind_glyph_for(name),
                tool_label: name,
                excerpt: excerpt.as_deref(),
                metadata_label: collapsed_with_output.then(|| format_output_size(output_size)),
                hover_hint: collapsed_with_output.then_some(chrome::TOOL_HOVER_HINT),
                tail_omitted_hint: (card.expanded && card.tail.truncated)
                    .then_some(chrome::TOOL_TAIL_OMITTED),
                body: card.expanded.then_some(card.tail.text.as_str()),
                // Header ONLY when the receipt is expanded AND we have a
                // nameable excerpt. A `None` excerpt (argument-less tool
                // start) opens without a chromeless header bar — the tool
                // label alone carries the receipt's identity. See
                // `ZETA-134` review r2 for why `excerpt: None` must not
                // collapse into any painted primary text.
                panel_header: card.expanded.then_some(excerpt.as_deref()).flatten(),
                // ZETA-135 (Trait 2): the diff card only paints when the
                // receipt is expanded AND the tool call carried typed
                // edit data. Every other receipt keeps the pre-r2
                // body-only expanded shape.
                edit_diff: card
                    .expanded
                    .then(|| card.edit_data.as_ref().map(build_edit_diff))
                    .flatten(),
            })
        }
        TranscriptEntry::Error {
            message,
            settings_action,
            ..
        } => RowText::Error(ErrorRowText {
            header: ERROR_HEADER_LABEL,
            message,
            settings_action_label: (*settings_action && settings_action_available)
                .then_some(chrome::OPEN_SETTINGS),
        }),
    }
}

/// Build the login-row model from a provider record and its outer-id
/// prefix. The provider label is resolved here — the renderer receives a
/// composed `header_label`/`start.label` and never touches `provider.label()`
/// again. This is the fix for the r1 review finding at main.rs:1564: the
/// login renderer bypassed the typed model by reading `provider.label()`
/// directly on every paint.
pub fn build_login(
    provider: &LoginProvider,
    prefix: &str,
    connection: &ConnectionState,
) -> LoginRowText {
    let outer_id = sel::login_base_id(prefix, &provider.provider);
    let label = provider.label().to_owned();
    let start_id = sel::login_start(&outer_id);
    let cancel_id = sel::login_cancel(&outer_id);
    let error_id = sel::login_error(&outer_id);
    let start_disabled =
        provider.progress.busy() || !matches!(connection, ConnectionState::Connected);
    let cancel = provider.progress.busy().then(|| LoginActionText {
        id: cancel_id,
        label: chrome::LOGIN_CANCEL.to_owned(),
        disabled: provider.progress == LoginProgress::Cancelling,
    });
    let error = match &provider.progress {
        LoginProgress::Failed { error } => Some(LoginErrorText {
            id: error_id,
            message: error.message.clone(),
        }),
        _ => None,
    };
    LoginRowText {
        outer_id,
        header_label: label.clone(),
        provider_slug: provider.provider.clone(),
        start: LoginActionText {
            id: start_id,
            label: format!("{}{}", chrome::LOGIN_START_PREFIX, label),
            disabled: start_disabled,
        },
        cancel,
        status_text: provider.status(),
        error,
    }
}

/// Compose the summary text for a collapsed tool-group row. Called by the
/// render layer once per group; the returned `ToolGroupRowText` carries the
/// composed count/total strings, the preview excerpts borrowed from the
/// underlying entries, and the accessible label (including expansion
/// state).
///
/// `preview_count` is capped at 2 by the caller per the contract; keeping
/// the cap in the render layer makes tests grep-able and avoids a
/// row_text-side constant that would need its own home in `chrome`.
pub fn build_tool_group<'a>(
    excerpts: &[&'a str],
    total_output_bytes: usize,
    preview_count: usize,
    expanded: bool,
) -> ToolGroupRowText<'a> {
    let count = excerpts.len();
    let unit = if count == 1 {
        chrome::TOOL_GROUP_CALL
    } else {
        chrome::TOOL_GROUP_CALLS
    };
    let count_label = format!("{count}{unit}");
    let total_label = (total_output_bytes > 0).then(|| format_output_size(total_output_bytes));
    let take = preview_count.min(2).min(excerpts.len());
    let preview_excerpts: Vec<&'a str> = excerpts.iter().take(take).copied().collect();
    // The separator glyph rides on the row unconditionally so the
    // render layer never has to reason about a missing value; it only
    // paints the separator when it also paints a preview or the total,
    // so an orphan separator cannot end up on a chromeless row.
    let separator = chrome::TOOL_GROUP_META_SEPARATOR;
    let state_suffix = if expanded {
        chrome::TOOL_GROUP_ARIA_EXPANDED
    } else {
        chrome::TOOL_GROUP_ARIA_COLLAPSED
    };
    let mut aria_label = String::new();
    aria_label.push_str(&count_label);
    for preview in &preview_excerpts {
        aria_label.push_str(chrome::TOOL_GROUP_META_SEPARATOR);
        aria_label.push_str(preview);
    }
    if let Some(total) = &total_label {
        aria_label.push_str(chrome::TOOL_GROUP_META_SEPARATOR);
        aria_label.push_str(total);
    }
    aria_label.push_str(state_suffix);
    ToolGroupRowText {
        count_label,
        total_label,
        preview_excerpts,
        separator,
        aria_label,
    }
}

fn fork_label_for(index: usize, session_view: &SessionView) -> Option<&'static str> {
    if !session_view.available {
        return None;
    }
    session_view
        .message_ids
        .get(&index)
        .map(|_| chrome::FORK_HERE)
}

/// One attachment chip label: `"{name}{sep}{size}{suffix}"`. The renderer
/// draws these verbatim — every visible piece comes from `chrome`.
fn format_attachment(name: &str, size: usize) -> String {
    format!(
        "{name}{sep}{size}{suffix}",
        sep = chrome::ATTACHMENT_SIZE_SEPARATOR,
        suffix = chrome::ATTACHMENT_SIZE_SUFFIX,
    )
}

/// Compact byte-size label for the collapsed tool-row output peek. Kept
/// short so the receipt still fits on one line at the wiki 1024px column.
/// Unit suffixes come from `chrome` so wording changes have one home.
fn format_output_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}{}", chrome::OUTPUT_SIZE_UNIT_B)
    } else if bytes < 1024 * 1024 {
        format!(
            "{:.1}{}",
            bytes as f64 / 1024.0,
            chrome::OUTPUT_SIZE_UNIT_KB
        )
    } else {
        format!(
            "{:.1}{}",
            bytes as f64 / (1024.0 * 1024.0),
            chrome::OUTPUT_SIZE_UNIT_MB
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::Card;
    use crate::login::{LoginError, LoginProgress, LoginProvider};
    use crate::state::ToolReceiptKey;

    fn empty_view() -> SessionView {
        SessionView::default()
    }

    #[test]
    fn thinking_row_header_never_carries_body_text() {
        let entry = TranscriptEntry::Thinking;
        let row = build(&entry, 0, &empty_view(), true);
        let RowText::Thinking(text) = row else {
            panic!("thinking entry must build a Thinking row")
        };
        assert_eq!(text.header, THINKING_HEADER_LABEL);
    }

    #[test]
    fn sentinel_in_thinking_source_never_lands_on_the_row_model() {
        // The Thinking entry carries no body — a sentinel-shaped provider
        // payload cannot flow into any RowText field. This mirrors the
        // ZETA-107 privacy invariant.
        let entry = TranscriptEntry::Thinking;
        let row = build(&entry, 0, &empty_view(), true);
        for text in row.visible_strings() {
            assert!(
                !text.contains("SENTINEL"),
                "thinking row text contained sentinel: {text:?}"
            );
        }
    }

    #[test]
    fn tool_row_never_emits_state_markers_on_the_visible_seam() {
        // Contract line 83: state is COLOR only. No textual
        // "[working]/[done]/[failed]/[canceled]" marker may reach a
        // visible field on any tool row, in any state combination.
        let markers = ["[working]", "[done]", "[failed]", "[canceled]"];
        for (complete, error, canceled) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, false, true),
        ] {
            let entry = TranscriptEntry::Tool {
                key: ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "id".into(),
                },
                name: "bash".into(),
                excerpt: Some("echo hello".into()),
                summary: "echo".into(),
                complete,
                error,
                canceled,
                card: Card {
                    expanded: true,
                    ..Default::default()
                },
            };
            let row = build(&entry, 0, &empty_view(), true);
            for text in row.visible_strings() {
                for marker in markers {
                    assert!(
                        !text.contains(marker),
                        "tool row visible text carried {marker} for (complete={complete}, error={error}, canceled={canceled}): {text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn chrome_literals_never_carry_state_markers() {
        // Adding a new chrome literal that happens to name a state
        // marker regresses the contract. Exhaustive sweep over the ALL
        // set (grep proof that ALL contains every chrome constant lives
        // in the fence's chrome-coverage check).
        for literal in chrome::ALL {
            for marker in ["[working]", "[done]", "[failed]", "[canceled]"] {
                assert!(
                    !literal.contains(marker),
                    "chrome literal {literal:?} carried state marker {marker}"
                );
            }
        }
    }

    #[test]
    fn user_row_model_carries_attachment_labels_and_optional_fork() {
        use crate::state::AppState;
        let mut state = AppState::default();
        state.session_view.available = true;
        state.session_view.message_ids.insert(0, "msg-1".into());
        state
            .session_view
            .attachments
            .insert(0, vec![("hero.png".into(), 4096)]);
        let entry = TranscriptEntry::User("hello".into());
        let row = build(&entry, 0, &state.session_view, true);
        let RowText::User(text) = row else {
            panic!("user entry must build a User row")
        };
        assert_eq!(text.content, "hello");
        assert_eq!(
            text.attachments,
            Some(vec!["hero.png · 4096 bytes".to_owned()])
        );
        assert_eq!(text.fork_label, Some(chrome::FORK_HERE));
    }

    #[test]
    fn user_row_without_message_id_hides_the_fork_label() {
        let entry = TranscriptEntry::User("hello".into());
        let row = build(&entry, 0, &empty_view(), true);
        let RowText::User(text) = row else {
            panic!("user entry must build a User row")
        };
        assert_eq!(text.fork_label, None);
    }

    #[test]
    fn user_row_distinguishes_absent_key_from_present_empty_attachments() {
        // Regression guard for the r1 review finding: text-only history
        // rows store `Some(vec![])` in `session_view.attachments`, and the
        // pre-typed-seam renderer painted the mt_1 container in that case.
        // The typed model must preserve the tri-state so the renderer can
        // keep those rows' height identical to the pre-seam behaviour.
        use crate::state::AppState;
        let entry = TranscriptEntry::User("hi".into());
        // Absent key: renderer must skip the container.
        let no_key = build(&entry, 0, &AppState::default().session_view, true);
        let RowText::User(text) = no_key else {
            unreachable!()
        };
        assert!(text.attachments.is_none(), "absent key must map to None");
        // Present empty: renderer must paint the (empty) container so the
        // row height matches text-only history rows before the refactor.
        let mut state = AppState::default();
        state.session_view.attachments.insert(0, Vec::new());
        let present_empty = build(&entry, 0, &state.session_view, true);
        let RowText::User(text) = present_empty else {
            unreachable!()
        };
        assert_eq!(
            text.attachments,
            Some(Vec::<String>::new()),
            "present-empty key must map to Some(empty)",
        );
    }

    #[test]
    fn tool_output_size_label_scales_across_unit_boundaries() {
        for (bytes, expected) in [
            (0usize, None),
            (900, Some("900B".to_owned())),
            (2048, Some("2.0KB".to_owned())),
            (3 * 1024 * 1024, Some("3.0MB".to_owned())),
        ] {
            let mut card = Card {
                expanded: false,
                ..Default::default()
            };
            card.tail.text = "x".repeat(bytes);
            // Metadata reads `bytes_seen` (the cumulative-output field
            // that survives tail truncation), so populate it to match
            // the on-screen text length. The label formatter picks its
            // unit off `bytes_seen`, not `text.len()`.
            card.tail.bytes_seen = bytes;
            let entry = TranscriptEntry::Tool {
                key: ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "id".into(),
                },
                name: "bash".into(),
                excerpt: Some("echo hi".into()),
                summary: "".into(),
                complete: true,
                error: false,
                canceled: false,
                card,
            };
            let row = build(&entry, 0, &empty_view(), true);
            let RowText::Tool(text) = row else {
                panic!("tool entry must build a Tool row")
            };
            assert_eq!(text.metadata_label, expected, "at {bytes}B");
        }
    }

    fn provider(progress: LoginProgress) -> LoginProvider {
        LoginProvider {
            provider: "claude".into(),
            credentials_present: false,
            progress,
        }
    }

    #[test]
    fn login_model_composes_start_label_from_chrome_prefix_and_provider_label() {
        let text = build_login(
            &provider(LoginProgress::Idle),
            "settings-login",
            &ConnectionState::Connected,
        );
        assert_eq!(text.header_label, "Claude");
        assert_eq!(text.start.label, "Log in with Claude");
        assert!(!text.start.disabled);
        assert_eq!(text.status_text, "Not logged in");
        assert!(text.cancel.is_none());
        assert!(text.error.is_none());
        // The composed IDs derive from the prefix + provider slug.
        assert_eq!(text.outer_id, "settings-login-claude");
        assert_eq!(text.start.id, "settings-login-claude-start");
    }

    #[test]
    fn login_model_disables_start_when_disconnected_and_shows_cancel_when_busy() {
        let text = build_login(
            &provider(LoginProgress::Idle),
            "settings-login",
            &ConnectionState::Reconnecting,
        );
        assert!(text.start.disabled, "disconnected must disable start");
        let busy = build_login(
            &provider(LoginProgress::Starting),
            "settings-login",
            &ConnectionState::Connected,
        );
        let cancel = busy.cancel.expect("busy provider paints a cancel button");
        assert_eq!(cancel.label, "Cancel");
        assert!(!cancel.disabled);
    }

    #[test]
    fn login_model_surfaces_the_failure_message() {
        let mut prov = provider(LoginProgress::Failed {
            error: LoginError {
                code: "boom".into(),
                message: "no browser".into(),
            },
        });
        prov.credentials_present = false;
        let text = build_login(&prov, "first-login", &ConnectionState::Connected);
        let err = text.error.expect("failed progress must produce an error");
        assert_eq!(err.message, "no browser");
        assert_eq!(err.id, "first-login-claude-error");
    }
}
