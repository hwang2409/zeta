//! Composer-attached slash-command menu.
//!
//! State only — rendering lives beside the composer so it can share theme
//! tokens with the rest of the composer chrome. The menu drives its filter
//! from the composer text (never a private buffer) so what the user sees in
//! the input matches what actually dispatches on Enter.

use zeta_gui::client::SlashCommandInfo;

pub const MAX_ROWS: usize = 8;

/// Copy strings the presenter renders. Kept beside the menu state so a
/// wording change lands here — the composer view references these names
/// rather than bare literals. The transcript-row model owns transcript
/// copy under the ZETA-109 fence; the composer-menu strip is view chrome
/// and owns its own strings here.
pub mod copy {
    pub const EMPTY: &str = "No matching commands";
    pub const CLIENT_ONLY_TEMPLATE: &str = "runs client-side; not yet wired in the desktop app";
    pub const UNKNOWN_TEMPLATE: &str = "no such command";
    pub const UNAVAILABLE: &str =
        "slash commands are unavailable on this server; upgrade zeta serve to use them";

    pub fn client_only(name: &str) -> String {
        format!("/{name} {CLIENT_ONLY_TEMPLATE}")
    }

    pub fn unknown(name: &str) -> String {
        format!("{UNKNOWN_TEMPLATE}: /{name}")
    }
}

/// One page of visible rows around the selection, plus the selection's
/// offset within that page. The presenter never touches raw indices — the
/// window helper keeps highlight and rendered rows in agreement even when
/// the filter has more matches than `MAX_ROWS`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisibleRows {
    pub rows: Vec<SlashCommandInfo>,
    pub selected: usize,
}

/// The menu's runtime state. Owned by the view; every mutation goes through
/// one of the helpers here so the invariants hold in tests.
#[derive(Debug, Default, Clone)]
pub struct SlashMenu {
    pub commands: Vec<SlashCommandInfo>,
    pub open: bool,
    pub selected: usize,
    pub filter: String,
    /// Notice returned by the last `slash_run` — the composer paints it as an
    /// inline receipt beside the menu so a rejected command explains itself.
    pub notice: Option<String>,
}

impl SlashMenu {
    /// Update the menu's open state and filter from the current composer
    /// value. The menu opens only when the value begins with `/` and stays
    /// closed for anything else so ordinary chat text never intercepts input.
    pub fn sync(&mut self, composer_text: &str) {
        match leading_slash_token(composer_text) {
            Some(filter) => {
                let previous_open = self.open;
                self.open = true;
                self.filter = filter.to_owned();
                if !previous_open {
                    self.selected = 0;
                    self.notice = None;
                }
                let matches = self.filtered().count();
                if matches == 0 {
                    self.selected = 0;
                } else if self.selected >= matches {
                    self.selected = matches - 1;
                }
            }
            None => self.dismiss(),
        }
    }

    pub fn dismiss(&mut self) {
        self.open = false;
        self.filter.clear();
        self.selected = 0;
    }

    pub fn set_commands(&mut self, commands: Vec<SlashCommandInfo>) {
        self.commands = commands;
    }

    pub fn set_notice(&mut self, notice: String) {
        self.notice = Some(notice);
    }

    pub fn clear_notice(&mut self) {
        self.notice = None;
    }

    /// Move the selection one row down, saturating at the last match.
    pub fn select_next(&mut self) {
        let count = self.filtered().count();
        if count == 0 {
            self.selected = 0;
            return;
        }
        self.selected = (self.selected + 1).min(count - 1);
    }

    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    /// The currently highlighted command, or `None` when no matches remain.
    pub fn current(&self) -> Option<&SlashCommandInfo> {
        self.filtered().nth(self.selected)
    }

    pub fn filtered(&self) -> impl Iterator<Item = &SlashCommandInfo> + '_ {
        let needle = self.filter.to_lowercase();
        self.commands
            .iter()
            .filter(move |command| command.name.to_lowercase().starts_with(&needle))
    }

    /// A page of visible rows around the selection, saturating at the ends.
    /// The window slides once the highlighted row would fall past the last
    /// rendered slot so Enter can never target a hidden match.
    pub fn visible_window(&self) -> VisibleRows {
        let matches: Vec<SlashCommandInfo> = self.filtered().cloned().collect();
        if matches.is_empty() {
            return VisibleRows::default();
        }
        let selected = self.selected.min(matches.len() - 1);
        let start = if selected < MAX_ROWS {
            0
        } else {
            selected + 1 - MAX_ROWS
        };
        let end = (start + MAX_ROWS).min(matches.len());
        VisibleRows {
            rows: matches[start..end].to_vec(),
            selected: selected - start,
        }
    }
}

/// True when the composer draft is a slash invocation — a leading `/` that
/// is not the `//` literal escape. Multiline drafts still classify: the
/// shared dispatcher parses the first line, so any `/`-prefixed value must
/// route through `slash_run` and never leak to the model as chat.
pub fn is_slash_draft(value: &str) -> bool {
    let trimmed = value.trim_start();
    trimmed.starts_with('/') && !trimmed.starts_with("//")
}

/// Turn a `//`-prefixed draft into its literal `/`-prefixed chat form so
/// the model receives what the user typed after the escape, matching the
/// shared `input_for_model` contract used by the TUI submission pipeline.
/// A draft that does not begin with `//` passes through unchanged.
pub fn unescape_literal_slash(value: String) -> String {
    if value.starts_with("//") {
        value[1..].to_owned()
    } else {
        value
    }
}

/// Return the argument-free command token that follows a leading `/`, or
/// `None` when the value is not a slash invocation. A trailing space means
/// the user has moved past the command name; the menu still tracks the token
/// but arrow keys keep working because the filter narrows to that one match.
pub fn leading_slash_token(value: &str) -> Option<&str> {
    let trimmed = value.trim_start();
    if !trimmed.starts_with('/') || trimmed.starts_with("//") {
        return None;
    }
    if trimmed.contains('\n') {
        return None;
    }
    let after = &trimmed[1..];
    let end = after
        .find(|ch: char| ch.is_whitespace())
        .unwrap_or(after.len());
    Some(&after[..end])
}

/// Assemble the composer's next value for a selected command. Keeps any
/// trailing argument tail the user already typed so `/hi foo` stays complete
/// when the menu snaps to the `hi` row.
pub fn compose_selection(current: &str, command: &SlashCommandInfo) -> String {
    let trimmed = current.trim_start();
    let after = &trimmed[1..];
    let tail_start = after
        .find(|ch: char| ch.is_whitespace())
        .unwrap_or(after.len());
    let tail = &after[tail_start..];
    if tail.is_empty() {
        format!("/{} ", command.name)
    } else {
        format!("/{}{}", command.name, tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeta_gui::client::SlashCommandInfo;

    fn info(name: &str) -> SlashCommandInfo {
        SlashCommandInfo {
            name: name.to_owned(),
            description: format!("{name} description"),
            kind: "builtin".to_owned(),
            source: "builtin".to_owned(),
            client_only: false,
            unavailable: None,
        }
    }

    #[test]
    fn leading_slash_token_extracts_the_command_word() {
        assert_eq!(leading_slash_token("/status"), Some("status"));
        assert_eq!(leading_slash_token("/status ext"), Some("status"));
        assert_eq!(leading_slash_token("  /st"), Some("st"));
        assert_eq!(leading_slash_token(""), None);
        assert_eq!(leading_slash_token("hello"), None);
        // The `//` escape stays a literal for the composer; never open the menu.
        assert_eq!(leading_slash_token("//literal"), None);
        // A newline finishes the composer draft — the menu closes rather than
        // filtering across lines that would confuse the dispatcher.
        assert_eq!(leading_slash_token("/status\ntail"), None);
    }

    #[test]
    fn is_slash_draft_covers_multiline_and_rejects_literal_escape() {
        // Menu filtering closes on newline (see the token test above), but a
        // submit-time guard MUST still classify a multiline draft that starts
        // with `/` as a slash invocation so it never reaches the model as
        // chat. The dispatcher itself parses the first line.
        assert!(is_slash_draft("/status"));
        assert!(is_slash_draft("  /status"));
        assert!(is_slash_draft("/status\nfoo"));
        assert!(!is_slash_draft(""));
        assert!(!is_slash_draft("hello"));
        assert!(!is_slash_draft("//literal"));
    }

    #[test]
    fn sync_opens_filters_and_dismisses() {
        let mut menu = SlashMenu::default();
        menu.set_commands(vec![info("status"), info("stop"), info("model")]);
        menu.sync("/st");
        assert!(menu.open);
        assert_eq!(menu.filter, "st");
        assert_eq!(menu.visible_window().rows.len(), 2);
        menu.select_next();
        assert_eq!(menu.current().unwrap().name, "stop");
        menu.select_next();
        // Selection saturates at the last visible row.
        assert_eq!(menu.current().unwrap().name, "stop");
        menu.select_prev();
        assert_eq!(menu.current().unwrap().name, "status");
        // Typing a non-slash value dismisses the menu; the invariant that
        // guarantees the composer stops intercepting arrow keys.
        menu.sync("hello");
        assert!(!menu.open);
        assert_eq!(menu.filter, "");
    }

    #[test]
    fn visible_window_scrolls_with_the_selection() {
        // Build MAX_ROWS + 3 matches. Selection past MAX_ROWS-1 must slide
        // the window so the highlight is always in the rendered page, or a
        // hidden row could win an Enter press.
        let mut menu = SlashMenu::default();
        let commands: Vec<_> = (0..MAX_ROWS + 3)
            .map(|i| info(&format!("cmd{i:02}")))
            .collect();
        menu.set_commands(commands.clone());
        menu.sync("/cmd");
        assert_eq!(menu.visible_window().rows.len(), MAX_ROWS);
        assert_eq!(menu.visible_window().rows[0].name, "cmd00");
        for _ in 0..MAX_ROWS + 2 {
            menu.select_next();
        }
        let window = menu.visible_window();
        assert_eq!(window.rows.len(), MAX_ROWS);
        assert_eq!(
            window.rows.last().unwrap().name,
            commands.last().unwrap().name
        );
        assert_eq!(window.selected, MAX_ROWS - 1);
        assert_eq!(
            window.rows[window.selected].name,
            commands.last().unwrap().name
        );
    }

    #[test]
    fn compose_selection_preserves_the_argument_tail() {
        let command = info("model");
        assert_eq!(compose_selection("/m", &command), "/model ");
        assert_eq!(compose_selection("/m gpt-5", &command), "/model gpt-5");
    }
}
