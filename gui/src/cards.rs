//! Bounded output and disclosure state shared by tool and delegated receipts.
pub const TAIL_LINES: usize = 20;
const TAIL_CHARS: usize = 16_000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputTail {
    pub text: String,
    pub truncated: bool,
    // Cumulative bytes flowed through this tail since the receipt began.
    // Grows unbounded across truncations so the row's size label reads as
    // "total output produced" rather than "bytes currently retained on
    // screen" — a group summary that sums `bytes_seen` across its members
    // stays consistent with each row's own label even after the 20-line /
    // 16k-char tail budgets clipped some of the bytes off screen.
    pub bytes_seen: usize,
}

impl OutputTail {
    pub fn append(&mut self, text: &str) {
        self.bytes_seen = self.bytes_seen.saturating_add(text.len());
        self.text.push_str(text);
        self.enforce_bounds();
    }

    /// Overwrite the visible tail text WITHOUT touching `bytes_seen`. Used
    /// by the `ToolEnd` handler for every streamed tool final payload.
    pub fn replace_visible(&mut self, text: &str) {
        self.text.clear();
        self.truncated = false;
        self.text.push_str(text);
        self.enforce_bounds();
    }

    fn enforce_bounds(&mut self) {
        let lines: Vec<_> = self.text.split_inclusive('\n').collect();
        if lines.len() > TAIL_LINES {
            self.text = lines[lines.len() - TAIL_LINES..].concat();
            self.truncated = true;
        }
        if let Some((offset, _)) = self.text.char_indices().rev().nth(TAIL_CHARS - 1) {
            if offset > 0 {
                self.text.drain(..offset);
                self.truncated = true;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Card {
    pub expanded: bool,
    pub tail: OutputTail,
    pub agent_label: Option<String>,
    pub child_instance_id: Option<String>,
    // The server turn this receipt was created in. Tool grouping never
    // spans a turn boundary, so a run of receipts across two turns paints
    // as two separate groups even when the receipts sit adjacent in the
    // transcript. See `AppState::tool_group_position`. Default is 0 which
    // matches the pre-tracking behaviour: single-turn tests all read as
    // one turn.
    pub turn: u64,
    // True once a `ServerEvent::ToolOutput` has fed the tail. The
    // `ToolEnd` handler reads this to skip re-appending the final result
    // for tools whose end payload repeats the streamed stdout (e.g. bash
    // per src/zeta/tools/bash/__init__.py:202). Without the flag, `bytes_seen` and
    // the on-screen tail double-count the same bytes. Agent-style tools
    // that only use `ToolEnd` (no `ToolOutput`) keep their content path
    // unchanged because `streamed` stays false.
    pub streamed: bool,
    // ZETA-135 (Trait 2 — diff card). Typed old_text/new_text pair
    // extracted from an edit tool call's arguments at construction time
    // (`state::extract_edit_data`). Populated for `edit`/`write` shaped
    // tool names when the arguments carry `old_string`+`new_string` (or
    // the `old_str`/`new_str` shorthand); `None` for every other tool
    // AND for edit calls that arrived without a diff pair (a bare
    // `write` that only names a path, say). The render layer paints a
    // side-by-side diff card from this field when it is `Some(...)`;
    // when `None`, the expanded panel keeps the pre-r2 body-only shape.
    pub edit_data: Option<EditData>,
}

/// One primitive edit — the old chunk about to be replaced and the new
/// chunk that replaces it. Consumed by the ZETA-135 diff card. Stored on
/// `Card` so the render layer never has to touch the raw
/// `tool_call.arguments` bag again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditData {
    pub old_text: String,
    pub new_text: String,
}

impl Card {
    pub fn toggle(&mut self) {
        self.expanded = !self.expanded;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tails_keep_last_twenty_lines_across_partial_chunks() {
        let mut tail = OutputTail::default();
        tail.append("first");
        tail.append(" line\n");
        for n in 1..20 {
            tail.append(&format!("{n}\n"));
        }
        assert!(!tail.truncated);
        tail.append("last");
        assert!(tail.truncated);
        assert_eq!(tail.text.lines().count(), TAIL_LINES);
        assert!(tail.text.starts_with("1\n"));
        tail.append(" line\n");
        assert!(tail.text.ends_with("last line\n"));
        assert_eq!(tail.text.lines().count(), TAIL_LINES);
    }
    #[test]
    fn huge_unicode_line_is_bounded_and_disclosure_survives_updates() {
        let mut card = Card::default();
        assert!(!card.expanded);
        card.toggle();
        card.tail.append(&"λ".repeat(TAIL_CHARS + 1));
        assert!(card.expanded);
        assert!(card.tail.truncated);
        assert_eq!(card.tail.text.chars().count(), TAIL_CHARS);
        card.toggle();
        assert!(!card.expanded);
    }

    #[test]
    fn bytes_seen_tracks_cumulative_input_across_truncation() {
        // Every appended byte grows `bytes_seen`, even after the on-screen
        // tail clips to the 20-line / 16k-char budgets. This is what lets a
        // group summary sum "total output produced" instead of "bytes still
        // on screen"; the ZETA-125 review flagged the mismatch when the
        // group total silently under-counted after truncation.
        let mut tail = OutputTail::default();
        tail.append("first line\n");
        assert_eq!(tail.bytes_seen, 11);
        for n in 1..40 {
            tail.append(&format!("line {n}\n"));
        }
        assert!(tail.truncated);
        // After truncation the on-screen text is shorter, but bytes_seen
        // still reflects every byte we ever appended.
        assert!(tail.bytes_seen > tail.text.len());
        let before = tail.bytes_seen;
        tail.append("more");
        assert_eq!(tail.bytes_seen, before + 4);
    }
}
