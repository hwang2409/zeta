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
