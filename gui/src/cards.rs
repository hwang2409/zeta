//! Bounded output and disclosure state shared by tool and delegated receipts.
pub const TAIL_LINES: usize = 20;
const TAIL_CHARS: usize = 16_000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputTail {
    pub text: String,
    pub truncated: bool,
}

impl OutputTail {
    pub fn append(&mut self, text: &str) {
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
}
