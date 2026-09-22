//! Pure vim editing state for the composer.

use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Insert,
    Normal,
    Visual,
    VisualLine,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Insert => "INSERT",
            Self::Normal => "NORMAL",
            Self::Visual => "VISUAL",
            Self::VisualLine => "V-LINE",
        }
    }
}

pub fn intercepts_key(enabled: bool, mode: Mode) -> bool {
    enabled && mode != Mode::Insert
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Delete,
    Change,
    Yank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingMotion {
    Find { forward: bool, till: bool },
    G,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Snapshot {
    text: String,
    cursor: usize,
    mode: Mode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VimBuffer {
    text: String,
    cursor: usize,
    mode: Mode,
    visual_anchor: Option<usize>,
    pending_operator: Option<Operator>,
    pending_motion: Option<PendingMotion>,
    count: usize,
    register: String,
    register_linewise: bool,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
}

impl VimBuffer {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            cursor: normal_cursor(&text, 0),
            text,
            mode: Mode::Insert,
            visual_anchor: None,
            pending_operator: None,
            pending_motion: None,
            count: 0,
            register: String::new(),
            register_linewise: false,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn selected_range(&self) -> Option<Range<usize>> {
        self.visual_anchor.map(|anchor| {
            let current = self.cursor;
            if self.mode == Mode::VisualLine {
                linewise_range(&self.text, anchor, current)
            } else {
                charwise_range(&self.text, anchor, current)
            }
        })
    }

    pub fn sync_input(&mut self, text: impl Into<String>, cursor: usize) {
        let text = text.into();
        if self.mode == Mode::Insert && self.text != text {
            self.undo.push(self.snapshot());
            self.redo.clear();
        }
        self.text = text;
        self.cursor = clip_char_boundary(&self.text, cursor.min(self.text.len()));
        if self.mode != Mode::Insert {
            self.cursor = normal_cursor(&self.text, self.cursor);
        }
    }

    pub fn handle_key(&mut self, key: &str) -> bool {
        match self.mode {
            Mode::Insert => self.handle_insert(key),
            Mode::Normal => self.handle_normal(key),
            Mode::Visual | Mode::VisualLine => self.handle_visual(key),
        }
    }

    pub fn escape(&mut self) -> bool {
        if self.mode == Mode::Insert {
            self.mode = Mode::Normal;
            self.cursor = normal_cursor(&self.text, self.cursor);
            self.pending_operator = None;
            self.pending_motion = None;
            self.count = 0;
            return true;
        }
        if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
            self.mode = Mode::Normal;
            self.visual_anchor = None;
            self.pending_operator = None;
            self.pending_motion = None;
            self.count = 0;
            return true;
        }
        false
    }

    pub fn undo(&mut self) -> bool {
        let Some(snapshot) = self.undo.pop() else {
            return false;
        };
        self.redo.push(self.snapshot());
        self.restore(snapshot);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(snapshot) = self.redo.pop() else {
            return false;
        };
        self.undo.push(self.snapshot());
        self.restore(snapshot);
        true
    }

    fn handle_insert(&mut self, key: &str) -> bool {
        if key == "escape" {
            return self.escape();
        }
        false
    }

    fn handle_normal(&mut self, key: &str) -> bool {
        if let Some(pending) = self.pending_motion {
            return self.finish_pending_motion(pending, key);
        }
        if let Some(operator) = self.pending_operator {
            if is_count_key(key, self.count) {
                self.push_count(key);
                return true;
            }
            if is_operator(key) {
                if operator_key(operator) == key {
                    let count = self.take_count();
                    self.apply_line_operator(operator, count);
                    self.pending_operator = None;
                    return true;
                }
                self.pending_operator = None;
                self.count = 0;
                return true;
            }
            if matches!(key, "f" | "F" | "t" | "T" | "g") {
                self.pending_motion = Some(match key {
                    "f" => PendingMotion::Find {
                        forward: true,
                        till: false,
                    },
                    "F" => PendingMotion::Find {
                        forward: false,
                        till: false,
                    },
                    "t" => PendingMotion::Find {
                        forward: true,
                        till: true,
                    },
                    "T" => PendingMotion::Find {
                        forward: false,
                        till: true,
                    },
                    _ => PendingMotion::G,
                });
                return true;
            }
            let count = self.take_count();
            if self.apply_motion(key, count, Some(operator)) {
                self.pending_operator = None;
                return true;
            }
            self.pending_operator = None;
            return true;
        }
        if is_count_key(key, self.count) {
            self.push_count(key);
            return true;
        }
        if is_operator(key) {
            self.pending_operator = Some(match key {
                "d" => Operator::Delete,
                "c" => Operator::Change,
                _ => Operator::Yank,
            });
            return true;
        }
        match key {
            "escape" => self.escape(),
            "i" => {
                self.mode = Mode::Insert;
                true
            }
            "I" => {
                self.cursor = line_start(&self.text, self.cursor);
                self.mode = Mode::Insert;
                true
            }
            "a" => {
                self.cursor = next_char(&self.text, self.cursor);
                self.mode = Mode::Insert;
                true
            }
            "A" => {
                self.cursor = line_end(&self.text, self.cursor);
                self.mode = Mode::Insert;
                true
            }
            "o" => {
                self.snapshot_before_edit();
                let end = line_end(&self.text, self.cursor);
                self.text.insert(end, '\n');
                self.cursor = end + 1;
                self.mode = Mode::Insert;
                true
            }
            "O" => {
                self.snapshot_before_edit();
                let start = line_start(&self.text, self.cursor);
                self.text.insert(start, '\n');
                self.cursor = start;
                self.mode = Mode::Insert;
                true
            }
            "v" => {
                self.mode = Mode::Visual;
                self.visual_anchor = Some(self.cursor);
                true
            }
            "V" => {
                self.mode = Mode::VisualLine;
                self.visual_anchor = Some(self.cursor);
                true
            }
            "x" => {
                let count = self.take_count().max(1);
                let end = repeat_right(&self.text, self.cursor, count);
                self.delete_range(self.cursor..end);
                true
            }
            "D" => {
                let count = self.take_count().max(1);
                let end = if count == 1 {
                    line_end(&self.text, self.cursor)
                } else {
                    repeat_right(&self.text, self.cursor, count)
                };
                self.delete_range(self.cursor..end);
                true
            }
            "C" => {
                let count = self.take_count().max(1);
                let end = if count == 1 {
                    line_end(&self.text, self.cursor)
                } else {
                    repeat_right(&self.text, self.cursor, count)
                };
                self.delete_range(self.cursor..end);
                self.mode = Mode::Insert;
                true
            }
            "s" => {
                let count = self.take_count().max(1);
                let end = repeat_right(&self.text, self.cursor, count);
                self.delete_range(self.cursor..end);
                self.mode = Mode::Insert;
                true
            }
            "p" | "P" => {
                self.paste(key == "p");
                true
            }
            "u" => self.undo(),
            "ctrl-r" => self.redo(),
            "g" => {
                self.pending_motion = Some(PendingMotion::G);
                true
            }
            "f" | "F" | "t" | "T" => {
                self.pending_motion = Some(PendingMotion::Find {
                    forward: key == "f" || key == "t",
                    till: key == "t" || key == "T",
                });
                true
            }
            _ => {
                let count = self.take_count();
                self.apply_motion(key, count, None)
            }
        }
    }

    fn handle_visual(&mut self, key: &str) -> bool {
        if key == "escape" {
            return self.escape();
        }
        if let Some(pending) = self.pending_motion {
            return self.finish_pending_motion(pending, key);
        }
        if is_count_key(key, self.count) {
            self.push_count(key);
            return true;
        }
        if is_operator(key) {
            let operator = match key {
                "d" => Operator::Delete,
                "c" => Operator::Change,
                _ => Operator::Yank,
            };
            self.apply_visual_operator(operator);
            return true;
        }
        if key == "p" || key == "P" {
            self.paste(key == "p");
            self.mode = Mode::Normal;
            self.visual_anchor = None;
            return true;
        }
        if matches!(key, "f" | "F" | "t" | "T" | "g") {
            self.pending_motion = Some(match key {
                "f" => PendingMotion::Find {
                    forward: true,
                    till: false,
                },
                "F" => PendingMotion::Find {
                    forward: false,
                    till: false,
                },
                "t" => PendingMotion::Find {
                    forward: true,
                    till: true,
                },
                "T" => PendingMotion::Find {
                    forward: false,
                    till: true,
                },
                _ => PendingMotion::G,
            });
            return true;
        }
        let count = self.take_count();
        self.apply_motion(key, count, None)
    }

    fn finish_pending_motion(&mut self, pending: PendingMotion, key: &str) -> bool {
        self.pending_motion = None;
        if let PendingMotion::G = pending {
            if key == "g" {
                let count = self.take_count();
                let target = if count > 0 {
                    line_start_n(&self.text, count.saturating_sub(1))
                } else {
                    line_start(&self.text, 0)
                };
                if let Some(operator) = self.pending_operator.take() {
                    self.apply_operator(operator, self.cursor, target);
                } else {
                    self.cursor = if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                        target
                    } else {
                        normal_cursor(&self.text, target)
                    };
                }
                return true;
            }
            return true;
        }
        let PendingMotion::Find { forward, till } = pending else {
            return true;
        };
        let Some(target) = key.chars().next() else {
            return true;
        };
        let count = self.take_count();
        let found = find_char(&self.text, self.cursor, target, forward, count);
        if let Some(mut offset) = found {
            if till {
                offset = if forward {
                    previous_char(&self.text, offset)
                } else {
                    next_char(&self.text, offset)
                };
            }
            if let Some(operator) = self.pending_operator.take() {
                let (start, end) = if forward {
                    (self.cursor, next_char(&self.text, offset))
                } else {
                    (offset, next_char(&self.text, self.cursor))
                };
                self.apply_operator(operator, start, end);
            } else {
                self.cursor = if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                    offset
                } else {
                    normal_cursor(&self.text, offset)
                };
            }
        }
        true
    }

    fn apply_motion(&mut self, key: &str, count: usize, operator: Option<Operator>) -> bool {
        let motion_count = count.max(1);
        let target = match key {
            "h" => repeat_left(&self.text, self.cursor, motion_count),
            "l" => repeat_right(&self.text, self.cursor, motion_count),
            "j" => vertical_move(&self.text, self.cursor, motion_count as isize),
            "k" => vertical_move(&self.text, self.cursor, -(motion_count as isize)),
            "w" => word_forward(&self.text, self.cursor, motion_count),
            "b" => word_backward(&self.text, self.cursor, motion_count),
            "e" => word_end(&self.text, self.cursor, motion_count),
            "0" => line_start(&self.text, self.cursor),
            "^" => first_non_blank(&self.text, self.cursor),
            "$" => line_end(&self.text, self.cursor),
            "G" if count == 0 => self.text.rfind('\n').map_or(0, |newline| newline + 1),
            "G" => line_start_n(&self.text, count.saturating_sub(1)),
            _ => return false,
        };
        let target = if operator == Some(Operator::Delete)
            || operator == Some(Operator::Change)
            || operator == Some(Operator::Yank)
        {
            if key == "e" {
                next_char(&self.text, target)
            } else {
                target
            }
        } else {
            target
        };
        if let Some(operator) = operator {
            self.apply_operator(operator, self.cursor, target);
        } else {
            self.cursor = normal_cursor(&self.text, target);
            if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                self.cursor = target.min(self.text.len());
            }
        }
        true
    }

    fn apply_visual_operator(&mut self, operator: Operator) {
        if self.mode == Mode::VisualLine {
            let Some(anchor) = self.visual_anchor else {
                return;
            };
            self.apply_operator(operator, anchor, self.cursor);
        } else {
            let Some(range) = self.selected_range() else {
                return;
            };
            self.apply_operator(operator, range.start, range.end);
        }
        self.mode = if operator == Operator::Change {
            Mode::Insert
        } else {
            Mode::Normal
        };
        self.visual_anchor = None;
    }

    fn delete_range(&mut self, range: Range<usize>) {
        let start = clip_char_boundary(&self.text, range.start.min(self.text.len()));
        let end = clip_char_boundary(&self.text, range.end.min(self.text.len()));
        if start >= end {
            return;
        }
        self.snapshot_before_edit();
        self.register = self.text[start..end].to_owned();
        self.register_linewise = self.register.ends_with('\n');
        self.text.replace_range(start..end, "");
        self.cursor = normal_cursor(&self.text, start.min(self.text.len()));
    }

    fn apply_operator(&mut self, operator: Operator, from: usize, to: usize) {
        let (mut start, mut end) = if from <= to { (from, to) } else { (to, from) };
        if self.mode != Mode::Visual && end == start {
            end = next_char(&self.text, end);
        }
        if self.mode == Mode::VisualLine {
            let range = linewise_range(&self.text, start, end);
            start = range.start;
            end = range.end;
        }
        if start == end {
            return;
        }
        if operator == Operator::Yank {
            self.register = self.text[start..end].to_owned();
            self.register_linewise =
                self.mode == Mode::VisualLine || self.text[start..end].ends_with('\n');
            return;
        }
        self.snapshot_before_edit();
        self.register = self.text[start..end].to_owned();
        self.register_linewise = self.mode == Mode::VisualLine || self.register.ends_with('\n');
        self.text.replace_range(start..end, "");
        self.cursor = normal_cursor(&self.text, start.min(self.text.len()));
        if operator == Operator::Change {
            self.mode = Mode::Insert;
        }
    }

    fn apply_line_operator(&mut self, operator: Operator, count: usize) {
        let start = line_start(&self.text, self.cursor);
        let end = line_end_n(&self.text, self.cursor, count.max(1));
        self.apply_operator(operator, start, end);
    }

    fn paste(&mut self, after: bool) {
        if self.register.is_empty() {
            return;
        }
        self.snapshot_before_edit();
        if self.register_linewise {
            let line = if after {
                line_end(&self.text, self.cursor)
            } else {
                line_start(&self.text, self.cursor)
            };
            let insert_at = if after && line < self.text.len() {
                line + 1
            } else {
                line
            };
            self.text.insert_str(insert_at, &self.register);
            self.cursor = normal_cursor(&self.text, insert_at);
        } else {
            let insert_at = if after {
                next_char(&self.text, self.cursor)
            } else {
                self.cursor
            };
            self.text.insert_str(insert_at, &self.register);
            self.cursor = normal_cursor(&self.text, insert_at);
        }
    }

    fn push_count(&mut self, key: &str) {
        let digit = key.as_bytes()[0].saturating_sub(b'0') as usize;
        self.count = self.count.saturating_mul(10).saturating_add(digit);
    }

    fn take_count(&mut self) -> usize {
        std::mem::take(&mut self.count)
    }

    fn snapshot_before_edit(&mut self) {
        self.undo.push(self.snapshot());
        self.redo.clear();
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            text: self.text.clone(),
            cursor: self.cursor,
            mode: self.mode,
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.text = snapshot.text;
        self.cursor = snapshot.cursor.min(self.text.len());
        self.mode = snapshot.mode;
        self.visual_anchor = None;
        self.pending_operator = None;
        self.pending_motion = None;
        self.count = 0;
    }
}

fn is_operator(key: &str) -> bool {
    matches!(key, "d" | "c" | "y")
}

fn operator_key(operator: Operator) -> &'static str {
    match operator {
        Operator::Delete => "d",
        Operator::Change => "c",
        Operator::Yank => "y",
    }
}

fn is_count_key(key: &str, count: usize) -> bool {
    key.len() == 1 && key.as_bytes()[0].is_ascii_digit() && (key != "0" || count > 0)
}

fn char_boundaries(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
}

fn clip_char_boundary(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .take_while(|&candidate| candidate <= offset)
        .last()
        .unwrap_or(0)
}

fn next_char(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .find(|&candidate| candidate > offset)
        .unwrap_or(text.len())
}

fn previous_char(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .take_while(|&candidate| candidate < offset)
        .last()
        .unwrap_or(0)
}

fn normal_cursor(text: &str, offset: usize) -> usize {
    if text.is_empty() {
        0
    } else {
        let offset = clip_char_boundary(text, offset.min(text.len()));
        if offset == text.len() {
            return previous_char(text, offset);
        }
        let is_newline = text[offset..]
            .chars()
            .next()
            .is_some_and(|character| character == '\n');
        let is_empty_line = is_newline && (offset == 0 || text[..offset].ends_with('\n'));
        if is_newline && !is_empty_line {
            previous_char(text, offset)
        } else {
            offset
        }
    }
}

fn line_start(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())]
        .rfind('\n')
        .map_or(0, |index| index + 1)
}

fn line_end(text: &str, offset: usize) -> usize {
    text[offset.min(text.len())..]
        .find('\n')
        .map_or(text.len(), |index| offset + index)
}

fn line_start_n(text: &str, line: usize) -> usize {
    text.match_indices('\n')
        .map(|(index, _)| index + 1)
        .nth(line)
        .unwrap_or_else(|| text.rfind('\n').map_or(0, |index| index + 1))
}

fn line_end_n(text: &str, offset: usize, count: usize) -> usize {
    let mut end = line_end(text, offset);
    for _ in 1..count {
        if end == text.len() {
            break;
        }
        end = line_end(text, end + 1);
    }
    if end < text.len() {
        end + 1
    } else {
        end
    }
}

fn first_non_blank(text: &str, offset: usize) -> usize {
    let start = line_start(text, offset);
    text[start..line_end(text, start)]
        .find(|ch: char| !ch.is_whitespace())
        .map_or(start, |index| start + index)
}

fn repeat_left(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        offset = previous_char(text, offset);
    }
    offset
}

fn repeat_right(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        offset = next_char(text, offset);
    }
    normal_cursor(text, offset)
}

fn vertical_move(text: &str, offset: usize, lines: isize) -> usize {
    let current_start = line_start(text, offset);
    let column = offset.saturating_sub(current_start);
    let current_line = text[..current_start]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count() as isize;
    let target_line = (current_line + lines).max(0) as usize;
    let target_start = line_start_n(text, target_line);
    (target_start + column).min(line_end(text, target_start))
}

fn word_forward(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        while offset < text.len()
            && text[offset..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = next_char(text, offset);
        }
        while offset < text.len()
            && text[offset..]
                .chars()
                .next()
                .is_some_and(|ch| !ch.is_whitespace())
        {
            offset = next_char(text, offset);
        }
        while offset < text.len()
            && text[offset..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = next_char(text, offset);
        }
    }
    offset.min(text.len())
}

fn word_backward(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        offset = previous_char(text, offset);
        while offset > 0
            && text[offset..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = previous_char(text, offset);
        }
        while offset > 0
            && !text[previous_char(text, offset)..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = previous_char(text, offset);
        }
    }
    offset
}

fn word_end(text: &str, mut offset: usize, count: usize) -> usize {
    for index in 0..count {
        if index > 0
            || text[offset..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = word_forward(text, offset, 1);
        }
        while next_char(text, offset) < text.len()
            && !text[next_char(text, offset)..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            offset = next_char(text, offset);
        }
    }
    offset
}

fn find_char(
    text: &str,
    offset: usize,
    target: char,
    forward: bool,
    count: usize,
) -> Option<usize> {
    let iter = text.char_indices().filter(|(index, _)| {
        if forward {
            *index > offset
        } else {
            *index < offset
        }
    });
    let matches = if forward {
        iter.collect::<Vec<_>>()
    } else {
        iter.collect::<Vec<_>>().into_iter().rev().collect()
    };
    matches
        .into_iter()
        .filter(|(_, ch)| *ch == target)
        .nth(count.max(1) - 1)
        .map(|(index, _)| index)
}

fn charwise_range(text: &str, anchor: usize, current: usize) -> Range<usize> {
    let (start, end) = if anchor <= current {
        (anchor, current)
    } else {
        (current, anchor)
    };
    start..next_char(text, end)
}

fn linewise_range(text: &str, anchor: usize, current: usize) -> Range<usize> {
    let start = line_start(text, anchor.min(current));
    let end = line_end_n(text, anchor.max(current), 1);
    start..end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(text: &str) -> VimBuffer {
        let mut vim = VimBuffer::new(text);
        vim.escape();
        vim
    }

    #[test]
    fn starts_insert_and_escape_enters_normal() {
        let mut vim = VimBuffer::new("hello");
        assert_eq!(vim.mode(), Mode::Insert);
        assert!(vim.escape());
        assert_eq!(vim.mode(), Mode::Normal);
    }

    #[test]
    fn insert_entries_cover_cursor_and_line_edges() {
        let mut vim = edit("one\ntwo");
        vim.handle_key("0");
        vim.handle_key("a");
        assert_eq!(vim.mode(), Mode::Insert);
        assert_eq!(vim.cursor(), 1);
        vim.escape();
        vim.handle_key("A");
        assert_eq!(vim.cursor(), 3);
        vim.escape();
        vim.handle_key("o");
        assert_eq!(vim.text(), "one\n\ntwo");
        vim.escape();
        vim.handle_key("O");
        assert_eq!(vim.text(), "one\n\n\ntwo");
    }

    #[test]
    fn motions_support_counts_words_lines_and_find() {
        let mut vim = edit("one two three\nfour");
        vim.handle_key("2");
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], "three\nfour");
        vim.handle_key("0");
        vim.handle_key("f");
        vim.handle_key("t");
        assert_eq!(&vim.text()[vim.cursor()..], "two three\nfour");
        vim.handle_key("G");
        assert_eq!(&vim.text()[vim.cursor()..], "four");
    }

    #[test]
    fn motions_move_across_characters_and_word_edges() {
        let mut vim = edit("one two");
        vim.handle_key("l");
        assert_eq!(vim.cursor(), 1);
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], "two");
        vim.handle_key("e");
        assert_eq!(&vim.text()[vim.cursor()..], "o");
        vim.handle_key("b");
        assert_eq!(vim.cursor(), 4);
    }

    #[test]
    fn doubled_and_counted_operators_fill_register() {
        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("2");
        vim.handle_key("d");
        vim.handle_key("d");
        assert_eq!(vim.text(), "three");
        vim.handle_key("P");
        assert_eq!(vim.text(), "one\ntwo\nthree");
    }

    #[test]
    fn visual_and_visual_line_operations_work() {
        let mut vim = edit("one two\nthree");
        vim.handle_key("0");
        vim.handle_key("v");
        vim.handle_key("2");
        vim.handle_key("l");
        assert_eq!(vim.mode(), Mode::Visual);
        vim.handle_key("y");
        assert_eq!(vim.mode(), Mode::Normal);
        vim.handle_key("$");
        vim.handle_key("p");
        assert_eq!(vim.text(), "one twoone\nthree");
        vim.handle_key("V");
        vim.handle_key("d");
        assert_eq!(vim.text(), "three");
    }

    #[test]
    fn undo_redo_restore_text_and_mode() {
        let mut vim = edit("abc");
        vim.handle_key("x");
        assert_eq!(vim.text(), "bc");
        assert!(vim.undo());
        assert_eq!(vim.text(), "abc");
        assert!(vim.redo());
        assert_eq!(vim.text(), "bc");
    }

    #[test]
    fn counted_line_operator_uses_the_count() {
        let mut vim = edit("abcdef");
        vim.handle_key("2");
        vim.handle_key("D");
        assert_eq!(vim.text(), "cdef");
    }

    #[test]
    fn change_operator_enters_insert_mode() {
        let mut vim = edit("abc");
        vim.handle_key("c");
        vim.handle_key("l");
        assert_eq!(vim.text(), "bc");
        assert_eq!(vim.mode(), Mode::Insert);
    }

    #[test]
    fn false_mode_can_be_a_noop_at_integration_boundary() {
        let mut vim = VimBuffer::new("abc");
        vim.sync_input("abc", 0);
        assert_eq!(vim.mode(), Mode::Insert);
        assert_eq!(vim.text(), "abc");
        assert!(!intercepts_key(false, Mode::Normal));
        assert!(!intercepts_key(true, Mode::Insert));
        assert!(intercepts_key(true, Mode::Normal));
    }
}
