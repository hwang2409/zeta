//! Pure vim editing state for the composer.

use std::ops::Range;

mod edit;
mod motion;

use edit::*;
use motion::*;

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
enum MotionShape {
    Charwise,
    Linewise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionInclusivity {
    Inclusive,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MotionResult {
    endpoint: usize,
    shape: MotionShape,
    inclusivity: MotionInclusivity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingMotion {
    Find { forward: bool, till: bool },
    G,
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
    operator_count: usize,
    motion_count: usize,
    desired_column: Option<usize>,
    register: String,
    register_shape: RegisterShape,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    insert_snapshot: Option<Snapshot>,
    insert_entry_cursor: Option<usize>,
    insert_transaction: bool,
    insert_changed: bool,
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
            operator_count: 0,
            motion_count: 0,
            desired_column: None,
            register: String::new(),
            register_shape: RegisterShape::Charwise,
            undo: Vec::new(),
            redo: Vec::new(),
            insert_snapshot: None,
            insert_entry_cursor: None,
            insert_transaction: false,
            insert_changed: false,
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
            if !self.insert_transaction {
                let snapshot = self.insert_snapshot.take().unwrap_or_else(|| {
                    let mut snapshot = self.snapshot();
                    snapshot.mode = Mode::Normal;
                    snapshot
                });
                self.undo.push(snapshot);
                self.insert_transaction = true;
                self.redo.clear();
            }
            self.insert_changed = true;
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
            if self.insert_changed {
                self.cursor = previous_char(&self.text, self.cursor);
            } else if let Some(entry_cursor) = self.insert_entry_cursor {
                self.cursor = entry_cursor;
            }
            self.mode = Mode::Normal;
            self.cursor = normal_cursor(&self.text, self.cursor);
            self.finish_insert_transaction();
            self.clear_pending();
            return true;
        }
        if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
            self.mode = Mode::Normal;
            self.visual_anchor = None;
            self.clear_pending();
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
            if is_count_key(key, self.motion_count) {
                self.push_motion_count(key);
                return true;
            }
            if is_operator(key) {
                if operator_key(operator) == key {
                    let count = self
                        .operator_count
                        .max(1)
                        .saturating_mul(self.motion_count.max(1));
                    self.apply_line_operator(operator, count);
                    self.clear_pending();
                    return true;
                }
                self.clear_pending();
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
            let count = if key == "G" && self.motion_count == 0 {
                0
            } else {
                self.operator_count
                    .max(1)
                    .saturating_mul(self.motion_count.max(1))
            };
            self.motion_count = 0;
            if self.apply_motion(key, count, Some(operator)) {
                self.pending_operator = None;
                self.operator_count = 0;
                return true;
            }
            self.clear_pending();
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
            self.operator_count = self.take_count().max(1);
            self.motion_count = 0;
            return true;
        }
        match key {
            "escape" => self.escape(),
            "i" => {
                self.begin_insert();
                true
            }
            "I" => {
                self.cursor = first_non_blank(&self.text, self.cursor);
                self.begin_insert();
                true
            }
            "a" => {
                let entry_cursor = self.cursor;
                self.cursor = next_char_same_line(&self.text, self.cursor);
                self.begin_insert_at(entry_cursor);
                true
            }
            "A" => {
                self.cursor = line_end(&self.text, self.cursor);
                self.begin_insert();
                true
            }
            "o" => {
                self.open_line(true);
                true
            }
            "O" => {
                self.open_line(false);
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
                let end = repeat_delete_right_same_line(&self.text, self.cursor, count);
                self.delete_range(self.cursor..end, RegisterShape::Charwise);
                true
            }
            "D" => {
                let count = self.take_count().max(1);
                let line = line_number(&self.text, self.cursor);
                let end = line_end(&self.text, line_start_n(&self.text, line + count - 1));
                self.delete_range(self.cursor..end, RegisterShape::Charwise);
                true
            }
            "C" => {
                let count = self.take_count().max(1);
                let line = line_number(&self.text, self.cursor);
                let end = line_end(&self.text, line_start_n(&self.text, line + count - 1));
                if self.delete_range(self.cursor..end, RegisterShape::Charwise) {
                    self.mode = Mode::Insert;
                    self.begin_insert_after_edit();
                }
                true
            }
            "s" => {
                let count = self.take_count().max(1);
                let end = repeat_delete_right_same_line(&self.text, self.cursor, count);
                if self.delete_range(self.cursor..end, RegisterShape::Charwise) {
                    self.begin_insert_after_edit();
                }
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
                self.motion_count = self.take_count();
                true
            }
            "f" | "F" | "t" | "T" => {
                self.pending_motion = Some(PendingMotion::Find {
                    forward: key == "f" || key == "t",
                    till: key == "t" || key == "T",
                });
                self.motion_count = self.take_count();
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
            self.motion_count = self.take_count();
            return true;
        }
        let count = self.take_count();
        self.apply_motion(key, count, None)
    }

    fn finish_pending_motion(&mut self, pending: PendingMotion, key: &str) -> bool {
        self.pending_motion = None;
        if let PendingMotion::G = pending {
            if key == "g" {
                let count = if self.pending_operator.is_some() {
                    self.operator_count
                        .max(1)
                        .saturating_mul(self.motion_count.max(1))
                } else {
                    self.motion_count.max(1)
                };
                self.motion_count = 0;
                let target = line_start_n(&self.text, count);
                if let Some(operator) = self.pending_operator.take() {
                    self.apply_operator(
                        operator,
                        MotionResult {
                            endpoint: target,
                            shape: MotionShape::Linewise,
                            inclusivity: MotionInclusivity::Inclusive,
                        },
                    );
                    self.operator_count = 0;
                } else {
                    self.cursor = if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                        target
                    } else {
                        normal_cursor(&self.text, target)
                    };
                    self.desired_column = None;
                }
                return true;
            }
            self.clear_pending();
            return true;
        }
        let PendingMotion::Find { forward, till } = pending else {
            return true;
        };
        let Some(target) = key.chars().next() else {
            return true;
        };
        let count = self
            .operator_count
            .max(1)
            .saturating_mul(self.motion_count.max(1));
        self.motion_count = 0;
        let Some(found) = find_char(&self.text, self.cursor, target, forward, count) else {
            self.clear_pending();
            return true;
        };
        let endpoint = if till && !forward {
            next_char(&self.text, found)
        } else {
            found
        };
        let result = MotionResult {
            endpoint,
            shape: MotionShape::Charwise,
            inclusivity: if till {
                MotionInclusivity::Exclusive
            } else {
                MotionInclusivity::Inclusive
            },
        };
        if let Some(operator) = self.pending_operator.take() {
            self.apply_operator(operator, result);
            self.operator_count = 0;
        } else {
            self.cursor = if till && forward {
                previous_char(&self.text, endpoint)
            } else if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                endpoint
            } else {
                normal_cursor(&self.text, endpoint)
            };
            self.desired_column = None;
        }
        true
    }

    fn apply_motion(&mut self, key: &str, count: usize, operator: Option<Operator>) -> bool {
        let motion_count = count.max(1);
        let result = match key {
            "h" => MotionResult {
                endpoint: repeat_left_same_line(&self.text, self.cursor, motion_count),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Exclusive,
            },
            "l" => MotionResult {
                endpoint: repeat_right_same_line(&self.text, self.cursor, motion_count),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Exclusive,
            },
            "j" => MotionResult {
                endpoint: self.vertical_move(motion_count as isize),
                shape: MotionShape::Linewise,
                inclusivity: MotionInclusivity::Inclusive,
            },
            "k" => MotionResult {
                endpoint: self.vertical_move(-(motion_count as isize)),
                shape: MotionShape::Linewise,
                inclusivity: MotionInclusivity::Inclusive,
            },
            "w" => MotionResult {
                endpoint: if operator == Some(Operator::Change) {
                    word_end(&self.text, self.cursor, motion_count)
                } else {
                    word_forward(&self.text, self.cursor, motion_count)
                },
                shape: MotionShape::Charwise,
                inclusivity: if operator == Some(Operator::Change) {
                    MotionInclusivity::Inclusive
                } else {
                    MotionInclusivity::Exclusive
                },
            },
            "b" => MotionResult {
                endpoint: word_backward(&self.text, self.cursor, motion_count),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Exclusive,
            },
            "e" => MotionResult {
                endpoint: word_end(&self.text, self.cursor, motion_count),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Inclusive,
            },
            "0" => MotionResult {
                endpoint: line_start(&self.text, self.cursor),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Exclusive,
            },
            "^" => MotionResult {
                endpoint: first_non_blank(&self.text, self.cursor),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Exclusive,
            },
            "$" => MotionResult {
                endpoint: line_end(&self.text, self.cursor),
                shape: MotionShape::Charwise,
                inclusivity: MotionInclusivity::Inclusive,
            },
            "G" => MotionResult {
                endpoint: if count == 0 {
                    self.text.rfind('\n').map_or(0, |newline| newline + 1)
                } else {
                    line_start_n(&self.text, count)
                },
                shape: MotionShape::Linewise,
                inclusivity: MotionInclusivity::Inclusive,
            },
            _ => return false,
        };
        if let Some(operator) = operator {
            self.apply_operator(operator, result);
        } else {
            self.cursor = if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
                result.endpoint.min(self.text.len())
            } else {
                normal_cursor(&self.text, result.endpoint)
            };
            if !matches!(key, "j" | "k") {
                self.desired_column = None;
            }
        }
        true
    }

    fn vertical_move(&mut self, lines: isize) -> usize {
        let current_start = line_start(&self.text, self.cursor);
        let current_column = self.cursor.saturating_sub(current_start);
        let desired = self.desired_column.unwrap_or(current_column);
        self.desired_column = Some(desired);
        let current_line = self.text[..current_start]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count();
        let target_line = if lines.is_negative() {
            current_line.saturating_sub(lines.unsigned_abs())
        } else {
            current_line.saturating_add(lines as usize)
        };
        let target_start = line_start_n(&self.text, target_line + 1);
        (target_start + desired).min(line_end(&self.text, target_start))
    }

    fn apply_visual_operator(&mut self, operator: Operator) {
        if self.mode == Mode::VisualLine {
            let Some(anchor) = self.visual_anchor else {
                return;
            };
            let range = linewise_range(&self.text, anchor, self.cursor);
            self.apply_range_operator(operator, range, RegisterShape::Linewise);
        } else {
            let Some(range) = self.selected_range() else {
                return;
            };
            self.apply_range_operator(operator, range, RegisterShape::Charwise);
        }
        self.mode = if operator == Operator::Change {
            Mode::Insert
        } else {
            Mode::Normal
        };
        self.visual_anchor = None;
        if operator == Operator::Change {
            self.begin_insert_after_edit();
        }
    }

    fn delete_range(&mut self, range: Range<usize>, shape: RegisterShape) -> bool {
        let start = clip_char_boundary(&self.text, range.start.min(self.text.len()));
        let end = clip_char_boundary(&self.text, range.end.min(self.text.len()));
        if start >= end {
            return false;
        }
        self.snapshot_before_edit();
        self.register = self.text[start..end].to_owned();
        self.register_shape = shape;
        self.text.replace_range(start..end, "");
        self.cursor = normal_cursor(&self.text, start.min(self.text.len()));
        true
    }

    fn apply_operator(&mut self, operator: Operator, motion: MotionResult) {
        let range = if motion.shape == MotionShape::Linewise {
            linewise_range(&self.text, self.cursor, motion.endpoint)
        } else {
            let (start, end) = if self.cursor <= motion.endpoint {
                (self.cursor, motion.endpoint)
            } else {
                (motion.endpoint, self.cursor)
            };
            let end = if motion.inclusivity == MotionInclusivity::Inclusive {
                inclusive_end(&self.text, end)
            } else {
                exclusive_operator_end(&self.text, start, end)
            };
            start..end
        };
        self.apply_range_operator(
            operator,
            range,
            if motion.shape == MotionShape::Linewise {
                RegisterShape::Linewise
            } else {
                RegisterShape::Charwise
            },
        );
        if operator == Operator::Change {
            self.mode = Mode::Insert;
            self.begin_insert_after_edit();
        }
    }

    fn apply_line_operator(&mut self, operator: Operator, count: usize) {
        let current_line = line_number(&self.text, self.cursor);
        let endpoint = line_start_n(&self.text, current_line.saturating_add(count.max(1) - 1));
        self.apply_operator(
            operator,
            MotionResult {
                endpoint,
                shape: MotionShape::Linewise,
                inclusivity: MotionInclusivity::Inclusive,
            },
        );
    }

    fn apply_range_operator(
        &mut self,
        operator: Operator,
        range: Range<usize>,
        shape: RegisterShape,
    ) {
        let start = clip_char_boundary(&self.text, range.start.min(self.text.len()));
        let end = clip_char_boundary(&self.text, range.end.min(self.text.len()));
        if start >= end {
            return;
        }
        self.register = self.text[start..end].to_owned();
        self.register_shape = shape;
        if operator == Operator::Yank {
            return;
        }
        let delete_start = if operator != Operator::Change
            && shape == RegisterShape::Linewise
            && start > 0
            && end == self.text.len()
        {
            start - 1
        } else {
            start
        };
        let delete_end = if operator == Operator::Change && shape == RegisterShape::Linewise {
            line_end(&self.text, start)
        } else {
            end
        };
        self.snapshot_before_edit();
        self.text.replace_range(delete_start..delete_end, "");
        self.cursor = normal_cursor(&self.text, delete_start.min(self.text.len()));
    }

    fn paste(&mut self, after: bool) {
        if self.register.is_empty() {
            return;
        }
        if self.mode == Mode::Visual || self.mode == Mode::VisualLine {
            let Some(range) = self.selected_range() else {
                return;
            };
            let start = clip_char_boundary(&self.text, range.start.min(self.text.len()));
            let end = clip_char_boundary(&self.text, range.end.min(self.text.len()));
            if start >= end {
                return;
            }
            self.snapshot_before_edit();
            self.text.replace_range(start..end, &self.register);
            self.cursor = normal_cursor(&self.text, start);
            return;
        }
        self.snapshot_before_edit();
        if self.register_shape == RegisterShape::Linewise {
            let start = line_start(&self.text, self.cursor);
            let end = line_end(&self.text, self.cursor);
            let insert_at = if after && end < self.text.len() {
                end + 1
            } else if after {
                end
            } else {
                start
            };
            let value = if after && insert_at == self.text.len() {
                format!("\n{}", self.register)
            } else if !self.register.ends_with('\n') {
                format!("{}\n", self.register)
            } else {
                self.register.clone()
            };
            self.text.insert_str(insert_at, &value);
            self.cursor = normal_cursor(&self.text, insert_at);
        } else {
            let insert_at = if after {
                next_char_same_line(&self.text, self.cursor)
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

    fn push_motion_count(&mut self, key: &str) {
        let digit = key.as_bytes()[0].saturating_sub(b'0') as usize;
        self.motion_count = self.motion_count.saturating_mul(10).saturating_add(digit);
    }

    fn take_count(&mut self) -> usize {
        std::mem::take(&mut self.count)
    }

    fn clear_pending(&mut self) {
        self.pending_operator = None;
        self.pending_motion = None;
        self.count = 0;
        self.operator_count = 0;
        self.motion_count = 0;
    }

    fn begin_insert(&mut self) {
        self.begin_insert_at(self.cursor);
    }

    fn begin_insert_at(&mut self, entry_cursor: usize) {
        let mut snapshot = self.snapshot();
        snapshot.cursor = entry_cursor;
        snapshot.mode = Mode::Normal;
        self.insert_snapshot = Some(snapshot);
        self.insert_entry_cursor = Some(entry_cursor);
        self.insert_transaction = false;
        self.insert_changed = false;
        self.mode = Mode::Insert;
        self.desired_column = None;
    }

    fn begin_insert_after_edit(&mut self) {
        self.insert_snapshot = None;
        self.insert_entry_cursor = Some(self.cursor);
        self.insert_transaction = true;
        self.insert_changed = false;
        self.redo.clear();
        self.desired_column = None;
    }

    fn finish_insert_transaction(&mut self) {
        self.insert_snapshot = None;
        self.insert_entry_cursor = None;
        self.insert_transaction = false;
        self.insert_changed = false;
    }

    fn open_line(&mut self, below: bool) {
        let start = line_start(&self.text, self.cursor);
        let end = line_end(&self.text, self.cursor);
        let indentation = self.text[start..end]
            .chars()
            .take_while(|character| character.is_whitespace())
            .collect::<String>();
        self.snapshot_before_edit();
        let insert_at = if below { end } else { start };
        let value = if below {
            format!("\n{}", indentation)
        } else {
            format!("{}\n", indentation)
        };
        self.text.insert_str(insert_at, &value);
        self.cursor = if below {
            insert_at + 1 + indentation.len()
        } else {
            insert_at + indentation.len()
        };
        self.mode = Mode::Insert;
        self.begin_insert_after_edit();
    }

    fn snapshot_before_edit(&mut self) {
        let mut snapshot = self.snapshot();
        snapshot.mode = Mode::Normal;
        self.undo.push(snapshot);
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
        self.desired_column = None;
        self.clear_pending();
        self.finish_insert_transaction();
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
    let end_start = line_start(text, anchor.max(current));
    let end = line_end(text, end_start);
    let end = if end < text.len() { end + 1 } else { end };
    start..end
}

fn inclusive_end(text: &str, endpoint: usize) -> usize {
    if endpoint >= text.len() || character_at(text, endpoint) == Some('\n') {
        endpoint
    } else {
        next_char(text, endpoint)
    }
}

fn exclusive_operator_end(text: &str, start: usize, endpoint: usize) -> usize {
    if endpoint > start && line_start(text, endpoint) == endpoint {
        line_end(text, start)
    } else {
        endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(text: &str) -> VimBuffer {
        let mut vim = VimBuffer::new(text);
        vim.escape();
        vim
    }

    fn type_text(vim: &mut VimBuffer, text: &str) {
        let mut value = vim.text().to_owned();
        value.insert_str(vim.cursor(), text);
        vim.sync_input(value, vim.cursor() + text.len());
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
    fn insert_entry_uses_first_non_blank_and_copies_indentation() {
        let mut vim = edit("  one\n  two");
        vim.handle_key("I");
        assert_eq!(vim.cursor(), 2);
        let mut vim = edit("  one");
        vim.handle_key("o");
        assert_eq!(vim.text(), "  one\n  ");
        assert_eq!(vim.cursor(), vim.text().len());
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
        assert_eq!(vim.text(), "");
    }

    #[test]
    fn counted_d_and_c_reach_the_end_of_the_counted_line() {
        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("2");
        vim.handle_key("D");
        assert_eq!(vim.text(), "\nthree");

        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("2");
        vim.handle_key("C");
        assert_eq!(vim.text(), "\nthree");
        assert_eq!(vim.mode(), Mode::Insert);
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
    fn operator_and_motion_counts_multiply() {
        let mut vim = edit("one two three four five six");
        for key in ["2", "d", "3", "w"] {
            vim.handle_key(key);
        }
        assert_eq!(vim.text(), "");
    }

    #[test]
    fn counted_g_and_gg_multiply_with_operator_counts() {
        let mut vim = edit("one\ntwo\nthree\nfour\nfive\nsix");
        for key in ["d", "3", "g", "g"] {
            vim.handle_key(key);
        }
        assert_eq!(vim.text(), "four\nfive\nsix");

        let mut vim = edit("one\ntwo\nthree\nfour\nfive\nsix");
        for key in ["2", "d", "3", "g", "g"] {
            vim.handle_key(key);
        }
        assert_eq!(vim.text(), "");
    }

    #[test]
    fn linewise_motion_and_zero_distance_delete_are_exact() {
        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("d");
        vim.handle_key("j");
        assert_eq!(vim.text(), "three");
        let mut vim = edit("abc");
        vim.handle_key("d");
        vim.handle_key("0");
        assert_eq!(vim.text(), "abc");
    }

    #[test]
    fn d_g_and_d_gg_delete_whole_lines() {
        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("j");
        vim.handle_key("d");
        vim.handle_key("G");
        assert_eq!(vim.text(), "one");
        let mut vim = edit("one\ntwo\nthree");
        vim.handle_key("j");
        vim.handle_key("d");
        vim.handle_key("g");
        vim.handle_key("g");
        assert_eq!(vim.text(), "three");
    }

    #[test]
    fn cw_is_ce_while_dw_includes_the_space() {
        let mut vim = edit("one two");
        vim.handle_key("c");
        vim.handle_key("w");
        assert_eq!(vim.text(), " two");
        let mut vim = edit("one two");
        vim.handle_key("d");
        vim.handle_key("w");
        assert_eq!(vim.text(), "two");
    }

    #[test]
    fn exclusive_word_operators_do_not_consume_a_following_newline() {
        let mut vim = edit("one\ntwo");
        vim.handle_key("d");
        vim.handle_key("w");
        assert_eq!(vim.text(), "\ntwo");

        let mut vim = edit("one\ntwo");
        vim.handle_key("y");
        vim.handle_key("w");
        assert_eq!(vim.register, "one");
    }

    #[test]
    fn counted_character_edits_stop_at_eol() {
        let mut vim = edit("abc\ndef");
        vim.handle_key("$");
        vim.handle_key("x");
        assert_eq!(vim.text(), "ab\ndef");
        let mut vim = edit("abc\ndef");
        vim.handle_key("2");
        vim.handle_key("s");
        assert_eq!(vim.text(), "c\ndef");
    }

    #[test]
    fn word_start_whitespace_and_punctuation_follow_vim_classes() {
        let mut vim = edit("one   two,three");
        vim.handle_key("3");
        vim.handle_key("l");
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], "two,three");
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], ",three");
    }

    #[test]
    fn word_motion_stops_on_punctuation_and_repeated_e_advances() {
        let mut vim = edit("foo.bar");
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], ".bar");
        vim.handle_key("w");
        assert_eq!(&vim.text()[vim.cursor()..], "bar");

        let mut vim = edit("foo bar");
        vim.handle_key("e");
        assert_eq!(&vim.text()[vim.cursor()..], "o bar");
        vim.handle_key("e");
        assert_eq!(&vim.text()[vim.cursor()..], "r");
    }

    #[test]
    fn linewise_change_keeps_an_empty_replacement_line() {
        let mut vim = edit("one\ntwo");
        vim.handle_key("c");
        vim.handle_key("c");
        type_text(&mut vim, "X");
        vim.escape();
        assert_eq!(vim.text(), "X\ntwo");
    }

    #[test]
    fn failed_find_cancels_a_pending_operator() {
        let mut vim = edit("abc\nz");
        vim.handle_key("d");
        vim.handle_key("t");
        vim.handle_key("z");
        vim.handle_key("l");
        assert_eq!(vim.text(), "abc\nz");
        assert_eq!(vim.cursor(), 1);
    }

    #[test]
    fn final_line_yyp_is_linewise_and_visual_p_replaces() {
        let mut vim = edit("abc");
        vim.handle_key("y");
        vim.handle_key("y");
        vim.handle_key("p");
        assert_eq!(vim.text(), "abc\nabc");
        let mut vim = edit("abcdef");
        vim.handle_key("y");
        vim.handle_key("l");
        vim.handle_key("y");
        vim.handle_key("0");
        vim.handle_key("v");
        vim.handle_key("l");
        vim.handle_key("p");
        assert_eq!(vim.text(), "acdef");
    }

    #[test]
    fn insert_sessions_undo_as_one_transaction() {
        let mut vim = edit("abc");
        vim.handle_key("c");
        vim.handle_key("w");
        type_text(&mut vim, "xy");
        vim.escape();
        assert_eq!(vim.text(), "xy");
        vim.handle_key("u");
        assert_eq!(vim.text(), "abc");
        vim.handle_key("x");
        assert!(!vim.redo());
    }

    #[test]
    fn visual_delete_undo_restores_normal_mode_without_an_anchor() {
        let mut vim = edit("abc");
        vim.handle_key("v");
        vim.handle_key("l");
        vim.handle_key("d");
        assert_eq!(vim.text(), "c");
        vim.handle_key("u");
        assert_eq!(vim.text(), "abc");
        assert_eq!(vim.mode(), Mode::Normal);
        assert_eq!(vim.selected_range(), None);
    }

    #[test]
    fn visual_change_and_insert_undo_as_one_normal_transaction() {
        let mut vim = edit("abc");
        vim.handle_key("v");
        vim.handle_key("l");
        vim.handle_key("c");
        type_text(&mut vim, "xy");
        vim.escape();
        assert_eq!(vim.text(), "xyc");
        vim.handle_key("u");
        assert_eq!(vim.text(), "abc");
        assert_eq!(vim.mode(), Mode::Normal);
        assert_eq!(vim.selected_range(), None);
    }

    #[test]
    fn insert_escape_keeps_a_entry_cursor_and_final_open_line() {
        let mut vim = edit("abc");
        vim.handle_key("a");
        vim.escape();
        assert_eq!(vim.cursor(), 0);

        let mut vim = edit("abc");
        vim.handle_key("o");
        vim.escape();
        assert_eq!(vim.text(), "abc\n");
        assert_eq!(vim.cursor(), vim.text().len());
        assert_eq!(vim.mode(), Mode::Normal);
    }

    #[test]
    fn one_based_g_and_vertical_motion_restore_the_preferred_column() {
        let mut vim = edit("12345\na\n1234");
        vim.handle_key("1");
        vim.handle_key("G");
        assert_eq!(vim.cursor(), 0);
        vim.handle_key("4");
        vim.handle_key("l");
        vim.handle_key("j");
        vim.handle_key("j");
        assert_eq!(vim.cursor(), 11);
        vim.handle_key("k");
        assert_eq!(vim.cursor(), 6);
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
