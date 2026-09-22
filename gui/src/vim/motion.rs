pub(super) fn char_boundaries(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
}
pub(super) fn clip_char_boundary(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .take_while(|&candidate| candidate <= offset)
        .last()
        .unwrap_or(0)
}

pub(super) fn next_char(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .find(|&candidate| candidate > offset)
        .unwrap_or(text.len())
}

pub(super) fn previous_char(text: &str, offset: usize) -> usize {
    char_boundaries(text)
        .take_while(|&candidate| candidate < offset)
        .last()
        .unwrap_or(0)
}

pub(super) fn normal_cursor(text: &str, offset: usize) -> usize {
    if text.is_empty() {
        0
    } else {
        let offset = clip_char_boundary(text, offset.min(text.len()));
        if offset == text.len() {
            return if text.ends_with('\n') {
                offset
            } else {
                previous_char(text, offset)
            };
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

pub(super) fn line_start(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())]
        .rfind('\n')
        .map_or(0, |index| index + 1)
}

pub(super) fn line_end(text: &str, offset: usize) -> usize {
    text[offset.min(text.len())..]
        .find('\n')
        .map_or(text.len(), |index| offset + index)
}

pub(super) fn line_number(text: &str, offset: usize) -> usize {
    text[..line_start(text, offset)]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        .saturating_add(1)
}

pub(super) fn line_start_n(text: &str, line: usize) -> usize {
    if line <= 1 {
        return 0;
    }
    text.match_indices('\n')
        .map(|(index, _)| index + 1)
        .nth(line - 2)
        .unwrap_or_else(|| text.rfind('\n').map_or(0, |index| index + 1))
}

pub(super) fn first_non_blank(text: &str, offset: usize) -> usize {
    let start = line_start(text, offset);
    text[start..line_end(text, start)]
        .find(|ch: char| !ch.is_whitespace())
        .map_or(start, |index| start + index)
}

pub(super) fn repeat_left_same_line(text: &str, mut offset: usize, count: usize) -> usize {
    let start = line_start(text, offset);
    for _ in 0..count {
        let next = previous_char(text, offset);
        if next < start || next == offset {
            break;
        }
        offset = next;
    }
    offset
}

pub(super) fn next_char_same_line(text: &str, offset: usize) -> usize {
    let next = next_char(text, offset);
    if next > line_end(text, offset) {
        offset
    } else {
        next
    }
}

pub(super) fn repeat_right_same_line(text: &str, mut offset: usize, count: usize) -> usize {
    let end = line_end(text, offset);
    for _ in 0..count {
        let next = next_char(text, offset);
        if next >= end || next == offset {
            break;
        }
        offset = next;
    }
    offset
}

pub(super) fn repeat_delete_right_same_line(text: &str, mut offset: usize, count: usize) -> usize {
    let end = line_end(text, offset);
    for _ in 0..count {
        let next = next_char(text, offset);
        if next > end || next == offset {
            break;
        }
        offset = next;
    }
    offset
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordClass {
    Word,
    Punctuation,
    Whitespace,
}

fn word_class(character: char) -> WordClass {
    if character.is_whitespace() {
        WordClass::Whitespace
    } else if character.is_alphanumeric() || character == '_' {
        WordClass::Word
    } else {
        WordClass::Punctuation
    }
}

pub(super) fn character_at(text: &str, offset: usize) -> Option<char> {
    text[offset.min(text.len())..].chars().next()
}

pub(super) fn word_forward(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        let Some(class) = character_at(text, offset).map(word_class) else {
            break;
        };
        if class == WordClass::Whitespace {
            while character_at(text, offset)
                .is_some_and(|character| word_class(character) == WordClass::Whitespace)
            {
                offset = next_char(text, offset);
            }
        } else {
            while character_at(text, offset).map(word_class) == Some(class) {
                offset = next_char(text, offset);
            }
        }
        while character_at(text, offset)
            .is_some_and(|character| word_class(character) == WordClass::Whitespace)
        {
            offset = next_char(text, offset);
        }
    }
    offset
}

pub(super) fn word_backward(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        offset = previous_char(text, offset);
        while character_at(text, offset)
            .is_some_and(|character| word_class(character) == WordClass::Whitespace)
        {
            if offset == 0 {
                break;
            }
            offset = previous_char(text, offset);
        }
        let Some(class) = character_at(text, offset).map(word_class) else {
            break;
        };
        while offset > 0 {
            let previous = previous_char(text, offset);
            if character_at(text, previous).map(word_class) != Some(class) {
                break;
            }
            offset = previous;
        }
    }
    offset
}

pub(super) fn word_end(text: &str, mut offset: usize, count: usize) -> usize {
    for _ in 0..count {
        while character_at(text, offset)
            .is_some_and(|character| word_class(character) == WordClass::Whitespace)
        {
            offset = next_char(text, offset);
        }
        let Some(class) = character_at(text, offset).map(word_class) else {
            break;
        };
        if character_at(text, next_char(text, offset)).map(word_class) != Some(class) {
            offset = next_char(text, offset);
            while character_at(text, offset)
                .is_some_and(|character| word_class(character) == WordClass::Whitespace)
            {
                offset = next_char(text, offset);
            }
        }
        let Some(class) = character_at(text, offset).map(word_class) else {
            break;
        };
        while character_at(text, next_char(text, offset)).map(word_class) == Some(class) {
            offset = next_char(text, offset);
        }
    }
    offset
}

pub(super) fn find_char(
    text: &str,
    offset: usize,
    target: char,
    forward: bool,
    count: usize,
) -> Option<usize> {
    let start = line_start(text, offset);
    let end = line_end(text, offset);
    let mut matches = text.char_indices().filter(|(index, character)| {
        *index >= start
            && *index < end
            && *character == target
            && if forward {
                *index > offset
            } else {
                *index < offset
            }
    });
    if forward {
        matches.nth(count.max(1) - 1).map(|(index, _)| index)
    } else {
        matches
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .nth(count.max(1) - 1)
            .map(|(index, _)| index)
    }
}
