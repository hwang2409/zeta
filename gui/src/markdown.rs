//! Bounded streaming source snapshots. GPUI Kit owns parsing and rendering.
use std::sync::Arc;

const STREAM_PREVIEW_BYTES: usize = 8 * 1024;
const STREAM_PREVIEW_LINES: usize = 40;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Markdown {
    // While streaming, retain only the visible tail. The server's committed
    // message supplies the full source. Render elements share this snapshot.
    pub source: Arc<str>,
    pub preview_truncated: bool,
    #[cfg(test)]
    preview_allocation_bytes: usize,
}

impl From<String> for Markdown {
    fn from(source: String) -> Self {
        Self {
            source: source.into(),
            preview_truncated: false,
            #[cfg(test)]
            preview_allocation_bytes: 0,
        }
    }
}
impl From<&str> for Markdown {
    fn from(source: &str) -> Self {
        source.to_owned().into()
    }
}
impl Markdown {
    pub fn streaming(source: String) -> Self {
        let mut doc = Self {
            source: Arc::from(""),
            preview_truncated: false,
            #[cfg(test)]
            preview_allocation_bytes: 0,
        };
        doc.push_str(&source);
        doc
    }

    pub fn push_str(&mut self, delta: &str) {
        // Slice the delta before copying: a single huge event must also have
        // bounded allocation and layout work. UTF-8 boundaries remain intact.
        let delta_tail = preview_tail(delta);
        let previous = if delta_tail.len() == delta.len() {
            preview_tail(&self.source)
        } else {
            ""
        };
        let mut next = String::with_capacity(previous.len() + delta_tail.len());
        next.push_str(previous);
        next.push_str(delta_tail);
        let tail = preview_tail(&next);
        self.preview_truncated |= previous.len() < self.source.len()
            || delta_tail.len() < delta.len()
            || tail.len() < next.len();
        #[cfg(test)]
        {
            self.preview_allocation_bytes = next.capacity() + tail.len();
        }
        self.source = Arc::from(tail);
    }
}

fn preview_tail(text: &str) -> &str {
    let mut start = text.len().saturating_sub(STREAM_PREVIEW_BYTES);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let tail = &text[start..];
    // A trailing newline terminates the last line, rather than adding a line.
    let line_start = tail
        .strip_suffix('\n')
        .unwrap_or(tail)
        .rmatch_indices('\n')
        .nth(STREAM_PREVIEW_LINES - 1)
        .map_or(0, |(offset, _)| offset + 1);
    &tail[line_start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_snapshots_bound_work_and_share_render_storage() {
        let mut doc = Markdown::streaming(String::new());
        let delta = "λ".repeat(512);
        // More than five MiB arrives in small deltas. No update can retain or
        // copy an ever-growing prefix; the render snapshot is shared by Arc.
        for _ in 0..6000 {
            let previous = doc.source.clone();
            doc.push_str(&delta);
            assert!(doc.source.len() <= STREAM_PREVIEW_BYTES);
            assert!(doc.preview_allocation_bytes <= 3 * STREAM_PREVIEW_BYTES);
            let snapshot = doc.source.clone();
            assert!(Arc::ptr_eq(&doc.source, &snapshot));
            assert_eq!(snapshot.as_ptr(), doc.clone().source.as_ptr());
            // The previous frame stays immutable after the next delta.
            assert!(previous.chars().all(|ch| ch == 'λ'));
        }
        assert!(doc.preview_truncated);
        doc.push_str(&"α\n".repeat(3 * 1024 * 1024));
        assert!(doc.source.len() <= STREAM_PREVIEW_BYTES);
        assert!(doc.preview_allocation_bytes <= 3 * STREAM_PREVIEW_BYTES);
        assert_eq!(doc.source.lines().count(), STREAM_PREVIEW_LINES);
        assert!(doc.source.ends_with("α\n"));
        doc.push_str(&"\n".repeat(STREAM_PREVIEW_BYTES * 2));
        assert!(doc.preview_allocation_bytes <= 3 * STREAM_PREVIEW_BYTES);
        assert_eq!(doc.source.lines().count(), STREAM_PREVIEW_LINES);
        let full = "complete response\n".repeat(10_000);
        let committed = Markdown::from(full.clone());
        assert_eq!(committed.source.as_ref(), full);
        assert!(!committed.preview_truncated);
    }
}
