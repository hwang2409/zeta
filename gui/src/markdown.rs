//! Markdown render tree, parsed on commit with bounded syntax work off the UI thread.
use crate::appearance::Appearance;
use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag, TagEnd};
use std::{
    ops::Range,
    sync::{mpsc, Arc, LazyLock},
    time::{Duration, Instant},
};
use syntect::{
    easy::HighlightLines, highlighting::ThemeSet, parsing::SyntaxSet, util::LinesWithEndings,
};

const STREAM_PREVIEW_BYTES: usize = 8 * 1024;
const STREAM_PREVIEW_LINES: usize = 40;

const MAX_MARKDOWN_BYTES: usize = 128 * 1024;
const MAX_FENCE_BYTES: usize = 16 * 1024;
const MAX_SYNTAX_LINE_BYTES: usize = 1024;
const HIGHLIGHT_BUDGET: Duration = Duration::from_millis(20);

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Inline {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    Document,
    Paragraph,
    Heading(u8),
    List(Option<u64>),
    Item,
    Quote,
    Code(String),
    Rule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxRun {
    pub range: Range<usize>,
    pub light: u32,
    pub dark: u32,
}

impl SyntaxRun {
    pub fn color(&self, appearance: Appearance) -> u32 {
        appearance.syntax_color(match appearance {
            Appearance::Light => self.light,
            Appearance::Dark => self.dark,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub kind: BlockKind,
    pub spans: Vec<Inline>,
    pub children: Vec<Block>,
    // Foreground-only runs: theme backgrounds cannot enter the render tree.
    pub syntax: Vec<SyntaxRun>,
}

impl Block {
    fn new(kind: BlockKind) -> Self {
        Self {
            kind,
            spans: vec![],
            children: vec![],
            syntax: vec![],
        }
    }
    pub fn text(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Markdown {
    // While streaming, retain only the visible tail. The server's committed
    // message supplies the full source. Render elements share this snapshot.
    pub source: Arc<str>,
    pub root: Option<Block>,
    pub preview_truncated: bool,
    #[cfg(test)]
    preview_allocation_bytes: usize,
}

impl From<String> for Markdown {
    fn from(source: String) -> Self {
        let root = (source.len() <= MAX_MARKDOWN_BYTES)
            .then(|| parse(&source, Instant::now() + HIGHLIGHT_BUDGET));
        Self {
            source: source.into(),
            root,
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
            root: None,
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
        self.root = None;
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

fn parse(source: &str, deadline: Instant) -> Block {
    let mut stack = vec![Block::new(BlockKind::Document)];
    let mut bold = 0;
    let mut italic = 0;
    for event in Parser::new(source) {
        let kind = match &event {
            Event::Start(Tag::Paragraph) => Some(BlockKind::Paragraph),
            Event::Start(Tag::Heading { level, .. }) => Some(BlockKind::Heading(*level as u8)),
            Event::Start(Tag::List(start)) => Some(BlockKind::List(*start)),
            Event::Start(Tag::Item) => Some(BlockKind::Item),
            Event::Start(Tag::BlockQuote(_)) => Some(BlockKind::Quote),
            Event::Start(Tag::CodeBlock(kind)) => Some(BlockKind::Code(match kind {
                CodeBlockKind::Fenced(info) => info.split_whitespace().next().unwrap_or("").into(),
                CodeBlockKind::Indented => String::new(),
            })),
            _ => None,
        };
        if let Some(kind) = kind {
            stack.push(Block::new(kind));
            continue;
        }
        match event {
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::List(_)
                | TagEnd::Item
                | TagEnd::BlockQuote(_)
                | TagEnd::CodeBlock,
            ) => {
                let mut block = stack.pop().expect("balanced markdown tags");
                if let BlockKind::Code(language) = &block.kind {
                    block.syntax = highlight(&block.text(), language, deadline);
                }
                stack.last_mut().unwrap().children.push(block);
            }
            Event::Start(Tag::Strong) => bold += 1,
            Event::End(TagEnd::Strong) => bold -= 1,
            Event::Start(Tag::Emphasis) => italic += 1,
            Event::End(TagEnd::Emphasis) => italic -= 1,
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                stack.last_mut().unwrap().spans.push(Inline {
                    text: text.into_string(),
                    bold: bold > 0,
                    italic: italic > 0,
                    code: false,
                });
            }
            Event::Code(text) => stack.last_mut().unwrap().spans.push(Inline {
                text: text.into_string(),
                code: true,
                bold: bold > 0,
                italic: italic > 0,
            }),
            Event::SoftBreak | Event::HardBreak => stack.last_mut().unwrap().spans.push(Inline {
                text: "\n".into(),
                ..Default::default()
            }),
            Event::Rule => stack
                .last_mut()
                .unwrap()
                .children
                .push(Block::new(BlockKind::Rule)),
            _ => {}
        }
    }
    stack.pop().unwrap()
}

static SYNTAX: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
static THEMES: LazyLock<ThemeSet> = LazyLock::new(ThemeSet::load_defaults);

struct HighlightRequest {
    text: String,
    language: String,
    deadline: Instant,
    reply: mpsc::SyncSender<Vec<SyntaxRun>>,
}

// One owner and one queued fence: a slow regex cannot create more threads or an
// unbounded backlog. Both syntax loading and regex execution stay off the UI.
static HIGHLIGHTER: LazyLock<Option<mpsc::SyncSender<HighlightRequest>>> = LazyLock::new(|| {
    let (sender, receiver) = mpsc::sync_channel::<HighlightRequest>(1);
    std::thread::Builder::new()
        .name("syntax-highlight".into())
        .spawn(move || {
            for request in receiver {
                if Instant::now() < request.deadline {
                    let runs = syntax_colors(&request.text, &request.language, request.deadline);
                    let _ = request.reply.try_send(runs);
                }
            }
        })
        .ok()
        .map(|_| sender)
});

fn highlight(text: &str, language: &str, deadline: Instant) -> Vec<SyntaxRun> {
    if text.len() > MAX_FENCE_BYTES
        || text.lines().any(|line| line.len() > MAX_SYNTAX_LINE_BYTES)
        || Instant::now() >= deadline
    {
        return Vec::new();
    }
    let Some(worker) = HIGHLIGHTER.as_ref() else {
        return Vec::new();
    };
    highlight_on(worker, text, language, deadline)
}

fn highlight_on(
    worker: &mpsc::SyncSender<HighlightRequest>,
    text: &str,
    language: &str,
    deadline: Instant,
) -> Vec<SyntaxRun> {
    let (reply, result) = mpsc::sync_channel(1);
    if worker
        .try_send(HighlightRequest {
            text: text.to_owned(),
            language: language.to_owned(),
            deadline,
            reply,
        })
        .is_err()
    {
        return Vec::new();
    }
    // This deadline covers ALL fences in the message, including both palettes.
    // Checking only between highlight_line calls cannot stop a single slow regex.
    result
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .unwrap_or_default()
}

fn syntax_colors(text: &str, language: &str, deadline: Instant) -> Vec<SyntaxRun> {
    let syntax = SYNTAX
        .find_syntax_by_token(language)
        .unwrap_or_else(|| SYNTAX.find_syntax_plain_text());
    let colors = |theme: &str| {
        let mut highlighter = HighlightLines::new(syntax, &THEMES.themes[theme]);
        let mut offset = 0;
        let mut runs = Vec::new();
        for line in LinesWithEndings::from(text) {
            if Instant::now() >= deadline {
                return None;
            }
            let Ok(parts) = highlighter.highlight_line(line, &SYNTAX) else {
                // A syntax failure falls back to plain text for the whole fence.
                return None;
            };
            if Instant::now() >= deadline {
                return None;
            }
            for (style, text) in parts {
                let color = style.foreground;
                runs.push((
                    offset..offset + text.len(),
                    (u32::from(color.r) << 16) | (u32::from(color.g) << 8) | u32::from(color.b),
                ));
                offset += text.len();
            }
        }
        Some(runs)
    };
    let (Some(light), Some(dark)) = (colors("InspiredGitHub"), colors("base16-ocean.dark")) else {
        return Vec::new();
    };
    // Theme changes can coalesce different token boundaries. Merge their endpoints.
    let mut boundaries: Vec<_> = light
        .iter()
        .chain(&dark)
        .flat_map(|(r, _)| [r.start, r.end])
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
        .windows(2)
        .map(|pair| SyntaxRun {
            range: pair[0]..pair[1],
            light: light
                .get(light.partition_point(|(range, _)| range.end <= pair[0]))
                .map_or(0x202024, |(_, c)| *c),
            dark: dark
                .get(dark.partition_point(|(range, _)| range.end <= pair[0]))
                .map_or(0xededf0, |(_, c)| *c),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::appearance::contrast;
    #[test]
    fn maps_headings_lists_inline_code_and_fences() {
        let source = "# Title\n\n## Second\n\nplain **bold** `code`\n\n3. first\n4. second\n   - nested\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n";
        let root = parse(source, Instant::now());
        let blocks = &root.children;
        assert_eq!(blocks[0].kind, BlockKind::Heading(1));
        assert_eq!(blocks[1].kind, BlockKind::Heading(2));
        assert!(blocks[2].spans.iter().any(|s| s.code && s.text == "code"));
        assert!(blocks[2].spans.iter().any(|s| s.bold && s.text == "bold"));
        assert_eq!(blocks[3].kind, BlockKind::List(Some(3)));
        assert_eq!(blocks[3].children.len(), 2);
        assert_eq!(
            blocks[3].children[1].children[0].kind,
            BlockKind::List(None)
        );
        let code = &blocks[4];
        assert_eq!(code.kind, BlockKind::Code("rust".into()));
        assert_eq!(code.text(), "fn main() { println!(\"hi\"); }\n");
        // Check colors independently of queue contention and cold-start timing.
        let syntax = syntax_colors(
            &code.text(),
            "rust",
            Instant::now() + Duration::from_secs(5),
        );
        assert!(syntax.len() > 1);
        assert_eq!(syntax.first().unwrap().range.start, 0);
        assert_eq!(syntax.last().unwrap().range.end, code.text().len());
        for mode in [Appearance::Light, Appearance::Dark] {
            assert!(syntax
                .iter()
                .all(|run| contrast(run.color(mode), mode.palette().background) >= 4.5));
        }
        assert!(syntax
            .iter()
            .any(|run| run.color(Appearance::Light) != run.color(Appearance::Dark)));
    }
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
            assert!(doc.root.is_none());
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

    #[test]
    fn streaming_defers_parse_until_commit() {
        let mut doc = Markdown::streaming("```python\nprint(".into());
        doc.push_str("'hello')\n```\n\nafter");
        assert!(doc.root.is_none());
        let committed = Markdown::from(doc.source.as_ref());
        let root = committed.root.unwrap();
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].text(), "print('hello')\n");
    }

    #[test]
    fn giant_fence_and_message_fall_back_without_syntax_work() {
        let text = "let x = 1;\n".repeat(MAX_FENCE_BYTES / 10);
        let doc = Markdown::from(format!("```rust\n{text}```"));
        let code = &doc.root.as_ref().unwrap().children[0];
        assert_eq!(code.text(), text);
        assert!(code.syntax.is_empty());
        let source = "x".repeat(MAX_MARKDOWN_BYTES + 1);
        let doc = Markdown::from(source.clone());
        assert!(doc.root.is_none());
        assert_eq!(doc.source.as_ref(), source);
    }

    #[test]
    fn pathological_syntax_uses_plain_text() {
        // Deeply nested, unterminated syntax exceeds the per-line work bound
        // even though the entire fence is below the byte limit.
        let text = format!("{}!\n", "(".repeat(MAX_SYNTAX_LINE_BYTES + 1));
        assert!(text.len() < MAX_FENCE_BYTES);
        let doc = Markdown::from(format!("```javascript\n{text}```"));
        let code = &doc.root.as_ref().unwrap().children[0];
        assert_eq!(code.text(), text);
        assert!(code.syntax.is_empty());
    }

    #[test]
    fn stalled_highlighter_cannot_block_the_caller_or_grow_the_queue() {
        // A worker that cannot finish even one regex must not hold the UI.
        let (worker, receiver) = mpsc::sync_channel(1);
        let started = Instant::now();
        assert!(
            highlight_on(&worker, "fn main() {}", "rust", started + HIGHLIGHT_BUDGET).is_empty()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        // The first request is still queued. A second request must fail fast,
        // regardless of its deadline, rather than waiting for queue capacity.
        assert!(highlight_on(
            &worker,
            "next",
            "rust",
            Instant::now() + Duration::from_secs(5)
        )
        .is_empty());
        assert!(started.elapsed() < Duration::from_secs(1));
        let pending = receiver.try_recv().unwrap();
        assert_eq!(pending.text, "fn main() {}");
        assert!(receiver.try_recv().is_err());
        assert!(pending.reply.try_send(Vec::new()).is_err());
    }

    #[test]
    fn expired_budget_preserves_fences_as_plain_text() {
        let root = parse("```rust\nfn main() {}\n```", Instant::now());
        assert_eq!(root.children[0].text(), "fn main() {}\n");
        assert!(root.children[0].syntax.is_empty());
    }
}
